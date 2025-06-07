package com.example.ffmpegvideoplayer;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.HardwareBuffer;
import android.os.Build;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.opengl.GLES20;
import android.opengl.GLES30;
import android.util.Log;


import androidx.annotation.RequiresApi;

import androidx.opengl.EGLExt;
import androidx.opengl.EGLSyncKHR;
import androidx.opengl.EGLImageKHR;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class OpenGLImageProcessor {

    private static final String TAG = "OpenGLImageProcessor";

    private EGLDisplay eglDisplay = EGL14.EGL_NO_DISPLAY;
    private EGLContext eglContext = EGL14.EGL_NO_CONTEXT;
    private EGLSurface eglSurface = EGL14.EGL_NO_SURFACE;

    private int programHandle;
    private int positionHandle;
    private int texCoordHandle;
    private int inputTextureHandle;
    private int texelSizeHandle;

    private int[] fboHandle = new int[1];
    private int[] renderTextureHandle = new int[1]; // A texture used as a target for glEGLImageTargetTexture2DOES
    private int currentInputTextureId = 0;

    // --- Zero-Copy Pipeline Fields ---
    private static final int PIPELINE_DEPTH = 3; // Triple buffering for the pipeline
    private boolean useZeroCopyPath = false;

    // Each slot in the pipeline has a bitmap, its corresponding EGLImage, and a fence
    private EGLImageKHR[] eglImages;
    private FrameData[] pipelineSlots;
    private int pipelineIndex = 0; // The next available slot to queue a frame into
    private BlockingQueue<FrameData> processedFrames; // Queue for frames that are ready for consumption

    // --- Bitmap Pooling Fields ---
    private static final int BITMAP_POOL_SIZE = PIPELINE_DEPTH + 2; // Pool should be slightly larger than pipeline depth
    private BlockingQueue<Bitmap> availableBitmaps;
    private List<Bitmap> allCreatedBitmapsInPool;

    private FloatBuffer vertexBuffer;
    private FloatBuffer texCoordBuffer;

    private int outputWidth;
    private int outputHeight;

    private Context context;

    private static class FrameData {
        Bitmap bitmap;
        EGLSyncKHR fence;
    }

    private static final float[] VERTICES = {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
    private static final float[] TEX_COORDS = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

    public OpenGLImageProcessor(Context context) {
        this.context = context.getApplicationContext();
        vertexBuffer = ByteBuffer.allocateDirect(VERTICES.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        vertexBuffer.put(VERTICES).position(0);

        texCoordBuffer = ByteBuffer.allocateDirect(TEX_COORDS.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        texCoordBuffer.put(TEX_COORDS).position(0);
    }

    public boolean setup(int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;

        if (!initEGL()) return false;

        String vertexShaderCode = readShaderFromRawResource(R.raw.vertex_shader);
        String fragmentShaderCode = readShaderFromRawResource(R.raw.fragment_shader);
        programHandle = createProgram(vertexShaderCode, fragmentShaderCode);
        if (programHandle == 0) return false;

        GLES20.glUseProgram(programHandle);
        positionHandle = GLES20.glGetAttribLocation(programHandle, "a_position");
        texCoordHandle = GLES20.glGetAttribLocation(programHandle, "a_coord");
        inputTextureHandle = GLES20.glGetUniformLocation(programHandle, "u_Texture");
        texelSizeHandle = GLES20.glGetUniformLocation(programHandle, "u_TexelSize");

        // The new path requires GLES3 and API 26+
        useZeroCopyPath = (GLES30.glGetString(GLES30.GL_VERSION) != null) && (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O);

        if (useZeroCopyPath) {
            if (!setupHardwareBitmapPipeline()) {
                Log.e(TAG, "Failed to setup zero-copy pipeline. Disabling.");
                useZeroCopyPath = false;
            } else {
                Log.i(TAG, "Zero-copy EGLImage pipeline setup successfully.");
            }
        }

        if (!useZeroCopyPath) {
            // Fallback path setup if needed (e.g., traditional PBO or synchronous)
            Log.w(TAG, "Zero-copy path not available or failed. App may have performance issues.");
            return false; // For this highly optimized case, we fail if zero-copy isn't possible.
        }

        // Setup reusable input texture
        int[] tempTex = new int[1];
        GLES20.glGenTextures(1, tempTex, 0);
        currentInputTextureId = tempTex[0];
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, currentInputTextureId);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, inputWidth, inputHeight, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0);

        // Setup FBO. It will be re-targeted to different EGLImage textures.
        GLES20.glGenFramebuffers(1, fboHandle, 0);
        GLES20.glGenTextures(1, renderTextureHandle, 0); // This texture is just a handle, it will be defined by the EGLImage

        return true;
    }

    @RequiresApi(Build.VERSION_CODES.O)
    private boolean setupHardwareBitmapPipeline() {
        availableBitmaps = new ArrayBlockingQueue<>(BITMAP_POOL_SIZE);
        allCreatedBitmapsInPool = new ArrayList<>(BITMAP_POOL_SIZE);
        eglImages = new EGLImageKHR[BITMAP_POOL_SIZE];
        pipelineSlots = new FrameData[PIPELINE_DEPTH];
        processedFrames = new ArrayBlockingQueue<>(PIPELINE_DEPTH);

        Log.i(TAG, "Setting up HARDWARE Bitmap pool with size: " + BITMAP_POOL_SIZE);
        for (int i = 0; i < BITMAP_POOL_SIZE; i++) {
            try {
                // 1. Create a HardwareBuffer first with the correct usage flags.
                HardwareBuffer hwBuffer = HardwareBuffer.create(outputWidth, outputHeight, HardwareBuffer.RGBA_8888, 1,
                        HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE | HardwareBuffer.USAGE_GPU_COLOR_OUTPUT);

                // 2. Wrap it in an immutable Bitmap. The Bitmap takes ownership of the buffer.
                Bitmap bmp = Bitmap.wrapHardwareBuffer(hwBuffer, null);
                if (bmp == null) {
                    Log.e(TAG, "Failed to wrap HardwareBuffer in Bitmap.");
                    hwBuffer.close(); // Clean up if wrap failed
                    return false;
                }
                allCreatedBitmapsInPool.add(bmp);

                // 3. Create the EGLImage from the same HardwareBuffer.
                eglImages[i] = EGLExt.eglCreateImageFromHardwareBuffer(eglDisplay, hwBuffer);
                if (eglImages[i] == null) {
                    Log.e(TAG, "Failed to create EGLImage for bitmap " + i);
                    bmp.recycle(); // This should also close the underlying HardwareBuffer
                    return false;
                }

                // The HardwareBuffer is now managed by the Bitmap and EGLImage, so we don't close it here.
                availableBitmaps.offer(bmp);

            } catch (Exception e) {
                Log.e(TAG, "Exception while pre-allocating hardware bitmap or EGLImage " + i, e);
                return false;
            }
        }

        for (int i = 0; i < PIPELINE_DEPTH; i++) {
            pipelineSlots[i] = new FrameData();
        }

        Log.i(TAG, "Hardware Bitmap and EGLImage pool pre-allocated successfully.");
        return true;
    }

    public void queueFrame(ByteBuffer inputBuffer, int inputWidth, int inputHeight) {
        if (!useZeroCopyPath) return;

        // 1. Wait for the slot we are about to use to be free.
        FrameData slot = pipelineSlots[pipelineIndex];
        if (slot.fence != null) {
            int waitResult = EGLExt.eglClientWaitSyncKHR(eglDisplay, slot.fence, EGLExt.EGL_SYNC_FLUSH_COMMANDS_BIT_KHR, EGLExt.EGL_FOREVER_KHR);
            if (waitResult == EGLExt.EGL_FALSE) {
                Log.e(TAG, "Failed to wait for fence on slot " + pipelineIndex);
                return;
            }
            EGLExt.eglDestroySyncKHR(eglDisplay, slot.fence);
            slot.fence = null;
            // The bitmap used in this slot is now fully processed and can be consumed.
            processedFrames.offer(slot);
        }

        // 2. Prepare for rendering this frame
        try {
            slot.bitmap = availableBitmaps.take(); // Get a fresh bitmap to render into
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return;
        }

        // Find the EGLImage corresponding to the bitmap
        EGLImageKHR eglImage = null;
        for(int i=0; i<allCreatedBitmapsInPool.size(); i++){
            if(allCreatedBitmapsInPool.get(i) == slot.bitmap){
                eglImage = eglImages[i];
                break;
            }
        }
        if(eglImage == null){
            Log.e(TAG, "Could not find EGLImage for the bitmap. This should not happen.");
            return;
        }

        // 3. Upload input texture
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, currentInputTextureId);
        GLES20.glTexSubImage2D(GLES20.GL_TEXTURE_2D, 0, 0, 0, inputWidth, inputHeight, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, inputBuffer);

        // 4. Bind FBO to the EGLImage-backed texture
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, renderTextureHandle[0]);
        EGLExt.glEGLImageTargetTexture2DOES(GLES20.GL_TEXTURE_2D, eglImage);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboHandle[0]);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, renderTextureHandle[0], 0);

        // 5. Draw
        GLES20.glViewport(0, 0, outputWidth, outputHeight);
        GLES20.glUseProgram(programHandle);
        GLES20.glEnableVertexAttribArray(positionHandle);
        GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer);
        GLES20.glEnableVertexAttribArray(texCoordHandle);
        GLES20.glVertexAttribPointer(texCoordHandle, 2, GLES20.GL_FLOAT, false, 0, texCoordBuffer);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, currentInputTextureId);
        GLES20.glUniform1i(inputTextureHandle, 0);
        GLES20.glUniform2f(texelSizeHandle, 1.0f / inputWidth, 1.0f / inputHeight);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        // 6. Insert fence
        slot.fence = EGLExt.eglCreateSyncKHR(eglDisplay, EGLExt.EGL_SYNC_FENCE_KHR, null);
        GLES20.glFlush();

        // 7. Advance pipeline index
        pipelineIndex = (pipelineIndex + 1) % PIPELINE_DEPTH;
    }

    public Bitmap getProcessedFrame() {
        FrameData frameData = processedFrames.poll();
        if (frameData != null) {
            return frameData.bitmap;
        }
        return null; // Non-blocking
    }

    public void releaseBitmapToPool(Bitmap bitmap) {
        if (bitmap != null && availableBitmaps != null) {
            if (allCreatedBitmapsInPool.contains(bitmap)) {
                if (!availableBitmaps.offer(bitmap)) {
                    Log.e(TAG, "Failed to offer bitmap back to pool.");
                }
            } else {
                Log.w(TAG, "Attempted to release a bitmap that was not from the pool.");
                if (!bitmap.isRecycled()) {
                    bitmap.recycle();
                }
            }
        }
    }

    public void release() {
        // Wait for all pipeline slots to finish
        if (useZeroCopyPath) {
            for (int i = 0; i < PIPELINE_DEPTH; i++) {
                FrameData slot = pipelineSlots[i];
                if (slot != null && slot.fence != null) {
                    EGLExt.eglClientWaitSyncKHR(eglDisplay, slot.fence, EGLExt.EGL_SYNC_FLUSH_COMMANDS_BIT_KHR, EGLExt.EGL_FOREVER_KHR);
                    EGLExt.eglDestroySyncKHR(eglDisplay, slot.fence);
                }
            }
        }

        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
            if (programHandle != 0) GLES20.glDeleteProgram(programHandle);
            if (renderTextureHandle[0] != 0) GLES20.glDeleteTextures(1, renderTextureHandle, 0);
            if (currentInputTextureId != 0) GLES20.glDeleteTextures(1, new int[]{currentInputTextureId}, 0);
            if (fboHandle[0] != 0) GLES20.glDeleteFramebuffers(1, fboHandle, 0);

            if (eglImages != null) {
                for (EGLImageKHR image : eglImages) {
                    if (image != null) {
                        EGLExt.eglDestroyImageKHR(eglDisplay, image);
                    }
                }
            }

            if (allCreatedBitmapsInPool != null) {
                for (Bitmap bmp : allCreatedBitmapsInPool) {
                    if (bmp != null && !bmp.isRecycled()) bmp.recycle();
                }
                allCreatedBitmapsInPool.clear();
            }

            if (eglSurface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(eglDisplay, eglSurface);
            if (eglContext != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(eglDisplay, eglContext);
            EGL14.eglTerminate(eglDisplay);
        }
        eglDisplay = EGL14.EGL_NO_DISPLAY;
    }

    // --- Utility Methods ---
    private boolean initEGL() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        int[] version = new int[2];
        EGL14.eglInitialize(eglDisplay, version, 0, version, 1);
        int[] attribList = {EGL14.EGL_RED_SIZE, 8, EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8, EGL14.EGL_ALPHA_SIZE, 8, EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT, EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT, EGL14.EGL_NONE};
        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfigs = new int[1];
        EGL14.eglChooseConfig(eglDisplay, attribList, 0, configs, 0, configs.length, numConfigs, 0);
        int[] contextAttribs = {EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE};
        eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, contextAttribs, 0);
        if (eglContext == EGL14.EGL_NO_CONTEXT) { // Fallback to GLES 2
            contextAttribs[1] = 2;
            eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, contextAttribs, 0);
        }
        int[] pbufferAttribs = {EGL14.EGL_WIDTH, outputWidth, EGL14.EGL_HEIGHT, outputHeight, EGL14.EGL_NONE};
        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, configs[0], pbufferAttribs, 0);
        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);
        return true;
    }

    private int createProgram(String vertexSource, String fragmentSource) {
        int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource);
        int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
        int program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vertexShader);
        GLES20.glAttachShader(program, fragmentShader);
        GLES20.glLinkProgram(program);
        GLES20.glDeleteShader(vertexShader);
        GLES20.glDeleteShader(fragmentShader);
        return program;
    }

    private int loadShader(int type, String shaderCode) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, shaderCode);
        GLES20.glCompileShader(shader);
        return shader;
    }

    private String readShaderFromRawResource(int resourceId) {
        StringBuilder shaderSource = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(context.getResources().openRawResource(resourceId)))) {
            String line;
            while ((line = reader.readLine()) != null) {
                shaderSource.append(line).append("\n");
            }
        } catch (IOException e) {
            Log.e(TAG, "Could not read shader: " + e.getMessage());
            return null;
        }
        return shaderSource.toString();
    }
}