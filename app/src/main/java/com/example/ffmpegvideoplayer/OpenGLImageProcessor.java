package com.example.ffmpegvideoplayer;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas; // Added for placeholder drawing
import android.os.Build;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.opengl.GLES20;
import android.opengl.GLES30; // For PBOs
import android.opengl.GLUtils;
import android.util.Log;
import android.util.Size;

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
    private EGLConfig eglConfig;
    private EGLSurface eglSurface = EGL14.EGL_NO_SURFACE;

    private int programHandle;
    private int positionHandle;
    private int texCoordHandle;
    private int inputTextureHandle;
    private int texelSizeHandle;

    private int[] fboHandle = new int[1];
    private int[] outputTextureHandle = new int[1];
    private int currentInputTextureId = 0; // For reusable input texture
    private ByteBuffer readbackBuffer = null; // For reusable readback buffer (fallback if PBOs not used)

    // PBO related fields
    private static final int PBO_COUNT = 7; // Number of PBOs for pipelining
    private int[] pboIds = null;
    private int pboReadIndex = 0;  // Index of PBO for glReadPixels
    private int pboMapIndex = -1;   // Index of PBO to map and read from CPU
    private int pboBufferSize = 0;
    private boolean isGLES3 = false; // Flag to indicate if GLES 3.0 context is available

    // Bitmap pooling related fields
    private static final int BITMAP_POOL_SIZE = 7; // Example pool size
    private BlockingQueue<Bitmap> availableBitmaps;
    private List<Bitmap> allCreatedBitmapsInPool; // To track all bitmaps created for the pool for final release

    private FloatBuffer vertexBuffer;
    private FloatBuffer texCoordBuffer;

    private int outputWidth;
    private int outputHeight;

    private Context context;

    // Vertex coordinates for a full-screen quad
    private static final float[] VERTICES = {
            -1.0f, -1.0f, // Bottom Left
             1.0f, -1.0f, // Bottom Right
            -1.0f,  1.0f, // Top Left
             1.0f,  1.0f  // Top Right
    };

    // Texture coordinates
    private static final float[] TEX_COORDS = {
            0.0f, 0.0f, // Bottom Left
            1.0f, 0.0f, // Bottom Right
            0.0f, 1.0f, // Top Left
            1.0f, 1.0f  // Top Right
    };

    public OpenGLImageProcessor(Context context) {
        this.context = context.getApplicationContext();
        vertexBuffer = ByteBuffer.allocateDirect(VERTICES.length * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
        vertexBuffer.put(VERTICES).position(0);

        texCoordBuffer = ByteBuffer.allocateDirect(TEX_COORDS.length * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
        texCoordBuffer.put(TEX_COORDS).position(0);
    }

    public boolean setup(int inputWidth, int inputHeight, int outputWidth, int outputHeight) { // Added inputWidth, inputHeight
        this.outputWidth = outputWidth;
        this.outputHeight = outputHeight;

        if (!initEGL()) {
            Log.e(TAG, "EGL initialization failed.");
            return false;
        }

        String vertexShaderCode = readShaderFromRawResource(R.raw.vertex_shader);
        String fragmentShaderCode = readShaderFromRawResource(R.raw.fragment_shader);

        if (vertexShaderCode == null || fragmentShaderCode == null) {
            Log.e(TAG, "Failed to read shader code.");
            release();
            return false;
        }

        programHandle = createProgram(vertexShaderCode, fragmentShaderCode);
        if (programHandle == 0) {
            Log.e(TAG, "Failed to create GL program.");
            release();
            return false;
        }

        GLES20.glUseProgram(programHandle);

        positionHandle = GLES20.glGetAttribLocation(programHandle, "a_position");
        texCoordHandle = GLES20.glGetAttribLocation(programHandle, "a_coord");
        inputTextureHandle = GLES20.glGetUniformLocation(programHandle, "u_Texture");
        texelSizeHandle = GLES20.glGetUniformLocation(programHandle, "u_TexelSize");

        if (positionHandle == -1 || texCoordHandle == -1 || inputTextureHandle == -1 || texelSizeHandle == -1) {
            Log.e(TAG, "Failed to get shader variable locations.");
            release();
            return false;
        }

        if (!setupFramebuffer(outputWidth, outputHeight)) { // This will also setup readbackBuffer
            Log.e(TAG, "Failed to setup framebuffer.");
            release();
            return false;
        }

        // Setup reusable input texture
        int[] tempTex = new int[1];
        GLES20.glGenTextures(1, tempTex, 0);
        currentInputTextureId = tempTex[0];
        if (currentInputTextureId == 0) {
            Log.e(TAG, "Failed to generate input texture ID.");
            release();
            return false;
        }
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, currentInputTextureId);
        // Pre-allocate texture storage for input texture
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, inputWidth, inputHeight, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        if (checkGlError("Setup Input Texture - glTexImage2D pre-allocation")) {
            release();
            return false;
        }
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0); // Unbind
        if (checkGlError("Setup Input Texture - parameters")) {
            release();
            return false;
        }

        if (isGLES3) {
            if (!setupPBOs()) {
                Log.w(TAG, "Failed to setup PBOs. PBO optimization will be disabled.");
                isGLES3 = false; // Fallback to non-PBO path
            } else {
                Log.i(TAG, "PBOs setup successfully.");
            }
        }

        if (!setupBitmapPool()) {
            Log.e(TAG, "Failed to setup Bitmap pool. Processing might be slow or fail.");
            // Depending on desired behavior, could return false here or try to continue without pool.
            // For now, log and continue; process() will try to create Bitmaps if pool is not usable.
        }


        Log.i(TAG, "OpenGL setup successful for output size: " + outputWidth + "x" + outputHeight + (isGLES3 ? " (GLES 3 with PBOs)" : " (GLES 2)"));
        return true;
    }

    private boolean initEGL() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            Log.e(TAG, "eglGetDisplay failed: " + android.opengl.GLUtils.getEGLErrorString(EGL14.eglGetError()));
            return false;
        }

        int[] version = new int[2];
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            Log.e(TAG, "eglInitialize failed: " + android.opengl.GLUtils.getEGLErrorString(EGL14.eglGetError()));
            return false;
        }
        Log.i(TAG, "EGL Initialized. Version: " + version[0] + "." + version[1]);


        int[] attribList = {
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT, // For offscreen rendering
                EGL14.EGL_NONE
        };
        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfigs = new int[1];
        if (!EGL14.eglChooseConfig(eglDisplay, attribList, 0, configs, 0, configs.length, numConfigs, 0)) {
            Log.e(TAG, "eglChooseConfig failed: " + android.opengl.GLUtils.getEGLErrorString(EGL14.eglGetError()));
            return false;
        }
        if (numConfigs[0] == 0) {
            Log.e(TAG, "No suitable EGLConfig found.");
            return false;
        }
        eglConfig = configs[0];

        int[] contextAttribs_GLES3 = {
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 3,
                EGL14.EGL_NONE
        };
        eglContext = EGL14.eglCreateContext(eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttribs_GLES3, 0);

        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            Log.w(TAG, "Failed to create GLES 3.0 context, trying GLES 2.0. Error: " + android.opengl.GLUtils.getEGLErrorString(EGL14.eglGetError()));
            int[] contextAttribs_GLES2 = {
                    EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
                    EGL14.EGL_NONE
            };
            eglContext = EGL14.eglCreateContext(eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttribs_GLES2, 0);
            isGLES3 = false;
        } else {
            isGLES3 = true;
            Log.i(TAG, "GLES 3.0 context created successfully.");
        }

        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            Log.e(TAG, "eglCreateContext failed: " + android.opengl.GLUtils.getEGLErrorString(EGL14.eglGetError()));
            return false;
        }

        // Create a Pbuffer surface for offscreen rendering
        int[] pbufferAttribs = {
                EGL14.EGL_WIDTH, outputWidth,
                EGL14.EGL_HEIGHT, outputHeight,
                EGL14.EGL_NONE
        };
        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, eglConfig, pbufferAttribs, 0);
        if (eglSurface == EGL14.EGL_NO_SURFACE) {
            Log.e(TAG, "eglCreatePbufferSurface failed: " + android.opengl.GLUtils.getEGLErrorString(EGL14.eglGetError()));
            return false;
        }

        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            Log.e(TAG, "eglMakeCurrent failed: " + android.opengl.GLUtils.getEGLErrorString(EGL14.eglGetError()));
            return false;
        }
        return true;
    }

    private boolean setupPBOs() {
        if (!isGLES3) return false; // Should not happen if called correctly

        pboIds = new int[PBO_COUNT];
        GLES30.glGenBuffers(PBO_COUNT, pboIds, 0);
        if (checkGlError("glGenBuffers for PBOs")) return false;

        pboBufferSize = outputWidth * outputHeight * 4; // RGBA8

        for (int i = 0; i < PBO_COUNT; i++) {
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pboIds[i]);
            GLES30.glBufferData(GLES30.GL_PIXEL_PACK_BUFFER, pboBufferSize, null, GLES30.GL_STREAM_READ);
            if (checkGlError("glBufferData for PBO " + i)) {
                GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0);
                return false;
            }
        }
        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0);
        pboReadIndex = 0;
        pboMapIndex = -1; // No PBO is ready to be mapped initially
        return true;
    }

    private boolean setupBitmapPool() {
        availableBitmaps = new ArrayBlockingQueue<>(BITMAP_POOL_SIZE);
        allCreatedBitmapsInPool = new ArrayList<>(BITMAP_POOL_SIZE);
        Log.i(TAG, "Setting up Bitmap pool with size: " + BITMAP_POOL_SIZE + " for " + outputWidth + "x" + outputHeight);
        for (int i = 0; i < BITMAP_POOL_SIZE; i++) {
            try {
                Bitmap bmp = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888);
                allCreatedBitmapsInPool.add(bmp);
                availableBitmaps.offer(bmp);
            } catch (OutOfMemoryError e) {
                Log.e(TAG, "OutOfMemoryError while pre-allocating bitmap " + (i + 1) + "/" + BITMAP_POOL_SIZE + " for pool", e);
                // If pre-allocation fails, the pool will be smaller or empty.
                // process() will try to create bitmaps if pool.take() fails or pool is empty initially.
                return false; // Indicate pool setup failed partially or completely
            } catch (Exception e) {
                Log.e(TAG, "Exception while pre-allocating bitmap " + (i + 1) + "/" + BITMAP_POOL_SIZE + " for pool", e);
                return false;
            }
        }
        Log.i(TAG, "Bitmap pool pre-allocated with " + availableBitmaps.size() + " bitmaps.");
        return true;
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

    private int loadShader(int type, String shaderCode) {
        int shader = GLES20.glCreateShader(type);
        if (shader == 0) {
            Log.e(TAG, "glCreateShader failed for type " + type);
            return 0;
        }
        GLES20.glShaderSource(shader, shaderCode);
        GLES20.glCompileShader(shader);
        int[] compiled = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            Log.e(TAG, "Could not compile shader " + type + ":");
            Log.e(TAG, GLES20.glGetShaderInfoLog(shader));
            GLES20.glDeleteShader(shader);
            shader = 0;
        }
        return shader;
    }

    private int createProgram(String vertexSource, String fragmentSource) {
        int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource);
        if (vertexShader == 0) return 0;
        int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
        if (fragmentShader == 0) {
            GLES20.glDeleteShader(vertexShader);
            return 0;
        }

        int program = GLES20.glCreateProgram();
        if (program == 0) {
            Log.e(TAG, "glCreateProgram failed");
            return 0;
        }
        GLES20.glAttachShader(program, vertexShader);
        GLES20.glAttachShader(program, fragmentShader);
        GLES20.glLinkProgram(program);
        int[] linkStatus = new int[1];
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0);
        if (linkStatus[0] == 0) {
            Log.e(TAG, "Could not link program: ");
            Log.e(TAG, GLES20.glGetProgramInfoLog(program));
            GLES20.glDeleteProgram(program);
            program = 0;
        }
        // Shaders can be deleted after linking
        GLES20.glDeleteShader(vertexShader);
        GLES20.glDeleteShader(fragmentShader);
        return program;
    }

    private boolean setupFramebuffer(int width, int height) {
        // Create texture for FBO - Always use RGBA8 for now to ensure stability
        GLES20.glGenTextures(1, outputTextureHandle, 0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, outputTextureHandle[0]);
        Log.i(TAG, "Setting up FBO with RGBA8 texture format.");
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, width, height, 0,
                GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        if (checkGlError("FBO Texture setup (RGBA8)")) return false;

        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        // Create FBO
        GLES20.glGenFramebuffers(1, fboHandle, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboHandle[0]);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D, outputTextureHandle[0], 0);

        int fboStatus = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
        if (fboStatus != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "Framebuffer not complete: " + Integer.toHexString(fboStatus) + " (Using RGBA8)");
            return false;
        }
        Log.i(TAG, "FBO successfully completed with RGBA8 format.");
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0); // Unbind FBO

        // Allocate reusable readback buffer
        if (readbackBuffer == null || readbackBuffer.capacity() < width * height * 4) {
            readbackBuffer = ByteBuffer.allocateDirect(width * height * 4).order(ByteOrder.nativeOrder());
        }
        return true;
    }

    // Method to update the pre-existing input texture
    private boolean updateInputTexture(Bitmap bitmap) {
        if (bitmap == null || bitmap.isRecycled()) {
            Log.e(TAG, "Cannot update texture from null or recycled bitmap.");
            return false;
        }
        if (currentInputTextureId == 0) {
            Log.e(TAG, "Input texture ID is not initialized.");
            return false;
        }

        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, currentInputTextureId);
        // Use texSubImage2D since texture storage is pre-allocated and size is fixed
        GLUtils.texSubImage2D(GLES20.GL_TEXTURE_2D, 0, 0, 0, bitmap);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0); // Unbind

        return !checkGlError("Update Input Texture with texSubImage2D");
    }

    public Bitmap process(Bitmap inputBitmap) {
        long totalStartTime = System.nanoTime();
        long dataUploadTimeNs, processTimeNs, dataReadbackTimeNs;

        if (inputBitmap == null || inputBitmap.isRecycled()) {
            Log.e(TAG, "Input bitmap is null or recycled.");
            return null;
        }
        if (eglDisplay == EGL14.EGL_NO_DISPLAY || eglContext == EGL14.EGL_NO_CONTEXT || currentInputTextureId == 0 ) {
            Log.e(TAG, "EGL not setup, resources not initialized, or already released.");
            return null;
        }
        // Check for bitmap pool readiness, though process() will try to handle null from pool.
        if (availableBitmaps == null) {
            Log.e(TAG, "Bitmap pool not initialized!");
            // Attempt to create a bitmap directly as a last resort, or return null
            // For now, let process() handle it, it might try to create one if pool.take() fails.
            // This path should ideally not be hit if setup was successful.
        }

        // Fallback readback buffer check if PBOs are not used
        if (!isGLES3 && readbackBuffer == null) {
            Log.e(TAG, "Readback buffer not initialized for GLES2 path.");
            return null;
        }


         if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            int error = EGL14.eglGetError(); // Get error immediately
            Log.e(TAG, "eglMakeCurrent failed in process. eglMakeCurrent returned false. EGL Error: " + error + " (" + android.opengl.GLUtils.getEGLErrorString(error) + ")");
            return null;
        }

        long updateTextureStartTime = System.nanoTime();
        if (!updateInputTexture(inputBitmap)) {
            Log.e(TAG, "Failed to update input texture.");
            return null;
        }
        dataUploadTimeNs = System.nanoTime() - updateTextureStartTime;

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboHandle[0]);
        GLES20.glViewport(0, 0, outputWidth, outputHeight);
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 0.0f); // Not strictly necessary for offscreen
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

        GLES20.glUseProgram(programHandle);

        GLES20.glEnableVertexAttribArray(positionHandle);
        GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer);

        GLES20.glEnableVertexAttribArray(texCoordHandle);
        GLES20.glVertexAttribPointer(texCoordHandle, 2, GLES20.GL_FLOAT, false, 0, texCoordBuffer);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, currentInputTextureId);
        GLES20.glUniform1i(inputTextureHandle, 0);

        // Pass texel size for bicubic interpolation
        float texelWidth = 1.0f / inputBitmap.getWidth();
        float texelHeight = 1.0f / inputBitmap.getHeight();
        GLES20.glUniform2f(texelSizeHandle, texelWidth, texelHeight);

        long drawArraysStartTime = System.nanoTime();
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        processTimeNs = System.nanoTime() - drawArraysStartTime;

        GLES20.glDisableVertexAttribArray(positionHandle);
        GLES20.glDisableVertexAttribArray(texCoordHandle);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0); // Unbind input texture
        // Do not delete currentInputTextureId here, it's reused.

        Bitmap outputBitmap = null;
        long readPixelsStartTime = System.nanoTime();

        if (isGLES3 && pboIds != null) {
            // --- PBO Path ---
            // 1. Issue read command to the current pboReadIndex
            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pboIds[pboReadIndex]);
            GLES30.glReadPixels(0, 0, outputWidth, outputHeight, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, 0); // Offset 0
            boolean glReadError = checkGlError("glReadPixels to PBO");

            if (glReadError) {
                Log.e(TAG, "glReadPixels to PBO failed. Will use placeholder.");
                // outputBitmap remains null, placeholder logic will be hit later
            }
            // Always unbind the PBO targeted by glReadPixels *after* attempting to map the other PBO,
            // or if an error occurred during read.
            // This PBO (pboIds[pboReadIndex]) is now done with its GPU-side work.
             GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0);


            // 2. Try to map and process the pboMapIndex (previous PBO an actual PBO was ready for mapping)
            if (!glReadError && pboMapIndex != -1) {
                GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pboIds[pboMapIndex]);
                ByteBuffer mappedBuffer = null;
                try {
                    mappedBuffer = (ByteBuffer) GLES30.glMapBufferRange(
                            GLES30.GL_PIXEL_PACK_BUFFER, 0, pboBufferSize, GLES30.GL_MAP_READ_BIT);

                    if (mappedBuffer != null) {
                        Log.i(TAG, "PBO: Attempting to take bitmap from pool. Pool size: " + availableBitmaps.size());
                        outputBitmap = availableBitmaps.take(); // Blocks if pool is empty
                        Log.i(TAG, "PBO: Took bitmap (hashCode: " + (outputBitmap != null ? outputBitmap.hashCode() : "null") + ") from pool. Pool size now: " + availableBitmaps.size());
                        if (outputBitmap.getWidth() != outputWidth || outputBitmap.getHeight() != outputHeight || outputBitmap.isRecycled()) {
                            Log.e(TAG, "PBO: Bitmap from pool (hashCode: " + outputBitmap.hashCode() +
                                    ") is not valid. Expected: " + outputWidth + "x" + outputHeight +
                                    ", Got: " + outputBitmap.getWidth() + "x" + outputBitmap.getHeight() +
                                    ", isRecycled: " + outputBitmap.isRecycled() + ". Re-creating. THIS IS A GC SOURCE.");
                            if(!outputBitmap.isRecycled()) outputBitmap.recycle();
                            outputBitmap = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888);
                        }
                        outputBitmap.copyPixelsFromBuffer(mappedBuffer);
                    } else {
                        checkGlError("glMapBufferRange failed for PBO " + pboMapIndex);
                        Log.w(TAG, "Failed to map PBO buffer: " + pboMapIndex + ". Will use placeholder.");
                        // outputBitmap remains null
                    }
                } catch (InterruptedException e) {
                    Log.e(TAG, "Interrupted while waiting for bitmap from pool for mapped PBO.", e);
                    Thread.currentThread().interrupt();
                    // outputBitmap might be null. Let placeholder logic handle it.
                } finally {
                    if (mappedBuffer != null) {
                        GLES30.glUnmapBuffer(GLES30.GL_PIXEL_PACK_BUFFER);
                    }
                    GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0); // Unbind pboMapIndex PBO
                }
            } else if (pboMapIndex == -1 && !glReadError) {
                Log.w(TAG, "Initial PBO cycle (pboMapIndex is -1), no PBO to map yet. Will use placeholder.");
                // outputBitmap remains null
            }
            // If glReadError was true, outputBitmap is also null here.

            // If outputBitmap is still null at this point, get/create a placeholder
            if (outputBitmap == null) {
                Log.w(TAG, "PBO processing yielded null or error, preparing placeholder. pboMapIndex: " + pboMapIndex + ", glReadError: " + glReadError);
                try {
                    Log.d(TAG, "PBO Placeholder: Attempting to take bitmap. Pool size: " + availableBitmaps.size());
                    outputBitmap = availableBitmaps.take();
                    Log.d(TAG, "PBO Placeholder: Took bitmap (hashCode: " + (outputBitmap != null ? outputBitmap.hashCode() : "null") + "). Pool size now: " + availableBitmaps.size());
                    if (outputBitmap.getWidth() != outputWidth || outputBitmap.getHeight() != outputHeight || outputBitmap.isRecycled()) {
                        Log.e(TAG, "PBO Placeholder: Bitmap from pool (hashCode: " + outputBitmap.hashCode() +
                                ") is not valid. Expected: " + outputWidth + "x" + outputHeight +
                                ", Got: " + outputBitmap.getWidth() + "x" + outputBitmap.getHeight() +
                                ", isRecycled: " + outputBitmap.isRecycled() + ". Re-creating. THIS IS A GC SOURCE.");
                        if(!outputBitmap.isRecycled()) outputBitmap.recycle();
                        outputBitmap = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888);
                    }
                    Canvas canvas = new Canvas(outputBitmap);
                    if (glReadError) {
                        canvas.drawColor(0xFFFF0000); // Red for glReadPixels error
                    } else if (pboMapIndex == -1) {
                        canvas.drawColor(0xFF0000FF); // Blue for initial unready PBO
                    } else { // Presumed map fail or interrupt during map data copy
                        canvas.drawColor(0xFFFFA500); // Orange for map fail/other
                    }
                } catch (InterruptedException e) {
                    Log.e(TAG, "Interrupted while getting placeholder bitmap.", e);
                    Thread.currentThread().interrupt();
                    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
                    return null; // Critical: cannot get even a placeholder.
                } catch (Exception e) {
                    Log.e(TAG, "Exception while creating/taking placeholder bitmap.", e);
                    GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
                    return null;
                }
            }

            // 3. Update PBO indices for next frame
            pboMapIndex = pboReadIndex;
            pboReadIndex = (pboReadIndex + 1) % PBO_COUNT;
            dataReadbackTimeNs = System.nanoTime() - readPixelsStartTime;

        } else {
            // --- Synchronous GLES20 Path (Fallback) ---
            if (readbackBuffer == null) { // Should have been caught earlier, but defensive check
                Log.e(TAG, "Fallback Readback buffer is null!");
                return null;
            }
            try {
                Log.d(TAG, "GLES2: Attempting to take bitmap from pool. Pool size: " + availableBitmaps.size());
                outputBitmap = availableBitmaps.take(); // Blocks if pool is empty
                Log.d(TAG, "GLES2: Took bitmap (hashCode: " + (outputBitmap != null ? outputBitmap.hashCode() : "null") + ") from pool. Pool size now: " + availableBitmaps.size());
                 if (outputBitmap.getWidth() != outputWidth || outputBitmap.getHeight() != outputHeight || outputBitmap.isRecycled()) {
                    Log.e(TAG, "GLES2 Path: Bitmap from pool (hashCode: " + outputBitmap.hashCode() +
                            ") is not valid. Expected: " + outputWidth + "x" + outputHeight +
                            ", Got: " + outputBitmap.getWidth() + "x" + outputBitmap.getHeight() +
                            ", isRecycled: " + outputBitmap.isRecycled() + ". Re-creating. THIS IS A GC SOURCE.");
                    if(!outputBitmap.isRecycled()) outputBitmap.recycle();
                    outputBitmap = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888);
                }
            } catch (InterruptedException e) {
                Log.e(TAG, "GLES2 Path: Interrupted while waiting for an available bitmap from pool.", e);
                Thread.currentThread().interrupt();
                return null;
            }
            readbackBuffer.clear(); // Ensure buffer is ready for new data
            GLES20.glReadPixels(0, 0, outputWidth, outputHeight, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, readbackBuffer);
            dataReadbackTimeNs = System.nanoTime() - readPixelsStartTime;
            readbackBuffer.rewind();
            outputBitmap.copyPixelsFromBuffer(readbackBuffer);
            if (checkGlError("GLES20 glReadPixels")) {
                 if (outputBitmap != null && !outputBitmap.isRecycled()) outputBitmap.recycle();
                 return null;
            }
        }

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0); // Unbind FBO (common for both paths)

        // With placeholder logic, outputBitmap should ideally not be null here,
        // unless a critical error occurred while obtaining the placeholder.
        if (checkGlError("Process Frame Post Read")) {
            // If a GL error occurred after all pixel operations, it's uncertain if outputBitmap is valid.
            // Depending on the error, we might still return outputBitmap or null.
            // For safety, if a GL error is detected here, and outputBitmap exists, recycle and return null.
            if (outputBitmap != null && !outputBitmap.isRecycled()) {
                Log.e(TAG, "GL error after processing. Recycling potentially corrupt outputBitmap.");
                outputBitmap.recycle();
            }
            return null;
        }

        long totalEndTime = System.nanoTime();
        Log.i(TAG, String.format("Process Times: Total: %.2f ms, Upload: %.2f ms, GPU Process: %.2f ms, Readback: %.2f ms",
                (totalEndTime - totalStartTime) / 1_000_000.0,
                dataUploadTimeNs / 1_000_000.0,
                processTimeNs / 1_000_000.0,
                dataReadbackTimeNs / 1_000_000.0));

        return outputBitmap; // Should be non-null if placeholder logic worked.
    }

    public void releaseBitmapToPool(Bitmap bitmap) {
        if (bitmap != null && availableBitmaps != null) {
            // Optionally, verify if this bitmap originated from the pool if strict control is needed,
            // but for now, assume any bitmap passed here is intended for pooling if it fits.
            // Check if it's one of the initially created ones or if its config matches.
            // For simplicity, just try to offer it. If the queue is full, it might be rejected (if offer is used and queue has capacity).
            // Using offer to avoid blocking if the pool is somehow overfilled or bitmap is not from pool.
            if (allCreatedBitmapsInPool.contains(bitmap)) { // Only pool bitmaps we know we created
                if (!availableBitmaps.offer(bitmap)) {
                    Log.e(TAG, "Failed to offer bitmap (hashCode: " + bitmap.hashCode() + ", recycled: " + bitmap.isRecycled() + ") back to pool. Pool size: " + availableBitmaps.size() + "/" + BITMAP_POOL_SIZE + ". THIS BITMAP WILL BE GC'D IF NOT RECYCLED ELSEWHERE AND NO OTHER STRONG REFS.");
                    // If offer fails for a bitmap that originated from the pool, it's a problem.
                    // It might indicate the pool is too small or bitmaps are not being consumed/released quickly enough.
                    // Recycling it here could be dangerous if it's still in use elsewhere or if the pool logic is flawed.
                    // However, if it truly cannot be re-pooled, it's better to log it as an error.
                }
            } else {
                 Log.w(TAG, "Attempted to release a bitmap to pool that was not tracked by the pool. Recycling it directly.");
                 if (!bitmap.isRecycled()) {
                    bitmap.recycle();
                 }
            }
        }
    }

    public void release() {
        Log.i(TAG, "Releasing OpenGL resources.");
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
            if (programHandle != 0) {
                GLES20.glDeleteProgram(programHandle);
                programHandle = 0;
            }
            if (outputTextureHandle[0] != 0) {
                GLES20.glDeleteTextures(1, outputTextureHandle, 0);
                outputTextureHandle[0] = 0;
            }
            if (currentInputTextureId != 0) {
                GLES20.glDeleteTextures(1, new int[]{currentInputTextureId}, 0);
                currentInputTextureId = 0;
            }
            if (fboHandle[0] != 0) {
                GLES20.glDeleteFramebuffers(1, fboHandle, 0);
                fboHandle[0] = 0;
            }
            if (isGLES3 && pboIds != null) {
                GLES30.glDeleteBuffers(PBO_COUNT, pboIds, 0);
                pboIds = null;
            }
            // Recycle all bitmaps created by the pool
            if (allCreatedBitmapsInPool != null) {
                for (Bitmap bmp : allCreatedBitmapsInPool) {
                    if (bmp != null && !bmp.isRecycled()) {
                        bmp.recycle();
                    }
                }
                allCreatedBitmapsInPool.clear();
                allCreatedBitmapsInPool = null;
            }
            if (availableBitmaps != null) {
                availableBitmaps.clear();
                availableBitmaps = null;
            }

            if (eglSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, eglSurface);
                eglSurface = EGL14.EGL_NO_SURFACE;
            }
            if (eglContext != EGL14.EGL_NO_CONTEXT) {
                EGL14.eglDestroyContext(eglDisplay, eglContext);
                eglContext = EGL14.EGL_NO_CONTEXT;
            }
            EGL14.eglTerminate(eglDisplay);
            eglDisplay = EGL14.EGL_NO_DISPLAY;
        }
        context = null; // Release context
        Log.i(TAG, "OpenGL resources released.");
    }

    private boolean checkGlError(String op) {
        int error;
        boolean anErrorOccurred = false;
        while ((error = GLES20.glGetError()) != GLES20.GL_NO_ERROR) {
            Log.e(TAG, op + ": glError " + error + " (" + android.opengl.GLUtils.getEGLErrorString(error) + ")");
            anErrorOccurred = true;
        }
        return anErrorOccurred;
    }
}