package com.example.ffmpegvideoplayer;

import android.content.Context;
import android.content.Context;
import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.opengl.GLES20;
import android.opengl.GLES30;
import android.util.Log;
import android.view.Surface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class OpenGLImageProcessor {

    private static final String TAG = "OpenGLImageProcessor";

    private EGLDisplay eglDisplay = EGL14.EGL_NO_DISPLAY;
    private EGLContext eglContext = EGL14.EGL_NO_CONTEXT;
    private EGLConfig eglConfig;
    private EGLSurface eglSurface = EGL14.EGL_NO_SURFACE; // This will be the window surface
    private EGLSurface eglPbufferSurface = EGL14.EGL_NO_SURFACE; // A temporary surface for setup
    private Surface surface;
    private volatile SurfaceTexture pendingSurfaceTexture;
    private boolean surfaceTextureIsPending = false;


    private int programHandle;
    private int positionHandle;
    private int texCoordHandle;
    private int texelSizeHandle;

    // Uniform handles
    private int uBaseTextureHandle;
    private int uSrPatchTextureHandle;
    private int uPatchRectHandle;
    private int uDrawPatchHandle;
    private int uRenderModeHandle;

    // FBO for the first pass (upscaling)
    private int[] upscaleFbo = new int[1];
    private int[] upscaledTexture = new int[1];

    // FBO for the second pass (compositing), which targets the final output
    // Texture IDs
    private int inputVideoTextureId = 0; // For the raw video frame
    private int srPatchTextureId = 0;    // For the SR patch from TFLite
    private int lastPatchWidth = 0;
    private int lastPatchHeight = 0;

    private FloatBuffer vertexBuffer;
    private FloatBuffer texCoordBuffer;

    private int outputWidth;
    private int outputHeight;
    private int surfaceWidth;
    private int surfaceHeight;

    private Context context;

    private static final float[] VERTICES = {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
    private static final float[] TEX_COORDS = {0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f}; // Flipped Y-axis

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

        // Get handles for all uniforms
        uRenderModeHandle = GLES20.glGetUniformLocation(programHandle, "u_RenderMode");
        uBaseTextureHandle = GLES20.glGetUniformLocation(programHandle, "u_BaseTexture");
        uSrPatchTextureHandle = GLES20.glGetUniformLocation(programHandle, "u_SrPatchTexture");
        uPatchRectHandle = GLES20.glGetUniformLocation(programHandle, "u_PatchRect");
        uDrawPatchHandle = GLES20.glGetUniformLocation(programHandle, "u_DrawPatch");
        texelSizeHandle = GLES20.glGetUniformLocation(programHandle, "u_TexelSize"); // Re-confirming handle, though name is same

        // Set texture unit uniforms once, since they don't change.
        GLES20.glUniform1i(uBaseTextureHandle, 0); // Corresponds to GL_TEXTURE0
        GLES20.glUniform1i(uSrPatchTextureHandle, 1); // Corresponds to GL_TEXTURE1

        // --- Create all necessary textures ---
        inputVideoTextureId = createTexture(inputWidth, inputHeight, false);
        srPatchTextureId = createTexture(0, 0, false); // SR patch is now RGBA8, not float
        upscaledTexture[0] = createTexture(outputWidth, outputHeight, false);

        // --- Setup FBO for Pass 1 (Upscaling) ---
        GLES20.glGenFramebuffers(1, upscaleFbo, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, upscaleFbo[0]);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D, upscaledTexture[0], 0);

        int fboStatus = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
        if (fboStatus != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            Log.e(TAG, "Failed to create FBO for upscale pass: FBO status is not complete! Status: " + fboStatus);
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
            return false;
        }

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);

        return true;
    }

    /**
     * Pass 1: Upscales the low-resolution video frame and renders it to an internal texture.
     */
    public void performUpscalePass(ByteBuffer inputBuffer, int inputWidth, int inputHeight) {
        GLES20.glUseProgram(programHandle);

        // Bind the FBO for the upscale pass
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, upscaleFbo[0]);
        GLES20.glViewport(0, 0, outputWidth, outputHeight);

        // Upload the new video frame data to its texture
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inputVideoTextureId);
        GLES20.glTexSubImage2D(GLES20.GL_TEXTURE_2D, 0, 0, 0, inputWidth, inputHeight, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, inputBuffer);

        // Set uniforms for the upscale shader
        GLES20.glUniform1i(uRenderModeHandle, 0); // Mode 0: Upscale
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inputVideoTextureId);
        GLES20.glUniform1i(uBaseTextureHandle, 0);
        GLES20.glUniform2f(texelSizeHandle, 1.0f / inputWidth, 1.0f / inputHeight);

        // Draw the quad
        drawQuad();

        // Use glFinish() for debugging to force synchronization. This is slow!
        // If this fixes the issue, it's a synchronization problem.
        GLES20.glFinish();

        // Unbind FBO
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
    }

    // * Pass 2: Composites the upscaled background with the SR patch onto a final hardware bitmap.
    public void performCompositePass(ByteBuffer srPatchBuffer, int patchWidth, int patchHeight, float[] patchRect) {
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            return; // EGL not initialized
        }

        handleSurfaceChange();

        if (eglSurface == null || eglSurface == EGL14.EGL_NO_SURFACE) {
            // This is expected if the TextureView is not available yet.
            return;
        }

        // --- Restore Rendering Logic ---
        GLES20.glUseProgram(programHandle);

        // Bind the default framebuffer (the one for the EGL window surface)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        GLES20.glViewport(0, 0, surfaceWidth, surfaceHeight);

        // Clear the screen with black before drawing.
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);

        // Upload SR patch data to its texture (if available)
        if (srPatchBuffer != null) {
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, srPatchTextureId);
            if (patchWidth != lastPatchWidth || patchHeight != lastPatchHeight) {
                GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA, patchWidth, patchHeight, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, srPatchBuffer);
                lastPatchWidth = patchWidth;
                lastPatchHeight = patchHeight;
            } else {
                GLES30.glTexSubImage2D(GLES30.GL_TEXTURE_2D, 0, 0, 0, patchWidth, patchHeight, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, srPatchBuffer);
            }
        }

        // Set uniforms for the composite shader
        GLES20.glUniform1i(uRenderModeHandle, 1); // Mode 1: Composite

        // --- FINAL FIX ---
        // Restore the original logic completely.
        GLES20.glUniform1i(uDrawPatchHandle, srPatchBuffer != null ? 1 : 0);
        if (srPatchBuffer != null) {
            GLES20.glUniform4f(uPatchRectHandle, patchRect[0], patchRect[1], patchRect[2], patchRect[3]);
        }

        // Bind textures
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, upscaledTexture[0]); // Background

        // Re-enable binding the patch texture
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, srPatchTextureId); // SR Patch

        // Draw the quad
        drawQuad();

        // --- Finalization ---
        // Swap buffers to display the rendered frame
        if (!EGL14.eglSwapBuffers(eglDisplay, eglSurface)) {
            Log.e(TAG, "eglSwapBuffers failed!");
            checkEglError("eglSwapBuffers");
        }
    }

    public void release() {
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
            if (programHandle != 0) GLES20.glDeleteProgram(programHandle);

            // Delete all textures
            int[] texturesToDelete = {inputVideoTextureId, srPatchTextureId, upscaledTexture[0]};
            GLES20.glDeleteTextures(texturesToDelete.length, texturesToDelete, 0);

            // Delete all FBOs
            GLES20.glDeleteFramebuffers(1, upscaleFbo, 0);

            if (eglSurface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(eglDisplay, eglSurface);
            if (eglPbufferSurface != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(eglDisplay, eglPbufferSurface);
            if (eglContext != EGL14.EGL_NO_CONTEXT) EGL14.eglDestroyContext(eglDisplay, eglContext);
            EGL14.eglTerminate(eglDisplay);
        }
        eglDisplay = EGL14.EGL_NO_DISPLAY;
        eglContext = EGL14.EGL_NO_CONTEXT;
        eglSurface = EGL14.EGL_NO_SURFACE;
        if (surface != null) {
            surface.release();
            surface = null;
        }
    }

    // --- Utility Methods ---

    private void drawQuad() {
        GLES20.glEnableVertexAttribArray(positionHandle);
        GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer);
        GLES20.glEnableVertexAttribArray(texCoordHandle);
        GLES20.glVertexAttribPointer(texCoordHandle, 2, GLES20.GL_FLOAT, false, 0, texCoordBuffer);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glDisableVertexAttribArray(positionHandle);
        GLES20.glDisableVertexAttribArray(texCoordHandle);
    }

    private int createTexture(int width, int height, boolean isFloat) {
        int[] texture = new int[1];
        GLES30.glGenTextures(1, texture, 0);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture[0]);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);
        if (width > 0 && height > 0) {
            if (isFloat) {
                // For TFLite FLOAT32 output (RGB)
                GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGB32F, width, height, 0, GLES30.GL_RGB, GLES30.GL_FLOAT, null);
            } else {
                // For RGBA8 video frames
                GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA, width, height, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null);
            }
        }
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, 0);
        return texture[0];
    }

    private boolean initEGL() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            Log.e(TAG, "eglGetDisplay failed");
            return false;
        }
        int[] version = new int[2];
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            Log.e(TAG, "eglInitialize failed");
            return false;
        }

        // Choose a config that supports rendering to a window and pbuffer
        int[] attribList = {
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT | EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_NONE
        };
        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfigs = new int[1];
        if (!EGL14.eglChooseConfig(eglDisplay, attribList, 0, configs, 0, configs.length, numConfigs, 0) || numConfigs[0] == 0) {
            Log.e(TAG, "eglChooseConfig failed");
            return false;
        }
        this.eglConfig = configs[0];

        int[] contextAttribs = {EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE};
        eglContext = EGL14.eglCreateContext(eglDisplay, this.eglConfig, EGL14.EGL_NO_CONTEXT, contextAttribs, 0);
        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            Log.w(TAG, "Failed to create GLES 3 context, falling back to GLES 2.");
            contextAttribs[1] = 2;
            eglContext = EGL14.eglCreateContext(eglDisplay, this.eglConfig, EGL14.EGL_NO_CONTEXT, contextAttribs, 0);
        }
        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            Log.e(TAG, "Failed to create EGL context");
            checkEglError("eglCreateContext");
            return false;
        }

        // Create a temporary 1x1 pbuffer surface to make the context current for setup.
        int[] pbufferAttribs = {EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE};
        eglPbufferSurface = EGL14.eglCreatePbufferSurface(eglDisplay, this.eglConfig, pbufferAttribs, 0);
        if (eglPbufferSurface == EGL14.EGL_NO_SURFACE) {
            Log.e(TAG, "Failed to create EGL pbuffer surface");
            checkEglError("eglCreatePbufferSurface");
            return false;
        }

        // Make the context current with the pbuffer surface.
        if (!EGL14.eglMakeCurrent(eglDisplay, eglPbufferSurface, eglPbufferSurface, eglContext)) {
            Log.e(TAG, "eglMakeCurrent failed on pbuffer");
            checkEglError("eglMakeCurrent (pbuffer)");
            return false;
        }
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

    public void setSurface(SurfaceTexture surfaceTexture, int width, int height) {
        // This method is called from the main thread.
        // We just store the new surface texture and dimensions.
        // The actual EGL surface creation will happen on the rendering thread.
        this.pendingSurfaceTexture = surfaceTexture;
        this.surfaceWidth = width;
        this.surfaceHeight = height;
        this.surfaceTextureIsPending = true; // Signal that a change is pending
    }

    private void handleSurfaceChange() {
        // This method is called from the rendering thread.
        if (!surfaceTextureIsPending) {
            return;
        }
        Log.d(TAG, "Handling pending surface texture change.");
        surfaceTextureIsPending = false;

        // Destroy the old window surface if it exists
        if (eglSurface != EGL14.EGL_NO_SURFACE) {
            Log.d(TAG, "Destroying old EGL surface.");
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
            EGL14.eglDestroySurface(eglDisplay, eglSurface);
            eglSurface = EGL14.EGL_NO_SURFACE;
        }
        // Release the old Java-level Surface object
        if (this.surface != null) {
            Log.d(TAG, "Releasing old Surface object.");
            this.surface.release();
            this.surface = null;
        }

        if (pendingSurfaceTexture != null) {
            Log.d(TAG, "Creating new EGL surface for new SurfaceTexture.");
            this.surface = new Surface(pendingSurfaceTexture);
            int[] surfaceAttribs = {EGL14.EGL_NONE};

            // CRITICAL FIX: Use the EGLConfig that was stored during context creation.
            // Do NOT query for a new one.
            if (this.eglConfig == null) {
                Log.e(TAG, "EGLConfig was not initialized. Cannot create window surface.");
                return;
            }

            eglSurface = EGL14.eglCreateWindowSurface(eglDisplay, this.eglConfig, surface, surfaceAttribs, 0);
            if (eglSurface == EGL14.EGL_NO_SURFACE) {
                Log.e(TAG, "eglCreateWindowSurface failed!");
                checkEglError("eglCreateWindowSurface");
                // If surface creation fails, we can't render.
                return;
            }

            if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
                Log.e(TAG, "eglMakeCurrent failed on new window surface!");
                checkEglError("eglMakeCurrent (window)");
                // If make current fails, we can't render.
                return;
            }
            Log.i(TAG, "Successfully created and made current the new EGL window surface.");

            // Now that we have a real surface, we can destroy the temporary pbuffer surface if it still exists.
            if (eglPbufferSurface != EGL14.EGL_NO_SURFACE) {
                Log.d(TAG, "Destroying temporary pbuffer surface.");
                EGL14.eglDestroySurface(eglDisplay, eglPbufferSurface);
                eglPbufferSurface = EGL14.EGL_NO_SURFACE;
            }
        } else {
            Log.d(TAG, "Pending surface texture was null. Cleaning up.");
            // If the new surface is null (e.g., TextureView destroyed), make no surface current.
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, eglContext);
        }
    }

    public void onSurfaceSizeChanged(int width, int height) {
        this.surfaceWidth = width;
        this.surfaceHeight = height;
    }
    private void checkEglError(String msg) {
        int error;
        while ((error = EGL14.eglGetError()) != EGL14.EGL_SUCCESS) {
            Log.e(TAG, msg + ": EGL error: 0x" + Integer.toHexString(error));
        }
    }
}