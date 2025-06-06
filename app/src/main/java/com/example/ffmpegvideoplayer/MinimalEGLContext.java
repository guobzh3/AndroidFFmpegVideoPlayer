package com.example.ffmpegvideoplayer;

import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.util.Log;

// 如果你想支持 GLES 3, 需要 EGL_OPENGL_ES3_BIT_KHR, 它在 EGL14 中定义为 EGL_OPENGL_ES3_BIT
// 但 EGL_OPENGL_ES3_BIT_KHR (0x00000040) 可能更明确，有些地方用 EGL_OPENGL_ES3_BIT (0x00000004)
// Android 的 EGL14.EGL_OPENGL_ES3_BIT 应该是正确的常量 (0x0040)
//import static android.opengl.EGL14.EGL_OPENGL_ES3_BIT;


public class MinimalEGLContext {
    private static final String TAG = "MinimalEGLContext";

    private EGLDisplay eglDisplay = EGL14.EGL_NO_DISPLAY;
    private EGLContext eglContext = EGL14.EGL_NO_CONTEXT;
    private EGLSurface eglSurface = EGL14.EGL_NO_SURFACE;
    private EGLConfig eglConfig = null;

    // 你可以选择 GLES 版本 (2 或 3)
    // 对于 GLES 3, 你可能需要 API Level 18+
    private static final int EGL_CONTEXT_CLIENT_VERSION = 2; // 设为 3 以创建 GLES 3 上下文

    /**
     * 初始化EGL环境，创建一个EGLContext并将其设置为当前。
     * @return true 如果成功, false 如果失败。
     */
    public boolean setup() {
        // 1. 获取默认显示设备
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) {
            Log.e(TAG, "eglGetDisplay failed: " + getEGLErrorString());
            return false;
        }

        // 2. 初始化EGL
        int[] version = new int[2]; // major, minor
        if (!EGL14.eglInitialize(eglDisplay, version, 0, version, 1)) {
            Log.e(TAG, "eglInitialize failed: " + getEGLErrorString());
            releaseInitialDisplay(); // 清理已获取的display
            return false;
        }
        Log.i(TAG, "EGL Initialized. Version: " + version[0] + "." + version[1]);

        // 3. 选择EGLConfig
        // 定义期望的配置属性
        int[] configAttribs = {
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                // EGL14.EGL_DEPTH_SIZE, 16, // 如果你需要深度缓冲
                // EGL14.EGL_STENCIL_SIZE, 8, // 如果你需要模板缓冲
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT, // 我们想要Pbuffer表面
                EGL14.EGL_NONE // 属性列表结束标记
        };
        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfigs = new int[1];
        if (!EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, configs.length, numConfigs, 0)) {
            Log.e(TAG, "eglChooseConfig failed: " + getEGLErrorString());
            release();
            return false;
        }
        if (numConfigs[0] == 0) {
            Log.e(TAG, "No suitable EGLConfig found.");
            release();
            return false;
        }
        eglConfig = configs[0];

        // 4. 创建EGLContext
        int[] contextAttribs = {
                EGL14.EGL_CONTEXT_CLIENT_VERSION, EGL_CONTEXT_CLIENT_VERSION,
                EGL14.EGL_NONE // 属性列表结束标记
        };
        eglContext = EGL14.eglCreateContext(eglDisplay, eglConfig, EGL14.EGL_NO_CONTEXT, contextAttribs, 0);
        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            Log.e(TAG, "eglCreateContext failed: " + getEGLErrorString());
            release();
            return false;
        }

        // 5. 创建EGLSurface (Pbuffer表面)
        // 对于Pbuffer，通常指定宽度和高度
        int[] pbufferAttribs = {
                EGL14.EGL_WIDTH, 1, // 最小1x1像素的Pbuffer
                EGL14.EGL_HEIGHT, 1,
                EGL14.EGL_NONE // 属性列表结束标记
        };
        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, eglConfig, pbufferAttribs, 0);
        if (eglSurface == EGL14.EGL_NO_SURFACE) {
            Log.e(TAG, "eglCreatePbufferSurface failed: " + getEGLErrorString());
            release();
            return false;
        }

        // 6. 将EGLContext和EGLSurface设置为当前线程的当前上下文/表面
        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) {
            Log.e(TAG, "eglMakeCurrent failed: " + getEGLErrorString());
            release();
            return false;
        }

        Log.i(TAG, "EGL context (GLES " + EGL_CONTEXT_CLIENT_VERSION + ") created and made current successfully.");
        return true;
    }

    /**
     * 释放所有EGL资源。
     */
    public void release() {
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            // 解绑当前上下文和表面
            EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);

            if (eglSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(eglDisplay, eglSurface);
                eglSurface = EGL14.EGL_NO_SURFACE;
            }
            if (eglContext != EGL14.EGL_NO_CONTEXT) {
                EGL14.eglDestroyContext(eglDisplay, eglContext);
                eglContext = EGL14.EGL_NO_CONTEXT;
            }
            EGL14.eglReleaseThread(); // 释放当前线程的EGL资源（如果适用）
            EGL14.eglTerminate(eglDisplay); // 终止EGL显示连接
        }
        eglDisplay = EGL14.EGL_NO_DISPLAY; // 重置
        Log.i(TAG, "EGL context released.");
    }

    /**
     * 仅在eglInitialize失败后，display已获取但未完全初始化时调用。
     */
    private void releaseInitialDisplay() {
        if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
            EGL14.eglTerminate(eglDisplay);
            eglDisplay = EGL14.EGL_NO_DISPLAY;
        }
    }

    /**
     * 获取EGL错误码对应的字符串描述。
     * @return EGL错误信息。
     */
    private String getEGLErrorString() {
        int error = EGL14.eglGetError();
        // android.opengl.GLUtils.getEGLErrorString(error) 可以将错误码转为字符串
        // 但为了最简化，这里只返回错误码
        return "EGL error: 0x" + Integer.toHexString(error);
    }

    // (可选) 获取当前EGLContext的方法，如果TFLite需要共享上下文
    public EGLContext getEglContext() {
        return eglContext;
    }

    public EGLDisplay getEglDisplay() {
        return eglDisplay;
    }
}