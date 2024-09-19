package com.example.ffmpegvideoplayer;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.GLES20;
import android.opengl.GLUtils;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.util.Log;
import android.view.TextureView;

import com.example.ffmpegvideoplayer.TextureView_filterEngine;

import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.egl.EGLSurface;

/*
 （1）要初始化OPENGLES相关的环境（这些是在GLSurfaceView中默认初始化的）
 （2）
 */
public class MyGLRenderer implements SurfaceTexture.OnFrameAvailableListener{
    private static final String TAG = "Filter_GLRenderer";
    private Context mContext;
    private HandlerThread mHandlerThread;
    private Handler mHandler;
    private TextureView mTextureView;

    // 设置两个id
    private int mOESTextureId_1; // SR patch
    private int mOESTextureId_2; // background
    private TextureView_filterEngine mFilterEngine; // 画图用的filter
    private FloatBuffer mDataBuffer;
    private int mShaderProgram = -1;
    private float[] transformMatrix = new float[16];

    private EGL10 mEgl = null;
    private EGLDisplay mEGLDisplay = EGL10.EGL_NO_DISPLAY; // no_display是什么鸟蛋？？
    private EGLContext mEGLContext = EGL10.EGL_NO_CONTEXT;
    private EGLConfig[] mEGLConfig = new EGLConfig[1];
    private EGLSurface mEglSurface;

    private static final int MSG_INIT = 1;
    private static final int MSG_RENDER = 2;
    private static final int MSG_DEINIT = 3;
    private SurfaceTexture mOESSurfaceTexture;
    private Bitmap test_bitmap;

    public void init(TextureView textureView, int oesTextureId_1 , int oesTextureId_2, Context context) {
        mContext = context;
        mTextureView = textureView;
        // 我有两个外部纹理
        mOESTextureId_1 = oesTextureId_1;
        mOESTextureId_2 = oesTextureId_2;

        mHandlerThread = new HandlerThread("Renderer Thread"); //单开一个线程进行渲染的
        mHandlerThread.start(); // 启动线程
        mHandler = new Handler(mHandlerThread.getLooper()){
            @Override
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case MSG_INIT: // 初始化 EGL
                        initEGL();
                        return;
                    case MSG_RENDER: // 进行画图
                        drawFrame();
                        return;
                    case MSG_DEINIT: // 不知道进行什么
                        return;
                    default:
                        return;
                }
            }
        };
        mHandler.sendEmptyMessage(MSG_INIT);

        final BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;	// No pre-scaling 不进行预缩放（默认情况下，Android会根据设备的分辨率和你放置图片的资源文件目录而预先缩放位图。我们不希望Android根据我们的情况对位图进行缩放，因此我们将inScaled设置为false）

        test_bitmap = BitmapFactory.decodeResource(context.getResources(), R.drawable.wechat_pic, options);
    }

    private void initEGL() {
        mEgl = (EGL10) EGLContext.getEGL();

        //获取显示设备
        mEGLDisplay = mEgl.eglGetDisplay(EGL10.EGL_DEFAULT_DISPLAY);
        if (mEGLDisplay == EGL10.EGL_NO_DISPLAY) {
            throw new RuntimeException("eglGetDisplay failed! " + mEgl.eglGetError());
        }

        //version中存放EGL版本号
        int[] version = new int[2];

        //初始化EGL
        if (!mEgl.eglInitialize(mEGLDisplay, version)) {
            throw new RuntimeException("eglInitialize failed! " + mEgl.eglGetError());
        }

        //构造需要的配置列表
        int[] attributes = {
                EGL10.EGL_RED_SIZE, 8,
                EGL10.EGL_GREEN_SIZE,8,
                EGL10.EGL_BLUE_SIZE, 8,
                EGL10.EGL_ALPHA_SIZE, 8,
                EGL10.EGL_BUFFER_SIZE, 32,
                EGL10.EGL_RENDERABLE_TYPE, 4,
                EGL10.EGL_SURFACE_TYPE, EGL10.EGL_WINDOW_BIT,
                EGL10.EGL_NONE
        };
        int[] configsNum = new int[1];

        //EGL选择配置
        if (!mEgl.eglChooseConfig(mEGLDisplay, attributes, mEGLConfig, 1, configsNum)) {
            throw new RuntimeException("eglChooseConfig failed! " + mEgl.eglGetError());
        }
        SurfaceTexture surfaceTexture = mTextureView.getSurfaceTexture(); // 这个是 TextureView中内置的SurfaceTexture
        if (surfaceTexture == null)
            return;

        //创建EGL显示窗口
        mEglSurface = mEgl.eglCreateWindowSurface(mEGLDisplay, mEGLConfig[0], surfaceTexture, null);

        //创建上下文
        int[] contextAttribs = {
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
                EGL10.EGL_NONE
        };
        mEGLContext = mEgl.eglCreateContext(mEGLDisplay, mEGLConfig[0], EGL10.EGL_NO_CONTEXT, contextAttribs);

        if (mEGLDisplay == EGL10.EGL_NO_DISPLAY || mEGLContext == EGL10.EGL_NO_CONTEXT){
            throw new RuntimeException("eglCreateContext fail failed! " + mEgl.eglGetError());
        }

        if (!mEgl.eglMakeCurrent(mEGLDisplay,mEglSurface, mEglSurface, mEGLContext)) {
            throw new RuntimeException("eglMakeCurrent failed! " + mEgl.eglGetError());
        }

        mFilterEngine = new TextureView_filterEngine( mContext,mOESTextureId_1,mOESTextureId_2);
//        mDataBuffer = mFilterEngine.getBuffer();
//        mShaderProgram = mFilterEngine.getShaderProgram();
    }

    private void drawFrame() {
        long t1, t2;
        t1 = System.currentTimeMillis();
        if (mOESSurfaceTexture != null) {
            mOESSurfaceTexture.updateTexImage();
            mOESSurfaceTexture.getTransformMatrix(transformMatrix);
        }
        mEgl.eglMakeCurrent(mEGLDisplay, mEglSurface, mEglSurface, mEGLContext); // 什么意思这
        GLES20.glViewport(0,0,mTextureView.getWidth(),mTextureView.getHeight()); // 这里获取的是textureView的宽度和高度？——那我是不是可以设置TextureView的宽度和高度？
        Log.i("test","w: " + mTextureView.getWidth()+ " h: " + mTextureView.getHeight());
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        GLES20.glClearColor(1f, 1f, 0f, 0f);
//        mFilterEngine.drawTexture(transformMatrix); // 在这里调用的GLES20的接口，并且进行了一系列的初始化着色器的操作
        mFilterEngine.drawframe(); // 在这里调用的GLES20的接口，并且进行了一系列的初始化着色器的操作——我没有传递transformMatrix参数，看看会发生什么

        mEgl.eglSwapBuffers(mEGLDisplay, mEglSurface); // mEGLDisplay 和 EglSurface是前缓冲区和后缓冲区的区别吗？？
        t2 = System.currentTimeMillis();
        Log.i(TAG, "drawFrame: time = " + (t2 - t1));
    }

    @Override
    public void onFrameAvailable(SurfaceTexture surfaceTexture) {
        if (mHandler != null) {
            mHandler.sendEmptyMessage(MSG_RENDER);
        }
    }

    // 我要把这个surfaceTexture显示出来吧？？
    public SurfaceTexture initOESTexture() {
        mOESSurfaceTexture = new SurfaceTexture(mOESTextureId_2); // 创建一个SurfaceTexture对象，关联到一个已有的纹理ID
        mOESSurfaceTexture.setOnFrameAvailableListener(this); // 设置这个 SurfaceTexture 的监听器，使得当新的帧数据可用时，当前类会接收到通知
        return mOESSurfaceTexture;
    }
    public void set_bi_texture(Bitmap bi_pic){
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,mOESTextureId_2);
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D,0,bi_pic,0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE,0);
    }
    public void set_sr_texture(Bitmap sr_pic){
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,mOESTextureId_1);
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D,0,sr_pic,0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE,0);
    }
}
