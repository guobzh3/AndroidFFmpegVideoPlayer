package com.example.ffmpegvideoplayer; // 这里引入了这个ffmpegVIdeoplayer的库

import android.app.ActivityManager;
import android.content.Context;
import android.content.pm.ConfigurationInfo;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.BitmapShader;
import android.graphics.Shader;
import android.graphics.RectF;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
//import android.renderscript.ScriptIntrinsicResize;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.ImageView;
import android.widget.TextView;

//import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.io.InputStream;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import com.example.ffmpegvideoplayer.analysis.InferenceTFLite;
import android.util.Size;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

// renderscript 相关
//import android.content.Context;
//import android.graphics.Bitmap;
//import android.renderscript.Allocation;
//import android.renderscript.Element;
//import android.renderscript.RenderScript;
//import android.renderscript.ScriptC;

import android.opengl.GLES31;
import android.opengl.GLSurfaceView;
import android.opengl.GLES20;


public class MainActivity extends AppCompatActivity {

    private static final int QUEUE_CAPACITY = 32;
    // 这个队列不能开太大，如果是 540P的图像数据的话，一个int[540*960]大小大概为2MB
    // 当时设置队列为4096的话程序运行一段时间就炸，因为内存爆了，当程序运行内存超过700MB以后就会Out of memory
    // 所以要设小点
    // BlockingQueue：线程安全的阻塞队列
    // 通过队列来构建流水线
    private static BlockingQueue<int[]> rgbBytesQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TensorImage> modelInputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TensorBuffer> modelOutputQueue = new ArrayBlockingQueue<>(8);
    private static BlockingQueue<int[]> viewOutQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    // 存储 BISR 后的结果
    private final static String mytag = "MyNativeCode";
    private final static String time_tag = "time";
    private static BlockingQueue<Bitmap> biSROutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    // width, height
    // 270p -> 540P
    private final Size video_output_shape = new Size(3840, 2160);
    private static int[] video_input_shape = new int[] {960, 540};
    private static int[] tf_input_shape = new int[] {480,270};
    private static int[] tf_output_shape = new int[] {1920,1080};

    private static  int[] tile_index = new int[] {1,1};
    private static int[] tile_split = new int[] {video_input_shape[0] / 2 , video_input_shape[1] / 2}; // width ， height
//    private static int[] video_input_shape = new int[] {480, 270};
//    private final Size video_output_shape = new Size(960, 540);

    // 静态代码块，加载一个本地动态库
    static {
        System.loadLibrary("ffmpegvideoplayer"); // libname 是类名
    }

    private SurfaceView surfaceView;

    private GLSurfaceView glsurfacebview;
    private SurfaceHolder surfaceHolder;
    private ImageView imageView;
    private Handler handler;

    // 创建一个GLSurfaceView
    private GLSurfaceView mGLSurfaceView;

    private TextView frameSizeTextView;
    private TextView fpsTextView;
    private boolean isPICO = true ;
    private InferenceTFLite srTFLite;
    private int[] outPixels;

    private MyRenderer renderer;
    Bitmap inputBitmap;
    Bitmap outputBitmap;

    ImageProcessor imageProcessor ;
//    public Bitmap applyBlur(Context context, Bitmap inputBitmap, float blurRadius) {
//        // 创建RenderScript实例
//        RenderScript rs = RenderScript.create(context);
//
//        // 创建输入和输出Allocation
//        Allocation inputAllocation = Allocation.createFromBitmap(rs, inputBitmap);
//        Allocation outputAllocation = Allocation.createTyped(rs, inputAllocation.getType());
//
//        // 加载RenderScript脚本
//
////        ScriptC_try blurScript = new ScriptC_try(rs);
////        blurScript.set_inImage(inputAllocation);
////        blurScript.set_outImage(outputAllocation);
////        blurScript.set_blurRadius(blurRadius);
////
////        // 执行模糊操作
////        blurScript.invoke_root();
//
//        // 创建输出位图并复制数据
//        Bitmap outputBitmap = Bitmap.createBitmap(inputBitmap.getWidth(), inputBitmap.getHeight(), inputBitmap.getConfig());
//        outputAllocation.copyTo(outputBitmap);
//
//        // 销毁资源和清理
//        inputAllocation.destroy();
//        outputAllocation.destroy();
//        rs.destroy();
//
//        return outputBitmap;
//    }

    private void initModel() {
        Log.i("dada","initModel. ispico : " + (this.isPICO));

        try {
            this.srTFLite = new InferenceTFLite();

            if (this.isPICO) {
                this.srTFLite.addGPUDelegate(); // pico的话，使用GPU代理
                Log.i("dada","addGPUDelegate");

            }
            else {
                Log.i("dada","addNNApiDelegate");
                this.srTFLite.addNNApiDelegate();
            }
//            this.srTFLite.addNNApiDelegate();

            this.srTFLite.initialModel(this);
        } catch (Exception e) {
            Log.e("Error Exception", "MainActivity initial model error: " + e.getMessage() + e.toString());
        }
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
//        EdgeToEdge.enable(this); // 启用一个无边框的沉浸式布局

        // 初始化GLSurfaceView
        mGLSurfaceView = new GLSurfaceView(this);
        final ActivityManager activityManager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
        final ConfigurationInfo configurationInfo = activityManager.getDeviceConfigurationInfo();
        final boolean supportsEs2 = configurationInfo.reqGlEsVersion >= 0x20000;

        if (supportsEs2)
        {
            // Request an OpenGL ES 2.0 compatible context.
            mGLSurfaceView.setEGLContextClientVersion(2);

            // Set the renderer to our demo renderer, defined below.
            // 我只需要修改这个setrenderer就可以展示我自己设计的renderer了 ！yes！
//			mGLSurfaceView.setRenderer(new LessonFourRenderer(this));
            renderer = new MyRenderer(this);
            mGLSurfaceView.setRenderer(renderer);
            mGLSurfaceView.setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
            Log.i("gles","support GLES");
        }
        else{
            Log.e("gles","not support GLES");
            return ;
        }
//        LinearLayout layout = findViewById(R.id.layout);
        setContentView(mGLSurfaceView);

//        setContentView(R.layout.activity_main);
//        glsurfacebview = findViewById(R.id.glsurfaceview);
//        glsurfacebview.setRenderer(renderer);
        // params
//        surfaceView = findViewById(R.id.surfaceView);
//        surfaceHolder = surfaceView.getHolder(); //获取对低层surface的访问
//        imageView = findViewById(R.id.imageView);
        /**
         * 用于线程管理：
             * Looper 是一个循环，负责管理和分发线程中的消息和任务队列。
             * 每个线程都可以拥有一个 Looper 对象，主线程（UI 线程）默认已经初始化了一个 Looper。
             * Looper 通过 Looper.prepare() 和 Looper.loop() 方法进行初始化和开始循环。
         * 这行代码的作用是创建一个与主线程（UI 线程）Looper 关联的 Handler 对象
         */
        handler = new Handler(Looper.getMainLooper());

        outPixels = new int [video_output_shape.getHeight() * video_output_shape.getWidth()];
        inputBitmap = Bitmap.createBitmap(video_input_shape[0], video_input_shape[1], Bitmap.Config.ARGB_8888); // 输入的图片(先宽后高）
        outputBitmap = Bitmap.createBitmap(video_output_shape.getWidth(), video_output_shape.getHeight(), Bitmap.Config.ARGB_8888); // 输出的图片

        // 出初始化对图像进行预处理的类
        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(tf_input_shape[1], tf_input_shape[0], ResizeOp.ResizeMethod.BILINEAR)) // (先高后宽）
                .add(new NormalizeOp(0, 255))
                .build();

//        bilinear_processor = new ImageProcessor.Builder()
//                .add(new ResizeOp(video_output_shape.getHeight() , video_output_shape.getWidth(),ResizeOp.ResizeMethod.BILINEAR))
//                .build();

        // 模型初始化
        initModel();

        // Decoding Threads：应该是调用的c++的ffmpeg接口，后面再去理解这部分，反正这里就是单开了一个线程来进行视频的解码
        new Thread(new Runnable() {
            @Override
            public void run() {
//                R.string.video_url
//                R.color.black
                mainDecoder(getString(R.string.video_url));

            }
        }).start();

//        ReceiveSRShow2();
        // 主要处理程序
        mainProcess();
//        surfaceHolder.addCallback(new SurfaceHolder.Callback() {
//            @Override
//            public void surfaceCreated(@NonNull SurfaceHolder holder) {
////               ReceiveAndShow();
//                ReceiveSRShow();
////                mainProcess();
//            }
//
//            @Override
//            public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int width, int height) {
//
//            }
//
//            @Override
//            public void surfaceDestroyed(@NonNull SurfaceHolder holder) {
//
//            }
//        });
    }
    public void ReceiveAndShow() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    try {
                        long startTime = System.currentTimeMillis();
                        int [] rgbData = rgbBytesQueue.take();
                        Bitmap rgbBitmap = Bitmap.createBitmap(video_input_shape[0], video_input_shape[1], Bitmap.Config.ARGB_8888);
                        // 用rgbData中的数据来填充这个bitmap
                        rgbBitmap.setPixels(rgbData, 0, video_input_shape[0], 0, 0, video_input_shape[0], video_input_shape[1]);
                        Matrix matrix = new Matrix();
                        if (!isPICO) {
                            matrix.postRotate(90);
                        }
                        Bitmap postTransformImageBitmap = Bitmap.createBitmap(rgbBitmap, 0, 0, video_input_shape[0], video_input_shape[1], matrix, false);
                        int outHeight = video_input_shape[1];
                        int outWidth = video_input_shape[0];
                        if (isPICO) {
                            outHeight = video_input_shape[0];
                            outWidth = video_input_shape[1];
                        }

                        Canvas canvas = surfaceHolder.lockCanvas();
                        if (canvas != null) {
                            try{
                                canvas.drawBitmap(postTransformImageBitmap, null, new Rect(0, 0, outHeight, outWidth), null);
                            } finally {
                                surfaceHolder.unlockCanvasAndPost(canvas);
                            }
                        }

                        long endTime = System.currentTimeMillis();
                        long costTime = endTime - startTime;
//                        updateTextView(Long.toString(costTime) + "ms");
                    } catch (InterruptedException e) {
                        Log.e("Error Exception", "MainActivity error: " + e.getMessage() + e.toString());
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }).start();
    }

    //tips: 学习多线程用队列进行流水线的写法
    public void mainProcess() {
        // 从rgbBytesQueue 中 拿数据 转成 模型推理的格式，并且将数据放入到biSRQueue 中
        preProcess();

        // 从modelInputQueue 中 拿数据推理
        inference();

        // 从BISRInputQueue 中 拿数据进行BISR，然后将输出存放到 BiSROutputQueue 中
//        bisr();

        // 从modelOutputQueue 中 拿数据转换,同时与 BiSROutputQueue 中的数据进行拼接
        afterProcess2();
//        afterProcess3();
        // 从viewOutQueue 中 拿数据输出
//        viewProcess();
    }

    public void preProcess() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    try {
                        long start_time , end_time , cost_time;
                        start_time = System.currentTimeMillis();
                        int[] rgbData = rgbBytesQueue.take(); // 把RGB数据拿出来
                        // 进行预处理
                        // 只提取 480 * 270 的部分来构建modelInput
                        // 验证了，这部分是没问题的
                        int[] model_input = new int[tf_input_shape[0] * tf_input_shape[1]];

                        int w_start = tile_index[0] * tf_input_shape[0];
                        int h_start = tile_index[1] * tf_input_shape[1];
                        Log.i(mytag, "w_start " + w_start + " h_start " + h_start + " tf_input_shape[0] " + tf_input_shape[0] + " tf_input_shape[1] " + tf_input_shape[1] + " tile_split[0]" + tile_split[0] + " tile_split[1]" + tile_split[1]);
                        int k = 0;
                        for (int i = h_start;i < h_start + tile_split[1] ;i++){
                            for (int j = w_start ; j < w_start + tile_split[0] ;j++){
                                int index = i * video_input_shape[0] + j;
                                model_input[k++] = rgbData[index];
                                // 打印出来检查一下
//                                if (j == w_start ){
//                                    Log.i(mytag, "i=" + i + " j = " + j + " index =  " + index + " k =  " + k + " data: "+ rgbData[index]);
//                                }

                            }
                        }
                        Bitmap model_input_bitmap= Bitmap.createBitmap(tf_input_shape[0] , tf_input_shape[1] , Bitmap.Config.ARGB_8888);
                        model_input_bitmap.setPixels(model_input , 0 , tf_input_shape[0],0,0,tf_input_shape[0],tf_input_shape[1]);
                        end_time = System.currentTimeMillis();
                        // TODO: 接着写
                        TensorImage modelInput = new TensorImage(DataType.FLOAT32);
                        modelInput.load(model_input_bitmap);
                        modelInput = imageProcessor.process(modelInput); // 进行预处理
                        cost_time = end_time - start_time ;
                        Log.i(time_tag , "create model_input : " + cost_time + " ms");
                        // 预处理后放到队列中
                        try {
                            modelInputQueue.put(modelInput);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                        // TODO：对剩下的数据进行bi放大，并放到bioutputQueue中
                        // 初始化bitmap并放大bitmap中
                        Bitmap rgbBitmap = Bitmap.createBitmap(video_input_shape[0], video_input_shape[1], Bitmap.Config.ARGB_8888);
                        rgbBitmap.setPixels(rgbData, 0, video_input_shape[0],  0,  0, video_input_shape[0], video_input_shape[1]);
//                        BitmapFactory.de
//                        start_time = System.currentTimeMillis();
                        // 1. 使用Bitmap.createScaledBitmap 接口进行bilinear放大
//                        Bitmap dst = Bitmap.createScaledBitmap(rgbBitmap, video_output_shape.getWidth(),video_output_shape.getHeight() , true);
//                        end_time = System.currentTimeMillis();
//                        cost_time = end_time - start_time;
//                        Log.i(time_tag , "bilinear time: " + cost_time + " ms");
                        // 存放到biSROutputQueue中
                        try {
                            biSROutputQueue.put(rgbBitmap);
//                            biSROutputQueue.put(dst);
//                            biSROutputQueue.put(model_input_bitmap);
                            Log.i(mytag, "bisroutputqueue size: " + biSROutputQueue.size());
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }


                    } catch (InterruptedException e) {
                        Log.e("Error Exception", "MainActivity error: " + e.getMessage() + e.toString());
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }).start();
    }

    public void inference() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    try {
                        // 从modelInputQueue 中拿数据，进行推理
                        TensorImage modelInput = modelInputQueue.take();
                        long start = System.currentTimeMillis();
                        TensorBuffer modelOutput = srTFLite.superResolution(modelInput , tf_output_shape);
                        long end = System.currentTimeMillis();
                        long cost_time = end - start;
                        Log.i(time_tag , "inference time: " + cost_time + " ms");
                        Log.i(mytag , "input: " + modelInput.getWidth() + " " + modelInput.getHeight()); // 480 270
                        Log.i(mytag , "output "+ modelOutput.getShape()[0] + " "+modelOutput.getShape()[1]); // 1 540 960 3
                        // 将得到的tensorbuffer放入队列中，等待后面的后处理
                        try {
                            modelOutputQueue.put(modelOutput);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    } catch (InterruptedException e) {
                        Log.e("Error Exception", "MainActivity error: " + e.getMessage() + e.toString());
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }).start();
    }

    public void afterProcess() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    try {
                        TensorBuffer hwcOutputTensorBuffer = modelOutputQueue.take();
                        float[] hwcOutputData = hwcOutputTensorBuffer.getFloatArray();
                        int outHeight = video_output_shape.getHeight();
                        int outWidth = video_output_shape.getWidth();
                        int[] pixels = new int [outHeight * outWidth];
                        int yp = 0;
                        for (int h = 0; h < outHeight; h++) {
                            for (int w = 0; w < outWidth; w++) {
                                int r = (int) (hwcOutputData[h * outWidth * 3 + w * 3] * 255);
                                int g = (int) (hwcOutputData[h * outWidth * 3 + w * 3 + 1] * 255);
                                int b = (int) (hwcOutputData[h * outWidth * 3 + w * 3 + 2] * 255);
                                r = r > 255 ? 255 : (Math.max(r, 0));
                                g = g > 255 ? 255 : (Math.max(g, 0));
                                b = b > 255 ? 255 : (Math.max(b, 0));
                                pixels[yp++] = 0xff000000 | (r << 16 & 0xff0000) | (g << 8 & 0xff00) | (b & 0xff);
                            }
                        }
                        try {
                            viewOutQueue.put(pixels);
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    } catch (InterruptedException e) {
                        Log.e("Error Exception", "MainActivity error: " + e.getMessage() + e.toString());
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }).start();
    }
    //后处理模块
    public void afterProcess2() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    try {
                        TensorBuffer hwcOutputTensorBuffer = modelOutputQueue.take();
                        Log.i(mytag , "hwcOutputTensorBuffer " + hwcOutputTensorBuffer.getShape()[0] + " " + hwcOutputTensorBuffer.getShape()[1] + " " +
                                hwcOutputTensorBuffer.getShape()[2] + " " + hwcOutputTensorBuffer.getShape()[3]);
                        long start_time , end_time , cost_time ;
                        start_time = System.currentTimeMillis();
                        float[] hwcOutputData = hwcOutputTensorBuffer.getFloatArray(); // 首先从tensorbuffer中获取浮点数组

                        int outHeight = tf_output_shape[0]; //
                        int outWidth = tf_output_shape[1];

                        // 将浮点数据转化成int[] 的ARGB数据
                        int yp = 0;
                        for (int h = 0; h < outHeight; h++) {
                            for (int w = 0; w < outWidth; w++) {
                                // 下标越界了这里
                                int r = (int) (hwcOutputData[h * outWidth * 3 + w * 3] * 255);
                                int g = (int) (hwcOutputData[h * outWidth * 3 + w * 3 + 1] * 255);
                                int b = (int) (hwcOutputData[h * outWidth * 3 + w * 3 + 2] * 255);
                                r = r > 255 ? 255 : (Math.max(r, 0));
                                g = g > 255 ? 255 : (Math.max(g, 0));
                                b = b > 255 ? 255 : (Math.max(b, 0));
                                // ARGB
                                outPixels[yp++] = 0xff000000 | (r << 16 & 0xff0000) | (g << 8 & 0xff00) | (b & 0xff);
                            }
                        }
                        Bitmap srBitmap = Bitmap.createBitmap(tf_output_shape[1],tf_output_shape[0], Bitmap.Config.ARGB_8888);
                        srBitmap.setPixels(outPixels,0,tf_output_shape[1],0,0,tf_output_shape[1],tf_output_shape[0]);

                        end_time = System.currentTimeMillis();
                        cost_time = end_time - start_time;
                        Log.i(time_tag , "inference post_process time: " + cost_time + " ms");
                        // TODO：从bisr队列中获取bi放大的bitmap，然后将outPixels填充到对应的位置中（不知道能不能直接填充）
                        Bitmap outBitmap = biSROutputQueue.take();
                        renderer.set_bi_Bitmap(outBitmap);
                        renderer.set_sr_bitmap(srBitmap);
                        renderer.updateSurface_Flag = true;
                        mGLSurfaceView.requestRender();

//                        renderer.readBufferPixelToBitmap(3840,2160);
//                        Context.getExternalFilesDir();
                        start_time = System.currentTimeMillis();
//                        int bitmapWidth = outBitmap.getWidth();
//                        int bitmapHeight = outBitmap.getHeight();
//                        Log.i(mytag, "run: " + bitmapWidth +  ' ' + bitmapHeight); // 3840 2160
//                        Log.i(mytag , "x: "+tile_index[0] * tf_output_shape[0] + " width: "+tf_output_shape[0] + " bitmap_width: "+ outBitmap.getWidth());
                        // stride 要填写要填入的patch的width值
                        // 这句话是把sr patch拷贝到outBitmap对应的位置，暂时注释掉
//                        outBitmap.setPixels(outPixels , 0 , tf_output_shape[0] , tile_index[0] * tf_output_shape[0]  , tile_index[1] * tf_output_shape[1]  , tf_output_shape[0] , tf_output_shape[1]);
//                        outBitmap.setPixels(outPixels , 0 , tf_output_shape[0] , tile_index[0] * tf_output_shape[0] , tile_index[1] * tf_output_shape[1] , tf_output_shape[0] , tf_output_shape[1]);

                     // 这里有问题-> 单独把这部分的东西显示出来看看（这里推理完是没错的，应该就是setpixels的问题 ）
//                        Bitmap inference_bitmap = Bitmap.createBitmap(3840 , 2160  , Bitmap.Config.ARGB_8888);
//                        inference_bitmap.setPixels(outPixels , 0 , 960 , 0,0,960 , 540);
                        // ARGB数据写入到bitmap中
//                        Bitmap outBitmap = Bitmap.createBitmap(outWidth, outHeight, Bitmap.Config.ARGB_8888);
//                        outBitmap.setPixels(outPixels, 0, outWiddth, 0, 0, outWidth, outHeight);

                        // 对bitmap数据进行后处理
                        Matrix matrix = new Matrix();
                        if (!isPICO) { // 在手机显示的话：需要旋转成竖屏
                            matrix.postRotate( 0);
                        }
//                        Bitmap postTransformImageBitmap = Bitmap.createBitmap(outBitmap, 0, 0, video_output_shape.getWidth(), video_output_shape.getHeight(), matrix, false);
//                        Bitmap postTransformImageBitmap = Bitmap.createBitmap(outBitmap, 0, 0, outBitmap.getWidth(),outBitmap.getHeight(), matrix, false);

                        // 使用创建的 Handler 对象，将一个任务（更新 ImageView 的图像）发送到主线程执行。
                        // tips：在 Android 中，所有的 UI 操作必须在主线程（UI 线程）上进行。如果你在后台线程（如异步任务或网络操作线程）上尝试更新 UI，会导致应用崩溃。这是因为 Android 的 UI 组件不是线程安全的。
                        // 只有主线程可以更新视图
//                        handler.post(()-> imageView.setImageBitmap(postTransformImageBitmap));
                        end_time = System.currentTimeMillis();
                        cost_time = end_time - start_time;
//                        long endTime = System.currentTimeMillis();
//                        long costTime = endTime - startTime;
//                        updateTextView(Long.toString(cost_time) + "ms");
                        Log.i(time_tag, "concat and show time : " + cost_time + " ms");
                    }
                    catch (InterruptedException e) {
                        Log.e("Error Exception", "MainActivity error: " + e.getMessage() + e.toString());
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }).start();
    }
    public void viewProcess() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    try {
                        long startTime = System.currentTimeMillis();
                        int[] outputPixels = viewOutQueue.take();
                        int outHeight = video_output_shape.getHeight();
                        int outWidth = video_output_shape.getWidth();
                        outputBitmap.setPixels(outputPixels, 0, outWidth, 0, 0, outWidth, outHeight);
                        Matrix matrix = new Matrix();
                        if (!isPICO) {
                            matrix.postRotate(90);
                        }
                        Bitmap postTransformImageBitmap = Bitmap.createBitmap(outputBitmap, 0, 0, outWidth, outHeight, matrix, false);
                        if (!isPICO) {
                            int tmp = outWidth;
                            outWidth = outHeight;
                            outHeight = tmp;
                        }
                        Canvas canvas = surfaceHolder.lockCanvas();
                        if (canvas != null) {
                            try{
                                canvas.drawBitmap(postTransformImageBitmap, null, new Rect(0, 0, outWidth, outHeight), null);
                            } finally {
                                surfaceHolder.unlockCanvasAndPost(canvas);
                            }
                        }
                        long endTime = System.currentTimeMillis();
                        updateTextView(Long.toString(endTime - startTime) + "ms");
                        Log.i("viewProcess:", Long.toString(endTime - startTime) + "ms");
                    } catch (InterruptedException e) {
                        Log.e("Error Exception", "MainActivity error: " + e.getMessage() + e.toString());
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }).start();
    }
    public void ReceiveSRShow() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    try {
                        long startTime = System.currentTimeMillis();
                        int [] rgbData = rgbBytesQueue.take();
                        inputBitmap.setPixels(rgbData, 0, video_input_shape[0], 0, 0, video_input_shape[0], video_input_shape[1]);
                        int[] outputSize = srTFLite.getOUTPUT_SIZE();
                        int outWidth = outputSize[2];
                        int outHeight = outputSize[1];

//                        int[] outPixels = srTFLite.superResolution(rgbBitmap);
                        srTFLite.superResolution(inputBitmap, outPixels);

                        outputBitmap.setPixels(outPixels, 0, outWidth, 0, 0, outWidth, outHeight);
                        Matrix matrix = new Matrix();
                        if (!isPICO) {
                            matrix.postRotate(90);
                        }
                        Bitmap postTransformImageBitmap = Bitmap.createBitmap(outputBitmap, 0, 0, outWidth, outHeight, matrix, false);
                        if (!isPICO) {
                            int tmp = outWidth;
                            outWidth = outHeight;
                            outHeight = tmp;
                        }
                        Canvas canvas = surfaceHolder.lockCanvas();
                        if (canvas != null) {
                            try{
                                canvas.drawBitmap(postTransformImageBitmap, null, new Rect(0, 0, outWidth, outHeight), null);
                            } finally {
                                surfaceHolder.unlockCanvasAndPost(canvas);
                            }
                        }
                        long endTime = System.currentTimeMillis();
                        long costTime = endTime - startTime;
                        updateTextView(Long.toString(costTime) + "ms");
                    } catch (InterruptedException e) {
                        Log.e("Error Exception", "MainActivity error: " + e.getMessage() + e.toString());
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }).start();
    }

    public void ReceiveSRShow2() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    try {
                        long startTime = System.currentTimeMillis();
                        int [] rgbData = rgbBytesQueue.take();
                        Bitmap rgbBitmap = Bitmap.createBitmap(video_input_shape[0], video_input_shape[1], Bitmap.Config.ARGB_8888);
                        rgbBitmap.setPixels(rgbData, 0, video_input_shape[0], 0, 0, video_input_shape[0], video_input_shape[1]);
                        int[] outputSize = srTFLite.getOUTPUT_SIZE();
                        int outWidth = outputSize[2];
                        int outHeight = outputSize[1];


//                        int[] outPixels = srTFLite.superResolution(rgbBitmap);
                        srTFLite.superResolution(rgbBitmap, outPixels);

                        Bitmap outBitmap = Bitmap.createBitmap(outWidth, outHeight, Bitmap.Config.ARGB_8888);
                        outBitmap.setPixels(outPixels, 0, outWidth, 0, 0, outWidth, outHeight);

                        Matrix matrix = new Matrix();
                        if (!isPICO) {
                            matrix.postRotate(90);
                        }
                        Bitmap postTransformImageBitmap = Bitmap.createBitmap(outBitmap, 0, 0, outWidth, outHeight, matrix, false);
                        // 先注释掉，使用GLSurfaceView来进行视图更新
//                        handler.post(()-> imageView.setImageBitmap(postTransformImageBitmap));
                        long endTime = System.currentTimeMillis();
                        long costTime = endTime - startTime;
                        updateTextView(Long.toString(costTime) + "ms");

                    } catch (InterruptedException e) {
                        Log.e("Error Exception", "MainActivity error: " + e.getMessage() + e.toString());
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }).start();
    }


    public void updateTextView(String fps) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                frameSizeTextView = findViewById(R.id.frame_size);
                fpsTextView = findViewById(R.id.inference_time);
                fpsTextView.setText(fps);
            }
        });
    }
    public static void putData(int[] data) {
        try {
            Log.i("rgbQueue", "pushing data");
            // 这是一个int的数组
            rgbBytesQueue.put(data); // 在jni代码中调用了这个函数，将解码出来的图片存储到了 rgbByteQueue队列中
        }
        catch (InterruptedException e) {
            Log.i("rgbQueue", "pushing data error!");
            Thread.currentThread().interrupt();
        }
    }

    // 声明本地方法
    public native void mainDecoder(String url);
}