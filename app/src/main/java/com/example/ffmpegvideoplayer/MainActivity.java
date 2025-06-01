




package com.example.ffmpegvideoplayer;

import android.graphics.Bitmap;
// import android.graphics.BitmapFactory; // 未使用
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Canvas;
import android.graphics.Paint; // 如果 model_input_bitmap 绘制需要，则保留。
// import android.graphics.BitmapShader; // 未使用
// import android.graphics.Shader; // 未使用
// import android.graphics.RectF; // 未使用
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.renderscript.ScriptIntrinsicResize;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
// import androidx.core.graphics.Insets; // 用于 EdgeToEdge，如果其他地方使用则保留
// import androidx.core.view.ViewCompat; // 用于 EdgeToEdge
// import androidx.core.view.WindowInsetsCompat; // 用于 EdgeToEdge

// import java.io.InputStream; // 未使用
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicLong;
import com.example.ffmpegvideoplayer.analysis.InferenceTFLite;
import android.util.Size;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
// import org.tensorflow.lite.support.image.ops.ResizeOp; // 将从 MainActivity 的 ImageProcessor 中移除
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
// import android.renderscript.ScriptC; // 未直接使用

// import android.opengl.GLES31; // 用于 GLSurfaceView，保留
// import android.opengl.GLSurfaceView; // 用于 GLSurfaceView，保留
// import android.opengl.GLES20; // 用于 GLSurfaceView，保留

import com.example.ffmpegvideoplayer.analysis.BiTFLite; // 保留，已初始化

public class MainActivity extends AppCompatActivity {

    private static final int QUEUE_CAPACITY = 64;
    private static BlockingQueue<TaggedData<byte[]>> yuvBytesQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TaggedData<Bitmap>> rgbFrameQueueForTfInput = new ArrayBlockingQueue<>(QUEUE_CAPACITY); // 用于 prepareTfInputLoop
    private static BlockingQueue<TaggedData<Bitmap>> rgbFrameQueueForUpsample = new ArrayBlockingQueue<>(QUEUE_CAPACITY); // 用于 upsampleLoop
    private static BlockingQueue<TaggedData<TensorImage>> modelInputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TaggedData<TensorBuffer>> modelOutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    // private static BlockingQueue<TaggedData<int[]>> viewOutQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY); // 在提供的逻辑中未使用
    private static BlockingQueue<TaggedData<Bitmap>> biSROutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);

    private static final AtomicLong frameCounter = new AtomicLong(0); // 全局帧号计数器

    private final static String mytag = "MyNativeCode";
    private final static String time_tag = "time";

    // Define input/output dimensions clearly
    // 视频输入尺寸（来自解码器）
    private static final int VIDEO_INPUT_W = 1920; // 原始: video_input_shape[0]
    private static final int VIDEO_INPUT_H = 1024; // 原始: video_input_shape[1]

    // TFLite 模型输入patch尺寸（从视频输入裁剪）
    private static final int TF_INPUT_W = 480;    // 原始: tf_input_shape[0]
    private static final int TF_INPUT_H = 270;    // 原始: tf_input_shape[1]

    // TFLite 模型输出patch尺寸（超分辨率）
    private static final int TF_OUTPUT_W = 960;   // 原始: tf_output_shape[0]
    private static final int TF_OUTPUT_H = 540;   // 原始: tf_output_shape[1]
    private static final int[] TF_OUTPUT_SHAPE = new int[]{TF_OUTPUT_W, TF_OUTPUT_H}; // 用于传递给 TFLite 模型

    // 最终视频输出尺寸（用于显示，SR patch放置在此处）
    private final Size video_output_shape = new Size(3840, 2048); // Width, Height

    // 平铺处理（保留原始逻辑）
    private static final int[] tile_index = new int[]{1, 1}; // {tile_x_index, tile_y_index}
    // private static int[] tile_split = new int[] {VIDEO_INPUT_W / 2 , VIDEO_INPUT_H / 2}; // SRx4 示例
    // private static int[] tile_split = new int[] {VIDEO_INPUT_W / 4 , VIDEO_INPUT_H / 4}; // SRx2 示例（原始，但在裁剪中未使用）

    static {
        System.loadLibrary("ffmpegvideoplayer");
    }

    private SurfaceView surfaceView;
    // private SurfaceHolder surfaceHolder; // 已初始化但在 mainProcess 中未直接使用
    private ImageView imageView;
    private Handler handler;
    // private GLSurfaceView mGLSurfaceView; // 已初始化但在 mainProcess 中未直接使用

    private TextView frameSizeTextView;
    private TextView fpsTextView;
    private boolean isPICO = false; // 配置标志

    private final static String deligater="gpu";

    private InferenceTFLite srTFLite;
    private BiTFLite biTFLite; // 已初始化，但其推断不在 mainProcess 中

    // 可重用位图
    // 这些位图现在是线程局部的，不再需要作为 MainActivity 的成员变量
    // private Bitmap inputBitmap;
    // private Bitmap model_input_bitmap;
    // private Bitmap bicubic_output_bitmap;

    // 用于 TFLite 输出patch像素的可重用数组
    private int[] sr_patch_pixels;

    // RenderScript 对象 (这些现在是线程局部的，不再需要作为 MainActivity 的成员变量)
    // private RenderScript mRsYuvToRgb;
    // private Allocation inAllocYuvToRgb;
    // private Allocation outAllocYuvToRgb;
    // private ScriptIntrinsicYuvToRGB scriptYuvToRgb;

    // private RenderScript mRsResize;
    // private Allocation inAllocResize;
    // private Allocation outAllocResize;
    // private ScriptIntrinsicResize scriptResize;

    private ImageProcessor imageProcessorTFLiteInput;

    // 线程管理
    private volatile boolean processingRunning = true;
    private Thread yuvToRgbThread;
    private Thread prepareTfInputThread;
    private Thread upsampleThread;
    private Thread inferenceThread;
    private Thread afterProcessThread;

    // 这些现在是线程局部的，不再需要作为 MainActivity 的成员变量
    // private Canvas modelInputCanvas;
    // private Paint modelInputPaint;

    private void initModel() {
        try {
            this.srTFLite = new InferenceTFLite();
            this.biTFLite = new BiTFLite(); // Original initialization
            if (deligater.equals("qnn")){
                this.srTFLite.addQNNDelegate(this);
//                this.biTFLite.addQNNDelegate(this); // 原始
            }
            else if (deligater.equals("gpu")) {
                this.srTFLite.addGPUDelegate();
//                this.biTFLite.addGPUDelegate(); // 原始
                Log.i(mytag, "Using GPU Delegate for TFLite (PICO configuration)");
            } else {
                this.srTFLite.addNNApiDelegate();
//                this.biTFLite.addNNApiDelegate(); // 原始
                Log.i(mytag, "Using NNAPI Delegate for TFLite");
            }
            this.srTFLite.initialModel(this);
//            this.biTFLite.initialModel(this); // 原始
        } catch (Exception e) {
            Log.e("Error Exception", "MainActivity initial model error: " + e.getMessage(), e);
            Toast.makeText(this, "Model Initialization Failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        surfaceView = findViewById(R.id.surfaceView);
        // surfaceHolder = surfaceView.getHolder(); // 原始
        imageView = findViewById(R.id.imageView);
        // mGLSurfaceView = findViewById(R.id.glSurfaceView); // 假设存在一个 ID
        handler = new Handler(Looper.getMainLooper());

        frameSizeTextView = findViewById(R.id.frame_size);
        fpsTextView = findViewById(R.id.inference_time);

        // 初始化可重用位图 (这些现在是线程局部的，不再需要在此处初始化)
        /*
        inputBitmap = Bitmap.createBitmap(VIDEO_INPUT_W, VIDEO_INPUT_H, Bitmap.Config.ARGB_8888);
        model_input_bitmap = Bitmap.createBitmap(TF_INPUT_W, TF_INPUT_H, Bitmap.Config.ARGB_8888);
        bicubic_output_bitmap = Bitmap.createBitmap(video_output_shape.getWidth(), video_output_shape.getHeight(), Bitmap.Config.ARGB_8888);

        // 用于 model_input_bitmap 的 Canvas (这些现在是线程局部的，不再需要在此处初始化)
        modelInputCanvas = new Canvas(model_input_bitmap);
        modelInputPaint = new Paint(Paint.FILTER_BITMAP_FLAG); // 如果缩小，用于质量，尽管此处裁剪是 1:1
        */

        // 初始化 SR patch的可重用像素数组
        // 这将根据实际模型输出进行大小调整，但目前使用配置的 TF_OUTPUT 尺寸
        sr_patch_pixels = new int[TF_OUTPUT_W * TF_OUTPUT_H];


        // 初始化 RenderScript 用于 YUV 到 RGB 转换 (这些现在是线程局部的，不再需要在此处初始化)
        /*
        mRsYuvToRgb = RenderScript.create(getApplicationContext());
        Type.Builder yuvTypeBuilder = new Type.Builder(mRsYuvToRgb, Element.U8(mRsYuvToRgb))
                .setX(VIDEO_INPUT_W).setY(VIDEO_INPUT_H).setYuvFormat(ImageFormat.YV12); // 假设 JNI 提供 YV12 格式
        inAllocYuvToRgb = Allocation.createTyped(mRsYuvToRgb, yuvTypeBuilder.create(), Allocation.USAGE_SCRIPT);

        Type.Builder rgbaTypeBuilder = new Type.Builder(mRsYuvToRgb, Element.RGBA_8888(mRsYuvToRgb))
                .setX(VIDEO_INPUT_W).setY(VIDEO_INPUT_H);
        outAllocYuvToRgb = Allocation.createTyped(mRsYuvToRgb, rgbaTypeBuilder.create(), Allocation.USAGE_SCRIPT);
        scriptYuvToRgb = ScriptIntrinsicYuvToRGB.create(mRsYuvToRgb, Element.RGBA_8888(mRsYuvToRgb));

        // 初始化 RenderScript 用于双三次缩放 (这些现在是线程局部的，不再需要在此处初始化)
        mRsResize = RenderScript.create(getApplicationContext());
        inAllocResize = Allocation.createFromBitmap(mRsResize, inputBitmap); // 缩放的输入是完整的 RGB 帧
        outAllocResize = Allocation.createFromBitmap(mRsResize, bicubic_output_bitmap); // 输出是大的双三次上采样位图
        scriptResize = ScriptIntrinsicResize.create(mRsResize);
        */

        // TFLite 输入的 ImageProcessor（此处无 ResizeOp，因为 model_input_bitmap 已是正确大小）
        imageProcessorTFLiteInput = new ImageProcessor.Builder()
                .add(new NormalizeOp(0f, 255f))
                // 如果 srTFLite 模型是 UINT8 且 InferenceTFLite 中的 IS_INT8 标志为 true，则在此处添加 QuantizeOp 和 CastOp
                // 这需要与 InferenceTFLite 的 IS_INT8 逻辑和模型要求对齐。
                // 目前，假设是 FLOAT32 模型或 InferenceTFLite 中 IS_INT8=false。
                .build();

        initModel();

        // 启动解码器线程
        new Thread(() -> {
            Log.i(mytag, "Decoder thread started.");
            mainDecoder(getString(R.string.video_url)); // Ensure R.string.video_url is defined
            Log.i(mytag, "Decoder thread finished.");
        }).start();

        mainProcess();
    }

    public void mainProcess() {
        processingRunning = true; // 在启动线程前设置标志

        yuvToRgbThread = new Thread(this::yuvToRgbLoop, "YuvToRgbThread");
        yuvToRgbThread.start();

        prepareTfInputThread = new Thread(this::prepareTfInputLoop, "PrepareTfInputThread");
        prepareTfInputThread.start();

        upsampleThread = new Thread(this::upsampleLoop, "UpsampleThread");
        upsampleThread.start();

        inferenceThread = new Thread(this::inferenceLoop, "InferenceThread");
        inferenceThread.start();

        afterProcessThread = new Thread(this::afterProcessLoop, "AfterProcessThread");
        afterProcessThread.start();
    }

    private void yuvToRgbLoop() {
        Log.i(mytag, "YUV to RGB Loop Started");
        // 每个线程创建独立的 RenderScript 实例
        RenderScript threadRsYuvToRgb = RenderScript.create(getApplicationContext());
        Type.Builder yuvTypeBuilder = new Type.Builder(threadRsYuvToRgb, Element.U8(threadRsYuvToRgb))
                .setX(VIDEO_INPUT_W).setY(VIDEO_INPUT_H).setYuvFormat(ImageFormat.YV12);
        Allocation inAlloc = Allocation.createTyped(threadRsYuvToRgb, yuvTypeBuilder.create(), Allocation.USAGE_SCRIPT);
        Type.Builder rgbaTypeBuilder = new Type.Builder(threadRsYuvToRgb, Element.RGBA_8888(threadRsYuvToRgb))
                .setX(VIDEO_INPUT_W).setY(VIDEO_INPUT_H);
        Allocation outAlloc = Allocation.createTyped(threadRsYuvToRgb, rgbaTypeBuilder.create(), Allocation.USAGE_SCRIPT);
        ScriptIntrinsicYuvToRGB scriptYuvToRgb = ScriptIntrinsicYuvToRGB.create(threadRsYuvToRgb, Element.RGBA_8888(threadRsYuvToRgb));

        // 线程局部位图对象
        Bitmap threadLocalInputBitmap = Bitmap.createBitmap(VIDEO_INPUT_W, VIDEO_INPUT_H, Bitmap.Config.ARGB_8888);

        while (processingRunning) {
            try {
                TaggedData<byte[]> taggedYuvData = yuvBytesQueue.take(); // 如果队列为空则阻塞
                byte[] yuvData = taggedYuvData.getData();
                long frameNum = taggedYuvData.getFrameNumber();
                Log.i(mytag, "YUV to RGB Loop: Processing frame " + frameNum);

                long startTime = System.currentTimeMillis();
                inAlloc.copyFrom(yuvData);
                scriptYuvToRgb.setInput(inAlloc);
                scriptYuvToRgb.forEach(outAlloc);
                outAlloc.copyTo(threadLocalInputBitmap); // 转换为 RGB

                long endTime = System.currentTimeMillis();
                Log.i(time_tag, "YUV to RGB: " + (endTime - startTime) + " ms");

                // 将转换后的 RGB 位图的副本放入两个队列
                // 必须创建副本，因为两个消费者线程会独立使用它们
                Bitmap rgbFrameCopy = threadLocalInputBitmap.copy(threadLocalInputBitmap.getConfig(), false);
                rgbFrameQueueForUpsample.put(new TaggedData<>(rgbFrameCopy.copy(threadLocalInputBitmap.getConfig(), false), frameNum));
                rgbFrameQueueForTfInput.put(new TaggedData<>(rgbFrameCopy.copy(threadLocalInputBitmap.getConfig(), false), frameNum));


            } catch (InterruptedException e) {
                Log.w(mytag, "YUV to RGB Loop interrupted. Exiting.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(mytag, "Error in YUV to RGB Loop: " + e.getMessage(), e);
            }
        }
        // 释放 RenderScript 资源
        if (inAlloc != null) inAlloc.destroy();
        if (outAlloc != null) outAlloc.destroy();
        if (scriptYuvToRgb != null) scriptYuvToRgb.destroy();
        if (threadRsYuvToRgb != null) threadRsYuvToRgb.destroy();
        if (threadLocalInputBitmap != null && !threadLocalInputBitmap.isRecycled()) threadLocalInputBitmap.recycle();
        Log.i(mytag, "YUV to RGB Loop Finished");
    }

    private void prepareTfInputLoop() {
        Log.i(mytag, "Prepare TFLite Input Loop Started");
        Rect srcRect = new Rect();
        Rect dstRect = new Rect(0, 0, TF_INPUT_W, TF_INPUT_H);

        // 线程局部位图对象
        Bitmap threadLocalModelInputBitmap = Bitmap.createBitmap(TF_INPUT_W, TF_INPUT_H, Bitmap.Config.ARGB_8888);
        Canvas threadLocalModelInputCanvas = new Canvas(threadLocalModelInputBitmap);
        Paint threadLocalModelInputPaint = new Paint(Paint.FILTER_BITMAP_FLAG);

        while (processingRunning) {
            try {
                long loopStart = System.currentTimeMillis();

                long takeStart = System.currentTimeMillis();
                TaggedData<Bitmap> taggedRgbFrameForTfInput = rgbFrameQueueForTfInput.take(); // 从 YUV 转 RGB 队列获取，用于 TFLite 输入
                Bitmap rgbFrame = taggedRgbFrameForTfInput.getData();
                long frameNum = taggedRgbFrameForTfInput.getFrameNumber();
                long takeEnd = System.currentTimeMillis();
                Log.i(mytag, "Prepare TFLite Input Loop: Processing frame " + frameNum);

                long processingStart = System.currentTimeMillis();
                int cropX = tile_index[0] * TF_INPUT_W;
                int cropY = tile_index[1] * TF_INPUT_H;

                if (cropX + TF_INPUT_W > VIDEO_INPUT_W || cropY + TF_INPUT_H > VIDEO_INPUT_H) {
                    Log.e(mytag, "Crop dimensions exceed inputBitmap bounds. Skipping frame for TFLite.");
                    // 可能会放置一个占位符或跳过将数据放入 modelInputQueue
                    // 目前，我们将继续，但这可能会在 createBitmap/drawBitmap 失败时崩溃
                    continue; // 跳过当前帧
                }

                srcRect.set(cropX, cropY, cropX + TF_INPUT_W, cropY + TF_INPUT_H);
                threadLocalModelInputCanvas.drawBitmap(rgbFrame, srcRect, dstRect, threadLocalModelInputPaint); // 裁剪到线程局部位图

                Bitmap threadLocalModelInputBitmap1 = threadLocalModelInputBitmap.copy(threadLocalModelInputBitmap.getConfig(), false );

                TensorImage modelInputTensor = new TensorImage(DataType.FLOAT32);
                modelInputTensor.load(threadLocalModelInputBitmap1);
                modelInputTensor = imageProcessorTFLiteInput.process(modelInputTensor);
                long processingEnd = System.currentTimeMillis();

                // 记录详细时间信息
                long loopEnd = System.currentTimeMillis();
                Log.i(time_tag, "TFLite Input - Total: " + (loopEnd - loopStart) + "ms | Take: " + (takeEnd - takeStart) + "ms | Processing: " + (processingEnd - processingStart) + "ms");

                modelInputQueue.put(new TaggedData<>(modelInputTensor, frameNum)); // 放入 TFLite 输入队列

            } catch (InterruptedException e) {
                Log.w(mytag, "Prepare TFLite Input Loop interrupted. Exiting.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(mytag, "Error in Prepare TFLite Input Loop: " + e.getMessage(), e);
            }
        }
        if (threadLocalModelInputBitmap != null && !threadLocalModelInputBitmap.isRecycled()) threadLocalModelInputBitmap.recycle();
        Log.i(mytag, "Prepare TFLite Input Loop Finished");
    }

    private void upsampleLoop() {
        Log.i(mytag, "Upsample Loop Started");
        // 每个线程创建独立的 RenderScript 实例
        RenderScript threadRsResize = RenderScript.create(getApplicationContext());
        // 线程局部位图对象
        Bitmap threadLocalBicubicOutputBitmap = Bitmap.createBitmap(video_output_shape.getWidth(), video_output_shape.getHeight(), Bitmap.Config.ARGB_8888);

        // RenderScript Allocation
        // 输出 Allocation
        Allocation outAlloc = Allocation.createFromBitmap(threadRsResize, threadLocalBicubicOutputBitmap);
        // 输入 Allocation - 预先创建，因为输入尺寸 (rgbFrame) 已知
        // rgbFrame 的尺寸是 VIDEO_INPUT_W, VIDEO_INPUT_H
        Type.Builder rgbaInputTypeBuilder = new Type.Builder(threadRsResize, Element.RGBA_8888(threadRsResize))
                .setX(VIDEO_INPUT_W)
                .setY(VIDEO_INPUT_H);
        Allocation inAlloc = Allocation.createTyped(threadRsResize, rgbaInputTypeBuilder.create(), Allocation.USAGE_SCRIPT);

        ScriptIntrinsicResize scriptResize = ScriptIntrinsicResize.create(threadRsResize);
        scriptResize.setInput(inAlloc); // 设置一次输入 Allocation

        while (processingRunning) {
            try {
                long taketime = System.currentTimeMillis();
                TaggedData<Bitmap> taggedRgbFrameForUpsample = rgbFrameQueueForUpsample.take(); // 从 YUV 转 RGB 队列获取，用于上采样
                Bitmap rgbFrame = taggedRgbFrameForUpsample.getData();
                long frameNum = taggedRgbFrameForUpsample.getFrameNumber();

                // 防御性检查，确保 rgbFrame 与 inAlloc 尺寸匹配
                if (rgbFrame == null || rgbFrame.getWidth() != VIDEO_INPUT_W || rgbFrame.getHeight() != VIDEO_INPUT_H) {
                    Log.e(mytag, "Upsample Loop: Received frame " + frameNum + " with unexpected dimensions or null. Expected " +
                            VIDEO_INPUT_W + "x" + VIDEO_INPUT_H + ", got " +
                            (rgbFrame != null ? rgbFrame.getWidth() + "x" + rgbFrame.getHeight() : "null") + ". Skipping.");
                    if (rgbFrame != null && !rgbFrame.isRecycled()) { // 如果不需要，则回收传入的位图
                        // rgbFrame.recycle(); // 注意：这个rgbFrame是从队列中获取的，其他线程可能也在使用它的副本。
                        // 这里的回收需要小心，确保这个实例不会再被其他地方使用。
                        // 根据 yuvToRgbLoop 的逻辑，放入队列的是副本，所以理论上这里回收是安全的，
                        // 但要非常确定这个Bitmap实例不会被其他地方引用。
                        // 为安全起见，暂时不回收，依赖队列和后续处理阶段的Bitmap管理。
                    }
                    continue;
                }

                Log.i(mytag, "Upsample Loop: Processing frame " + frameNum);

                long startTime = System.currentTimeMillis();
                // 使用 copyFrom 更新预先创建的 inAlloc 的内容
                inAlloc.copyFrom(rgbFrame);
                // scriptResize.setInput(inAlloc); // 已经在循环外设置

                scriptResize.forEach_bicubic(outAlloc);
                outAlloc.copyTo(threadLocalBicubicOutputBitmap); // 上采样结果

                // 不再需要在循环内销毁临时的输入 Allocation
                // inAlloc.destroy(); // 已移到循环外

                long endTime = System.currentTimeMillis();
                Log.i(time_tag, "Bicubic Upscale: " + (endTime - startTime) + " ms" + " (take time: " + (startTime - taketime) + " ms)");

                // 将上采样结果的副本放入队列，因为 afterProcessLoop 会修改它
                Bitmap biSrOutputCopy = threadLocalBicubicOutputBitmap.copy(threadLocalBicubicOutputBitmap.getConfig(), true);
                biSROutputQueue.put(new TaggedData<>(biSrOutputCopy, frameNum)); // 放入上采样结果队列

            } catch (InterruptedException e) {
                Log.w(mytag, "Upsample Loop interrupted. Exiting.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(mytag, "Error in Upsample Loop: " + e.getMessage(), e);
            }
        }
        // 释放 RenderScript 资源
        if (inAlloc != null) inAlloc.destroy(); // 在此处销毁预先创建的 inAlloc
        if (outAlloc != null) outAlloc.destroy();
        if (scriptResize != null) scriptResize.destroy();
        if (threadRsResize != null) threadRsResize.destroy();
        if (threadLocalBicubicOutputBitmap != null && !threadLocalBicubicOutputBitmap.isRecycled()) {
            threadLocalBicubicOutputBitmap.recycle();
        }
        Log.i(mytag, "Upsample Loop Finished");
    }
    private void inferenceLoop() {
        Log.i(mytag, "Inference Loop Started");
        while (processingRunning) {
            try {
                long loopStart = System.currentTimeMillis();

                long takeStart = System.currentTimeMillis();
                TaggedData<TensorImage> taggedModelInput = modelInputQueue.take(); // 如果队列为空则阻塞
                TensorImage modelInput = taggedModelInput.getData();
                long frameNum = taggedModelInput.getFrameNumber();
                long takeEnd = System.currentTimeMillis();
                Log.i(mytag, "Inference Loop: Processing frame " + frameNum);

                long inferenceStart = System.currentTimeMillis();
                // 将 TF_OUTPUT_W 和 TF_OUTPUT_H 传递给 superResolution
                // InferenceTFLite 需要 [宽度, 高度] 作为其 tf_output_shape 参数
                TensorBuffer modelOutput = srTFLite.superResolution(modelInput, TF_OUTPUT_SHAPE);
                long inferenceEnd = System.currentTimeMillis();

                // 记录详细时间信息
                long loopEnd = System.currentTimeMillis();
                Log.i(time_tag, "Inference Loop - Total: " + (loopEnd - loopStart) + "ms | Take: " + (takeEnd - takeStart) + "ms | Inference: " + (inferenceEnd - inferenceStart) + "ms");

                modelOutputQueue.put(new TaggedData<>(modelOutput, frameNum)); // 如果队列已满则阻塞

            } catch (InterruptedException e) {
                Log.w(mytag, "Inference Loop interrupted. Exiting.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(mytag, "Error in Inference Loop: " + e.getMessage(), e);
            }
        }
        Log.i(mytag, "Inference Loop Finished");
    }

    private void afterProcessLoop() {
        Log.i(mytag, "AfterProcessing Loop Started");
        Matrix displayMatrix = new Matrix();
        if (!isPICO) { // 手机显示的原始逻辑
            // matrix.postRotate(0); // Original was 0, if 90 is needed:
            // displayMatrix.postRotate(90);
        }

        while (processingRunning) {
            try {
                long loopStart = System.currentTimeMillis();

                long takeStart = System.currentTimeMillis();
                TaggedData<Bitmap> taggedBicubicBaseBitmap = biSROutputQueue.take(); // 如果队列为空则阻塞
                Bitmap bicubicBaseBitmap = taggedBicubicBaseBitmap.getData();
                long frameNumBiSR = taggedBicubicBaseBitmap.getFrameNumber();
                long takeMiddle = System.currentTimeMillis();
                TaggedData<TensorBuffer> taggedHwcOutputTensorBuffer = modelOutputQueue.take(); // 如果队列为空则阻塞
                TensorBuffer hwcOutputTensorBuffer = taggedHwcOutputTensorBuffer.getData();
                long frameNumModelOutput = taggedHwcOutputTensorBuffer.getFrameNumber();
                long takeEnd = System.currentTimeMillis();
                if (frameNumBiSR != frameNumModelOutput) {
                    Log.w(mytag, "AfterProcess Loop: Mismatched frame numbers! BiSR: " + frameNumBiSR + ", ModelOutput: " + frameNumModelOutput);
                }
                Log.i(mytag, "AfterProcess Loop: Processing frame " + frameNumBiSR);

                // 1. 将 TFLite TensorBuffer 转换为 ARGB int[] patch
                long tensorProcessingStart = System.currentTimeMillis();
                float[] hwcOutputData = hwcOutputTensorBuffer.getFloatArray();

                // 从张量缓冲区获取实际尺寸（如果模型一致，应与 TF_OUTPUT_W、TF_OUTPUT_H 匹配）
                // 对于 TFLite 图像模型，形状通常为 [Batch, Height, Width, Channels] (NHWC)
                // 或者，如果 InferenceTFLite 以这种方式构建，则为 [Batch, Width, Height, Channels]。
                // 基于 InferenceTFLite 的先前逻辑：形状为 [1, H_param, W_param, 3]
                // 其中 H_param 是 tf_output_shape[1]，W_param 是来自 MainActivity 的 tf_output_shape[0]。
                // 因此，如果 MainActivity 发送 [TF_OUTPUT_W, TF_OUTPUT_H]，那么在 InferenceTFLite 内部：
                // H_param = TF_OUTPUT_H, W_param = TF_OUTPUT_W。
                // 结果 TensorBuffer 形状：[1, TF_OUTPUT_H, TF_OUTPUT_W, 3]
                int batch = hwcOutputTensorBuffer.getShape()[0]; // Should be 1
                int patchH = hwcOutputTensorBuffer.getShape()[1]; // Expected: TF_OUTPUT_H
                int patchW = hwcOutputTensorBuffer.getShape()[2]; // Expected: TF_OUTPUT_W
                int channels = hwcOutputTensorBuffer.getShape()[3]; // Expected: 3

                if (patchH != TF_OUTPUT_H || patchW != TF_OUTPUT_W) {
                    Log.w(mytag, "警告: SR 输出patch尺寸 (" + patchW + "x" + patchH +
                            ") 与配置的 TF_OUTPUT (" + TF_OUTPUT_W + "x" + TF_OUTPUT_H + ") 不同。正在调整 sr_patch_pixels 数组大小。");
                    sr_patch_pixels = new int[patchW * patchH];
                } else if (sr_patch_pixels.length != patchW * patchH) {
                    sr_patch_pixels = new int[patchW * patchH]; // Ensure correct size
                }


                int yp = 0;
                for (int h = 0; h < patchH; h++) {
                    for (int w = 0; w < patchW; w++) {
                        int r_idx = (h * patchW + w) * channels + 0;
                        int g_idx = (h * patchW + w) * channels + 1;
                        int b_idx = (h * patchW + w) * channels + 2;

                        int r = (int) (hwcOutputData[r_idx] * 255f);
                        int g = (int) (hwcOutputData[g_idx] * 255f);
                        int b = (int) (hwcOutputData[b_idx] * 255f);

//                        r = Math.max(0, Math.min(255, r));
//                        g = Math.max(0, Math.min(255, g));
//                        b = Math.max(0, Math.min(255, b));
                        sr_patch_pixels[yp++] = 0xFF000000 | (r << 16) | (g << 8) | b;
                    }
                }
                long tensorProcessingEnd = System.currentTimeMillis();

                // 2. 将 SR patch放置在双三次上采样的基础图像上
                long compositionStart = System.currentTimeMillis();
                int offsetX = tile_index[0] * patchW; // Offset by output patch width
                int offsetY = tile_index[1] * patchH; // Offset by output patch height

                if (offsetX + patchW > bicubicBaseBitmap.getWidth() || offsetY + patchH > bicubicBaseBitmap.getHeight()) {
                    Log.e(mytag, "SR patch放置超出 bicubicBaseBitmap 边界。跳过 setPixels。");
                } else {
                    // bicubicBaseBitmap 是从 biSROutputQueue 中获取的副本。
                    // 因此，在此处修改它是安全的。
                    bicubicBaseBitmap.setPixels(sr_patch_pixels, 0, patchW, offsetX, offsetY, patchW, patchH);
                }
                long compositionEnd = System.currentTimeMillis();

                // 3. 应用矩阵变换（如果有）并显示
                long displayStart = System.currentTimeMillis();
                Bitmap finalBitmapToDisplay = bicubicBaseBitmap; // Default
                if (!displayMatrix.isIdentity()) { // 仅在需要变换时创建新位图
                    finalBitmapToDisplay = Bitmap.createBitmap(bicubicBaseBitmap, 0, 0,
                            bicubicBaseBitmap.getWidth(), bicubicBaseBitmap.getHeight(), displayMatrix, true);
                }

                final Bitmap displayBitmap = finalBitmapToDisplay; // 对 lambda 来说是有效的 final
                handler.post(() -> imageView.setImageBitmap(displayBitmap));
                long displayEnd = System.currentTimeMillis();

                // 记录详细时间信息
                long loopEnd = System.currentTimeMillis();
                long displayCost = displayEnd - displayStart;
                Log.i(time_tag, "AfterProcess - Total: " + (loopEnd - loopStart) + "ms | Take: " + (takeMiddle - takeStart) + "ms, " + (takeEnd - takeMiddle)  + "ms | Tensor: " + (tensorProcessingEnd - tensorProcessingStart) + "ms | Composite: " + (compositionEnd - compositionStart) + "ms | Display: " + displayCost + "ms");

                updateTextView(Long.toString(loopEnd - loopStart) + "ms (display: " + displayCost + "ms)");
                Log.i(time_tag, "AfterProcess - Queue sizes: ModelOut=" + modelOutputQueue.size() + " BiSR=" + biSROutputQueue.size() +
                        " YUV=" + yuvBytesQueue.size() + " RGBForTfInput=" + rgbFrameQueueForTfInput.size() +
                        " RGBForUpsample=" + rgbFrameQueueForUpsample.size());


            } catch (InterruptedException e) {
                Log.w(mytag, "AfterProcessing Loop interrupted. Exiting.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(mytag, "Error in AfterProcessing Loop: " + e.getMessage(), e);
            }
        }
        Log.i(mytag, "AfterProcessing Loop Finished");
    }

    public void updateTextView(String time) { // 为清晰起见更改参数名称
        runOnUiThread(() -> {
            if (fpsTextView != null) { // 检查是否为空，以防视图未准备好/已消失
                fpsTextView.setText(time);
            }
            // frameSizeTextView.setText(...); // 原始代码未在此处更新此项
        });
    }

    // 从 JNI 调用
    public static void putData(byte[] data) {
        try {
            long currentFrameNumber = frameCounter.getAndIncrement(); // 获取并递增帧号
            // Log.i(mytag, "JNI putData: received " + data.length + " bytes for frame " + currentFrameNumber + ". Queue capacity: " + yuvBytesQueue.remainingCapacity());
            yuvBytesQueue.put(new TaggedData<>(data, currentFrameNumber));
        } catch (InterruptedException e) {
            Log.e(mytag, "JNI putData: Interrupted while putting data into yuvBytesQueue for frame.");
            Thread.currentThread().interrupt();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.i(mytag, "onDestroy called. Shutting down processing threads and resources.");
        processingRunning = false; // Signal loops to stop

        // 中断线程以使其脱离阻塞队列操作
        if (yuvToRgbThread != null) {
            yuvToRgbThread.interrupt();
        }
        if (prepareTfInputThread != null) {
            prepareTfInputThread.interrupt();
        }
        if (upsampleThread != null) {
            upsampleThread.interrupt();
        }
        if (inferenceThread != null) {
            inferenceThread.interrupt();
        }
        if (afterProcessThread != null) {
            afterProcessThread.interrupt();
        }

        // 等待线程完成（可选，带超时）
        try {
            if (yuvToRgbThread != null) yuvToRgbThread.join(1000);
            if (prepareTfInputThread != null) prepareTfInputThread.join(1000);
            if (upsampleThread != null) upsampleThread.join(1000);
            if (inferenceThread != null) inferenceThread.join(1000);
            if (afterProcessThread != null) afterProcessThread.join(1000);
        } catch (InterruptedException e) {
            Log.w(mytag, "Interrupted while joining threads.");
            Thread.currentThread().interrupt();
        }

        // 清空队列（可选，如果线程未完全排空，有助于垃圾回收）
        yuvBytesQueue.clear();
        rgbFrameQueueForTfInput.clear();
        rgbFrameQueueForUpsample.clear();
        modelInputQueue.clear();
        modelOutputQueue.clear();
        biSROutputQueue.clear();
        frameCounter.set(0); // 重置帧号计数器

        // 释放 TFLite 模型
        if (srTFLite != null) {
            srTFLite.close(); // 假设 InferenceTFLite 有一个 close() 方法
        }
        if (biTFLite != null) {
            // biTFLite.close(); // 假设 BiTFLite 也有一个 close() 方法
        }


        Log.i(mytag, "onDestroy finished.");
    }

    // 本地方法声明
    public native void mainDecoder(String url);
}

