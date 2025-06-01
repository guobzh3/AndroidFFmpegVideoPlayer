




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
    private static BlockingQueue<byte[]> yuvBytesQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TensorImage> modelInputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TensorBuffer> modelOutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    // private static BlockingQueue<int[]> viewOutQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY); // 在提供的逻辑中未使用
    private static BlockingQueue<Bitmap> biSROutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);

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
    private Bitmap inputBitmap;        // 用于 YUV->RGB 转换输出（完整帧）
    private Bitmap model_input_bitmap; // 用于 TFLite 输入（裁剪patch），可重用
    private Bitmap bicubic_output_bitmap; // 用于 RenderScript 双三次上采样输出（完整帧），可重用

    // 用于 TFLite 输出patch像素的可重用数组
    private int[] sr_patch_pixels;

    // RenderScript 对象
    private RenderScript mRsYuvToRgb;
    private Allocation inAllocYuvToRgb;
    private Allocation outAllocYuvToRgb;
    private ScriptIntrinsicYuvToRGB scriptYuvToRgb;

    private RenderScript mRsResize;
    private Allocation inAllocResize;
    private Allocation outAllocResize;
    private ScriptIntrinsicResize scriptResize;

    private ImageProcessor imageProcessorTFLiteInput;

    // 线程管理
    private volatile boolean processingRunning = true;
    private Thread preProcessThread;
    private Thread inferenceThread;
    private Thread afterProcessThread;

    private Canvas modelInputCanvas; // 用于在 model_input_bitmap 上绘制裁剪区域
    private Paint modelInputPaint;   // 可选，用于绘制选项

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

        // 初始化可重用位图
        inputBitmap = Bitmap.createBitmap(VIDEO_INPUT_W, VIDEO_INPUT_H, Bitmap.Config.ARGB_8888);
        model_input_bitmap = Bitmap.createBitmap(TF_INPUT_W, TF_INPUT_H, Bitmap.Config.ARGB_8888);
        bicubic_output_bitmap = Bitmap.createBitmap(video_output_shape.getWidth(), video_output_shape.getHeight(), Bitmap.Config.ARGB_8888);

        // 用于 model_input_bitmap 的 Canvas
        modelInputCanvas = new Canvas(model_input_bitmap);
        modelInputPaint = new Paint(Paint.FILTER_BITMAP_FLAG); // 如果缩小，用于质量，尽管此处裁剪是 1:1

        // 初始化 SR patch的可重用像素数组
        // 这将根据实际模型输出进行大小调整，但目前使用配置的 TF_OUTPUT 尺寸
        sr_patch_pixels = new int[TF_OUTPUT_W * TF_OUTPUT_H];


        // 初始化 RenderScript 用于 YUV 到 RGB 转换
        mRsYuvToRgb = RenderScript.create(getApplicationContext());
        Type.Builder yuvTypeBuilder = new Type.Builder(mRsYuvToRgb, Element.U8(mRsYuvToRgb))
                .setX(VIDEO_INPUT_W).setY(VIDEO_INPUT_H).setYuvFormat(ImageFormat.YV12); // 假设 JNI 提供 YV12 格式
        inAllocYuvToRgb = Allocation.createTyped(mRsYuvToRgb, yuvTypeBuilder.create(), Allocation.USAGE_SCRIPT);

        Type.Builder rgbaTypeBuilder = new Type.Builder(mRsYuvToRgb, Element.RGBA_8888(mRsYuvToRgb))
                .setX(VIDEO_INPUT_W).setY(VIDEO_INPUT_H);
        outAllocYuvToRgb = Allocation.createTyped(mRsYuvToRgb, rgbaTypeBuilder.create(), Allocation.USAGE_SCRIPT);
        scriptYuvToRgb = ScriptIntrinsicYuvToRGB.create(mRsYuvToRgb, Element.RGBA_8888(mRsYuvToRgb));

        // 初始化 RenderScript 用于双三次缩放
        mRsResize = RenderScript.create(getApplicationContext());
        inAllocResize = Allocation.createFromBitmap(mRsResize, inputBitmap); // 缩放的输入是完整的 RGB 帧
        outAllocResize = Allocation.createFromBitmap(mRsResize, bicubic_output_bitmap); // 输出是大的双三次上采样位图
        scriptResize = ScriptIntrinsicResize.create(mRsResize);

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

        preProcessThread = new Thread(this::preProcessLoop, "PreProcessThread");
        preProcessThread.start();

        inferenceThread = new Thread(this::inferenceLoop, "InferenceThread");
        inferenceThread.start();

        afterProcessThread = new Thread(this::afterProcessLoop, "AfterProcessThread");
        afterProcessThread.start();
    }

    private void preProcessLoop() {
        Log.i(mytag, "PreProcessing Loop Started");
        Rect srcRect = new Rect();
        Rect dstRect = new Rect(0, 0, TF_INPUT_W, TF_INPUT_H);

        while (processingRunning) {
            try {
                long startTimeFullLoop, endTimeFullLoop, costTimeFullLoop;
                startTimeFullLoop = System.currentTimeMillis();

                byte[] yuvData = yuvBytesQueue.take(); // 如果队列为空则阻塞

                // 1. 使用 RenderScript 将 YUV 转换为 RGB
                long startTime = System.currentTimeMillis();
                inAllocYuvToRgb.copyFrom(yuvData);
                scriptYuvToRgb.setInput(inAllocYuvToRgb);
                scriptYuvToRgb.forEach(outAllocYuvToRgb);
                outAllocYuvToRgb.copyTo(inputBitmap); // inputBitmap 现在包含完整的 RGB 帧
                long endTime = System.currentTimeMillis();
                Log.i(time_tag, "preProcess - YUV to RGB: " + (endTime - startTime) + " ms");

                // 2. 准备 TFLite 模型输入（裁剪和处理）
                startTime = System.currentTimeMillis();
                int cropX = tile_index[0] * TF_INPUT_W;
                int cropY = tile_index[1] * TF_INPUT_H;

                // 确保裁剪区域在 inputBitmap 的边界内
                if (cropX + TF_INPUT_W > VIDEO_INPUT_W || cropY + TF_INPUT_H > VIDEO_INPUT_H) {
                    Log.e(mytag, "Crop dimensions exceed inputBitmap bounds. Skipping frame for TFLite.");
                    // 可能会放置一个占位符或跳过将数据放入 modelInputQueue
                    // 目前，我们将继续，但这可能会在 createBitmap/drawBitmap 失败时崩溃
                }

                srcRect.set(cropX, cropY, cropX + TF_INPUT_W, cropY + TF_INPUT_H);
                modelInputCanvas.drawBitmap(inputBitmap, srcRect, dstRect, modelInputPaint); // model_input_bitmap 现在包含裁剪后的patch

                TensorImage modelInputTensor = new TensorImage(DataType.FLOAT32); // 假设是 FLOAT32，如果模型是量化的则调整
                modelInputTensor.load(model_input_bitmap);
                modelInputTensor = imageProcessorTFLiteInput.process(modelInputTensor);
                endTime = System.currentTimeMillis();
                Log.i(time_tag, "preProcess - Create TFLite Input: " + (endTime - startTime) + " ms");

                modelInputQueue.put(modelInputTensor); // 如果队列已满则阻塞

                // 3. 使用 RenderScript 对图像的其余部分执行双三次上采样
                startTime = System.currentTimeMillis();
                // Update inAllocResize if inputBitmap content changed and it's not auto-synced.
                // createFromBitmap 将 Allocation 链接到 Bitmap。如果 Bitmap 像素更改，
                // 如果链接不是实时的，inAllocResize 需要被通知或重新创建。
                // 为了安全起见，或者如果出现问题，可以重新复制：
                inAllocResize.copyFrom(inputBitmap); // Ensure fresh data from inputBitmap
                scriptResize.setInput(inAllocResize);
                scriptResize.forEach_bicubic(outAllocResize);
                outAllocResize.copyTo(bicubic_output_bitmap); // bicubic_output_bitmap 现在包含完整的上采样图像
                endTime = System.currentTimeMillis();
                Log.i(time_tag, "preProcess - Bicubic Upscale: " + (endTime - startTime) + " ms");

                // 将 bicubic_output_bitmap 的副本放入队列
                // 如果 afterProcessLoop 修改了它接收到的位图，这很重要
                Bitmap biSrOutputCopy = bicubic_output_bitmap.copy(bicubic_output_bitmap.getConfig(), true);
                biSROutputQueue.put(biSrOutputCopy); // 如果队列已满则阻塞

                endTimeFullLoop = System.currentTimeMillis();
                costTimeFullLoop = endTimeFullLoop - startTimeFullLoop;
                Log.i(time_tag, "preProcess - Full Loop Time: " + costTimeFullLoop + " ms. Queue sizes: YUV=" + yuvBytesQueue.size() + " ModelIn=" + modelInputQueue.size() + " BiSR=" + biSROutputQueue.size());


            } catch (InterruptedException e) {
                Log.w(mytag, "PreProcessing Loop interrupted. Exiting.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(mytag, "Error in PreProcessing Loop: " + e.getMessage(), e);
                // 根据错误严重性继续循环或中断
            }
        }
        Log.i(mytag, "PreProcessing Loop Finished");
    }

    private void inferenceLoop() {
        Log.i(mytag, "Inference Loop Started");
        while (processingRunning) {
            try {
                TensorImage modelInput = modelInputQueue.take(); // 如果队列为空则阻塞

                long startTime = System.currentTimeMillis();
                // 将 TF_OUTPUT_W 和 TF_OUTPUT_H 传递给 superResolution
                // InferenceTFLite 需要 [宽度, 高度] 作为其 tf_output_shape 参数
                TensorBuffer modelOutput = srTFLite.superResolution(modelInput, TF_OUTPUT_SHAPE);
                long endTime = System.currentTimeMillis();
                Log.i(time_tag, "Inference - SR Time: " + (endTime - startTime) + " ms");
                // Log.i(mytag, "Input: " + modelInput.getWidth() + "x" + modelInput.getHeight());
                // Log.i(mytag, "Output shape: " + Arrays.toString(modelOutput.getShape()));

                modelOutputQueue.put(modelOutput); // 如果队列已满则阻塞

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
                long startTimeFullLoop = System.currentTimeMillis();
                TensorBuffer hwcOutputTensorBuffer = modelOutputQueue.take(); // 如果队列为空则阻塞
                Bitmap bicubicBaseBitmap = biSROutputQueue.take(); // 如果队列为空则阻塞

                // 1. 将 TFLite TensorBuffer 转换为 ARGB int[] patch
                long startTime = System.currentTimeMillis();
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

                        r = Math.max(0, Math.min(255, r));
                        g = Math.max(0, Math.min(255, g));
                        b = Math.max(0, Math.min(255, b));
                        sr_patch_pixels[yp++] = 0xFF000000 | (r << 16) | (g << 8) | b;
                    }
                }
                long endTime = System.currentTimeMillis();
                Log.i(time_tag, "afterProcess - Tensor to ARGB: " + (endTime - startTime) + " ms");

                // 2. 将 SR patch放置在双三次上采样的基础图像上
                startTime = System.currentTimeMillis();
                int offsetX = tile_index[0] * patchW; // Offset by output patch width
                int offsetY = tile_index[1] * patchH; // Offset by output patch height

                if (offsetX + patchW > bicubicBaseBitmap.getWidth() || offsetY + patchH > bicubicBaseBitmap.getHeight()) {
                    Log.e(mytag, "SR patch放置超出 bicubicBaseBitmap 边界。跳过 setPixels。");
                } else {
                    // bicubicBaseBitmap 是从 biSROutputQueue 中获取的副本。
                    // 因此，在此处修改它是安全的。
                    bicubicBaseBitmap.setPixels(sr_patch_pixels, 0, patchW, offsetX, offsetY, patchW, patchH);
                }
                endTime = System.currentTimeMillis();
                Log.i(time_tag, "afterProcess - Composite Patch: " + (endTime - startTime) + " ms");

                // 3. 应用矩阵变换（如果有）并显示
                startTime = System.currentTimeMillis();
                Bitmap finalBitmapToDisplay = bicubicBaseBitmap; // Default
                if (!displayMatrix.isIdentity()) { // 仅在需要变换时创建新位图
                    finalBitmapToDisplay = Bitmap.createBitmap(bicubicBaseBitmap, 0, 0,
                            bicubicBaseBitmap.getWidth(), bicubicBaseBitmap.getHeight(), displayMatrix, true);
                }

                final Bitmap displayBitmap = finalBitmapToDisplay; // 对 lambda 来说是有效的 final
                handler.post(() -> imageView.setImageBitmap(displayBitmap));
                endTime = System.currentTimeMillis();
                long displayCost = endTime - startTime;
                Log.i(time_tag, "afterProcess - Display: " + displayCost + " ms");

                long endTimeFullLoop = System.currentTimeMillis();
                long totalLoopCost = endTimeFullLoop - startTimeFullLoop;
                updateTextView(Long.toString(totalLoopCost) + "ms (display: " + displayCost + "ms)");
                Log.i(time_tag, "afterProcess - Full Loop Time: " + totalLoopCost + " ms. Queue sizes: ModelOut=" + modelOutputQueue.size() + " BiSR=" + biSROutputQueue.size());


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
            // Log.i(mytag, "JNI putData: received " + data.length + " bytes. Queue capacity: " + yuvBytesQueue.remainingCapacity());
            yuvBytesQueue.put(data);
        } catch (InterruptedException e) {
            Log.e(mytag, "JNI putData: Interrupted while putting data into yuvBytesQueue.");
            Thread.currentThread().interrupt();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.i(mytag, "onDestroy called. Shutting down processing threads and resources.");
        processingRunning = false; // Signal loops to stop

        // 中断线程以使其脱离阻塞队列操作
        if (preProcessThread != null) {
            preProcessThread.interrupt();
        }
        if (inferenceThread != null) {
            inferenceThread.interrupt();
        }
        if (afterProcessThread != null) {
            afterProcessThread.interrupt();
        }

        // 等待线程完成（可选，带超时）
        try {
            if (preProcessThread != null) preProcessThread.join(1000);
            if (inferenceThread != null) inferenceThread.join(1000);
            if (afterProcessThread != null) afterProcessThread.join(1000);
        } catch (InterruptedException e) {
            Log.w(mytag, "Interrupted while joining threads.");
            Thread.currentThread().interrupt();
        }

        // 清空队列（可选，如果线程未完全排空，有助于垃圾回收）
        yuvBytesQueue.clear();
        modelInputQueue.clear();
        modelOutputQueue.clear();
        biSROutputQueue.clear();


        // 释放 RenderScript 资源
        if (inAllocYuvToRgb != null) inAllocYuvToRgb.destroy();
        if (outAllocYuvToRgb != null) outAllocYuvToRgb.destroy();
        if (scriptYuvToRgb != null) scriptYuvToRgb.destroy();
        if (mRsYuvToRgb != null) mRsYuvToRgb.destroy();

        if (inAllocResize != null) inAllocResize.destroy();
        if (outAllocResize != null) outAllocResize.destroy();
        if (scriptResize != null) scriptResize.destroy();
        if (mRsResize != null) mRsResize.destroy();

        // 释放 TFLite 模型
        if (srTFLite != null) {
            srTFLite.close(); // 假设 InferenceTFLite 有一个 close() 方法
        }
        if (biTFLite != null) {
            // biTFLite.close(); // 假设 BiTFLite 也有一个 close() 方法
        }

        // 回收可重用位图，如果它们不受已销毁的 Allocations 管理
        // 与 Allocations 链接的位图（例如通过 createFromBitmap）可能由 RS 管理。
        // 独立位图应回收。
        if (inputBitmap != null && !inputBitmap.isRecycled()) inputBitmap.recycle();
        if (model_input_bitmap != null && !model_input_bitmap.isRecycled()) model_input_bitmap.recycle();
        if (bicubic_output_bitmap != null && !bicubic_output_bitmap.isRecycled()) bicubic_output_bitmap.recycle();


        Log.i(mytag, "onDestroy finished.");
    }

    // 本地方法声明
    public native void mainDecoder(String url);
}

