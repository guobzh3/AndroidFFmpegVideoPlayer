




package com.example.ffmpegvideoplayer;

import android.graphics.Bitmap;

import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Canvas;
import android.graphics.Paint;
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

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.Deque;
import java.util.LinkedList;
import com.example.ffmpegvideoplayer.analysis.InferenceTFLite;
import android.util.Size;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;

import com.example.ffmpegvideoplayer.OpenGLImageProcessor;

public class MainActivity extends AppCompatActivity {

    private static final int QUEUE_CAPACITY = 64;
    private static BlockingQueue<TaggedData<byte[]>> yuvBytesQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TaggedData<Bitmap>> rgbFrameQueueForTfInput = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TaggedData<Bitmap>> rgbFrameQueueForUpsample = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TaggedData<TensorImage>> modelInputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TaggedData<TensorBuffer>> modelOutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TaggedData<Bitmap>> biSROutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);

    private static final AtomicLong frameCounter = new AtomicLong(0);

    private final static String mytag = "MyNativeCode";
    private final static String time_tag = "time";

    private static final int VIDEO_INPUT_W = 1920;
    private static final int VIDEO_INPUT_H = 1024;

    private static final int TF_INPUT_W = 480;
    private static final int TF_INPUT_H = 270;

    private static final int TF_OUTPUT_W = 960;
    private static final int TF_OUTPUT_H = 540;
    private static final int[] TF_OUTPUT_SHAPE = new int[]{TF_OUTPUT_W, TF_OUTPUT_H};

    private final Size video_output_shape = new Size(3840, 2048);

    private static final int[] tile_index = new int[]{1, 1};

    static {
        System.loadLibrary("ffmpegvideoplayer");
    }

    private SurfaceView surfaceView;
    private ImageView imageView;
    private Handler handler;

    private TextView fpsTextView;
    private boolean isPICO = false;

    private final static String deligater="gpu";

    private InferenceTFLite srTFLite;

    private int[] sr_patch_pixels;

    private ImageProcessor imageProcessorTFLiteInput;

    private volatile boolean processingRunning = true;
    private Thread yuvToRgbThread;
    private Thread prepareTfInputThread;
    private Thread upsampleThread;
    private Thread inferenceThread;
    private Thread afterProcessThread;
    private OpenGLImageProcessor openGLImageProcessor;
    private volatile boolean openGLImageProcessorIsSetup = false;

    private void initModel() {
        try {
            this.srTFLite = new InferenceTFLite();
            if (deligater.equals("qnn")){
                this.srTFLite.addQNNDelegate(this);
            }
            else if (deligater.equals("gpu")) {
                this.srTFLite.addGPUDelegate();
                Log.i(mytag, "Using GPU Delegate for TFLite (PICO configuration)");
            } else {
                this.srTFLite.addNNApiDelegate();
                Log.i(mytag, "Using NNAPI Delegate for TFLite");
            }
            this.srTFLite.initialModel(this);
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
        imageView = findViewById(R.id.imageView);
        handler = new Handler(Looper.getMainLooper());

        fpsTextView = findViewById(R.id.inference_time);

        sr_patch_pixels = new int[TF_OUTPUT_W * TF_OUTPUT_H];

        imageProcessorTFLiteInput = new ImageProcessor.Builder()
                .add(new NormalizeOp(0f, 255f))
                .build();

        initModel();

        new Thread(() -> {
            Log.i(mytag, "Decoder thread started.");
            mainDecoder(getString(R.string.video_url));
            Log.i(mytag, "Decoder thread finished.");
        }).start();

        mainProcess();
    }

    public void mainProcess() {
        processingRunning = true; // 在启动线程前设置标志

        openGLImageProcessor = new OpenGLImageProcessor(getApplicationContext());
        // Setup will be called in the upsampleLoop thread.

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
        Log.i(mytag, "Upsample Loop Started (OpenGL ES)");

        // OpenGLImageProcessor instance is created in mainProcess.
        // Setup is called here on the upsampleThread.
        if (openGLImageProcessor != null && !openGLImageProcessorIsSetup) {
            Log.i(mytag, "Upsample Loop: Attempting to setup OpenGLImageProcessor.");
            if (openGLImageProcessor.setup(video_output_shape.getWidth(), video_output_shape.getHeight())) {
                openGLImageProcessorIsSetup = true;
                Log.i(mytag, "Upsample Loop: OpenGLImageProcessor setup successful.");
            } else {
                Log.e(mytag, "Upsample Loop: Failed to setup OpenGLImageProcessor. Upsampling will be skipped.");
                // openGLImageProcessor = null; // Or handle more gracefully, for now, it will be skipped in the loop.
                // We can't show a Toast from a background thread directly.
                // Consider sending a message to handler if UI feedback is needed.
            }
        }

        while (processingRunning) {
            try {
                long taketime = System.currentTimeMillis();
                TaggedData<Bitmap> taggedRgbFrameForUpsample = rgbFrameQueueForUpsample.take(); // Get from YUV to RGB queue for upsampling
                Bitmap rgbFrame = taggedRgbFrameForUpsample.getData();
                long frameNum = taggedRgbFrameForUpsample.getFrameNumber();

                if (rgbFrame == null || rgbFrame.isRecycled()) {
                    Log.e(mytag, "Upsample Loop: Received null or recycled frame " + frameNum + ". Skipping.");
                    continue;
                }

                if (rgbFrame.getWidth() != VIDEO_INPUT_W || rgbFrame.getHeight() != VIDEO_INPUT_H) {
                    Log.e(mytag, "Upsample Loop: Received frame " + frameNum + " with unexpected dimensions. Expected " +
                            VIDEO_INPUT_W + "x" + VIDEO_INPUT_H + ", got " +
                            rgbFrame.getWidth() + "x" + rgbFrame.getHeight() + ". Skipping.");
                    continue;
                }

                Log.i(mytag, "Upsample Loop: Processing frame " + frameNum + " with OpenGL ES");

                long startTime = System.currentTimeMillis();
                Bitmap upscaledBitmap = null;
                if (openGLImageProcessor != null && openGLImageProcessorIsSetup) {
                     upscaledBitmap = openGLImageProcessor.process(rgbFrame);
                } else {
                    Log.e(mytag, "Upsample Loop: openGLImageProcessor is null or not setup, cannot process frame " + frameNum + ". Skipping GL processing.");
                    // Skip putting anything into biSROutputQueue, or put original/placeholder
                    // For now, skipping, which might stall downstream if it expects a frame.
                    // Consider putting rgbFrame (original size) or a black bitmap of target size.
                    // Let's put the original rgbFrame to avoid breaking the pipeline completely,
                    // though it won't be upscaled.
                    // upscaledBitmap = rgbFrame.copy(rgbFrame.getConfig(), false); // Fallback
                    // For now, let's just log and skip queueing if GL fails
                    if (rgbFrame != null && !rgbFrame.isRecycled()) {
                        // rgbFrame.recycle(); // It's a copy, let GC handle it if not used.
                    }
                    continue;
                }

                long endTime = System.currentTimeMillis();
                Log.i(time_tag, "OpenGL Bicubic Upscale: " + (endTime - startTime) + " ms" + " (take time: " + (startTime - taketime) + " ms)");

                Bitmap frameToQueue;
                String logSuffix;
                if (upscaledBitmap != null) {
                    frameToQueue = upscaledBitmap;
                    logSuffix = " (Upscaled)";
                } else {
                    // PBO might return null for initial frames if data isn't ready.
                    // Queue the original rgbFrame (a copy of it) as a placeholder to keep frame numbers in sync.
                    Log.w(mytag, "Upsample Loop: OpenGL processing returned null for frame " + frameNum + ". Using original frame as placeholder.");
                    frameToQueue = rgbFrame.copy(rgbFrame.getConfig(), false); // Use a copy of the original
                    logSuffix = " (Original - Placeholder)";
                }
                biSROutputQueue.put(new TaggedData<>(frameToQueue, frameNum));
                // Log.i(mytag, "Upsample Loop: Queued frame " + frameNum + logSuffix); // Optional detailed log
                // The input rgbFrame is a copy from yuvToRgbLoop.
                // It's used by openGLImageProcessor.process(). After that, this specific copy
                // is no longer needed by this loop. It will be garbage collected.
                // No explicit recycle here for rgbFrame to avoid complexity with its lifecycle outside this take().

            } catch (InterruptedException e) {
                Log.w(mytag, "Upsample Loop interrupted. Exiting.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(mytag, "Error in Upsample Loop (OpenGL ES): " + e.getMessage(), e);
            }
        }
        // OpenGLImageProcessor resources are released in MainActivity's onDestroy
        Log.i(mytag, "Upsample Loop Finished (OpenGL ES)");
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
            // matrix.postRotate(0);
            // displayMatrix.postRotate(90);
        }

        Deque<TaggedData<Bitmap>> pendingBiSRFrames = new LinkedList<>();
        Deque<TaggedData<TensorBuffer>> pendingModelFrames = new LinkedList<>();

        while (processingRunning) {
            try {
                TaggedData<Bitmap> bicubicFrameToProcess = null;
                TaggedData<TensorBuffer> modelFrameToProcess = null;
                long currentFrameNumForProcessing = -1;
                long loopStart = System.currentTimeMillis(); // For overall loop timing, move if needed

                // Synchronize frames
                while (processingRunning && (bicubicFrameToProcess == null || modelFrameToProcess == null)) {
                    if (pendingBiSRFrames.isEmpty()) {
                        if (!processingRunning) break;
                        pendingBiSRFrames.addLast(biSROutputQueue.take());
                    }
                    if (pendingModelFrames.isEmpty()) {
                        if (!processingRunning) break;
                        pendingModelFrames.addLast(modelOutputQueue.take());
                    }

                    TaggedData<Bitmap> headBiSR = pendingBiSRFrames.peekFirst();
                    TaggedData<TensorBuffer> headModel = pendingModelFrames.peekFirst();

                    if (headBiSR == null || headModel == null) { // Should only happen if interrupted during take
                        if (!processingRunning) break;
                        Log.e(mytag, "AfterProcess Loop: Peeked null from pending queues. Loop continuing.");
                        Thread.sleep(5); // Small delay before retrying to avoid busy wait on error
                        continue;
                    }

                    long frameNumBiSR = headBiSR.getFrameNumber();
                    long frameNumModel = headModel.getFrameNumber();

                    if (frameNumBiSR == frameNumModel) {
                        bicubicFrameToProcess = pendingBiSRFrames.removeFirst();
                        modelFrameToProcess = pendingModelFrames.removeFirst();
                        currentFrameNumForProcessing = frameNumModel; // Use matched frame number
                        Log.i(mytag, "AfterProcess Loop: Matched and processing frame " + currentFrameNumForProcessing);
                    } else if (frameNumBiSR < frameNumModel) {
                        Log.w(mytag, "AfterProcess Loop: Discarding stale BiSR frame " + frameNumBiSR + " (Model is at " + frameNumModel + ")");
                        TaggedData<Bitmap> staleBiSR = pendingBiSRFrames.removeFirst();
                        if (staleBiSR.getData() != null && !staleBiSR.getData().isRecycled()) {
                            // staleBiSR.getData().recycle(); // Manage bitmap recycling carefully
                        }
                    } else { // frameNumBiSR > frameNumModel
                        Log.w(mytag, "AfterProcess Loop: Discarding stale Model frame " + frameNumModel + " (BiSR is at " + frameNumBiSR + ")");
                        pendingModelFrames.removeFirst(); // Discard stale model frame
                    }
                } // End of synchronization loop

                if (!processingRunning || bicubicFrameToProcess == null || modelFrameToProcess == null) {
                    if (processingRunning) { // If not shutting down but still no match, log error
                        Log.e(mytag, "AfterProcess Loop: Failed to get a matched pair of frames. Skipping processing cycle.");
                    }
                    continue; // Skip this processing cycle
                }

                // Now we have matched frames: bicubicFrameToProcess and modelFrameToProcess
                // Their frame numbers are currentFrameNumForProcessing.
                Bitmap bicubicBaseBitmap = bicubicFrameToProcess.getData();
                TensorBuffer hwcOutputTensorBuffer = modelFrameToProcess.getData();

                // Check if the bicubicBaseBitmap is a placeholder (original size) or truly upscaled
                boolean isPlaceholder = bicubicBaseBitmap.getWidth() == VIDEO_INPUT_W && bicubicBaseBitmap.getHeight() == VIDEO_INPUT_H;
                if (isPlaceholder) {
                    Log.w(mytag, "AfterProcess Loop: Frame " + currentFrameNumForProcessing + " BiSR data is a placeholder (original size). SR patch might not align as expected.");
                }


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
//                Log.i(time_tag, "AfterProcess - Total: " + (loopEnd - loopStart) + "ms | Take: " + (takeMiddle - takeStart) + "ms, " + (takeEnd - takeMiddle)  + "ms | Tensor: " + (tensorProcessingEnd - tensorProcessingStart) + "ms | Composite: " + (compositionEnd - compositionStart) + "ms | Display: " + displayCost + "ms");

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

        if (openGLImageProcessor != null) {
            openGLImageProcessor.release();
            openGLImageProcessor = null;
        }

        Log.i(mytag, "onDestroy finished.");
    }

    // 本地方法声明
    public native void mainDecoder(String url);
}

