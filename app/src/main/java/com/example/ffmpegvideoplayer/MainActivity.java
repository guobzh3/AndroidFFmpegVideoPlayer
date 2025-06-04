




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
import androidx.core.util.Pools;
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

    private static final int QUEUE_CAPACITY = 16;
    private static BlockingQueue<TaggedData<byte[]>> yuvBytesQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TaggedData<Bitmap>> rgbFrameQueueForTfInput = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TaggedData<Bitmap>> rgbFrameQueueForUpsample = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TaggedData<TensorImage>> modelInputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TaggedData<TensorBuffer>> modelOutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TaggedData<Bitmap>> biSROutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);

    private static final AtomicLong frameCounter = new AtomicLong(0);

    // TaggedData Pools
    private static final Pools.SimplePool<TaggedData<byte[]>> yuvTaggedDataPool = new Pools.SimplePool<>(QUEUE_CAPACITY + 5);
    @SuppressWarnings("unchecked") // Suppressing for generic array creation if we were to use array of pools
    private static final Pools.SimplePool<TaggedData<Bitmap>> bitmapTaggedDataPool = new Pools.SimplePool<>(QUEUE_CAPACITY * 2 + 10); // Bitmap is used in a few queues
    private static final Pools.SimplePool<TaggedData<TensorImage>> tensorImageTaggedDataPool = new Pools.SimplePool<>(QUEUE_CAPACITY + 5);
    private static final Pools.SimplePool<TaggedData<TensorBuffer>> tensorBufferTaggedDataPool = new Pools.SimplePool<>(QUEUE_CAPACITY + 5);
 
    // AtomicLongs for storing processing times of different stages
    private final AtomicLong yuvToRgbTimeMs = new AtomicLong(0);
    private final AtomicLong prepareTfInputTakeTimeMs = new AtomicLong(0);
    private final AtomicLong prepareTfInputProcessingTimeMs = new AtomicLong(0);
    private final AtomicLong inferenceTakeTimeMs = new AtomicLong(0);
    private final AtomicLong inferenceTimeMs = new AtomicLong(0);
    private final AtomicLong upsampleTakeTimeMs = new AtomicLong(0);
    private final AtomicLong upsampleProcessingTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessSyncTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessTensorProcessingTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessCompositionTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessDisplayPrepTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessTotalLoopTimeMs = new AtomicLong(0);
 
    // private final static String mytag = "MyNativeCode"; // Replaced by specific tags
    // private final static String time_tag = "time"; // Replaced by specific tags and integrated messages
    private static final String TAG_MAIN = "PlayerActivity";
    private static final String TAG_DECODER = "DecoderThread";
    private static final String TAG_YUV_RGB = "YuvToRgb";
    private static final String TAG_TF_INPUT = "TfInputPrep";
    private static final String TAG_UPSAMPLE = "Upsample";
    private static final String TAG_INFERENCE = "Inference";
    private static final String TAG_AFTER_PROC = "AfterProc";
    private static final String TAG_JNI = "JNI_Bridge";
    private static final String TAG_ERROR = "PlayerError"; // For general errors not specific to a thread
    private static final String TAG_TIME = "PerfTime"; // For performance-specific logs if needed separately
 
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
                Log.i(TAG_MAIN, "Using GPU Delegate for TFLite");
            } else {
                this.srTFLite.addNNApiDelegate();
                Log.i(TAG_MAIN, "Using NNAPI Delegate for TFLite");
            }
            this.srTFLite.initialModel(this);
        } catch (Exception e) {
            Log.e(TAG_ERROR, "Model init error: " + e.getMessage(), e);
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
            Log.i(TAG_DECODER, "Started.");
            mainDecoder(getString(R.string.video_url));
            Log.i(TAG_DECODER, "Finished.");
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
        Log.i(TAG_YUV_RGB, "Loop Started");
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
                TaggedData<byte[]> taggedYuvData = yuvBytesQueue.take();
                byte[] yuvData = taggedYuvData.getData();
                long frameNum = taggedYuvData.getFrameNumber();
                // Release TaggedData<byte[]> to pool
                taggedYuvData.reset();
                yuvTaggedDataPool.release(taggedYuvData);
                Log.i(TAG_YUV_RGB, "Proc frame " + frameNum); // Use Log.d for per-frame logs

                long startTime = System.currentTimeMillis();
                inAlloc.copyFrom(yuvData);
                scriptYuvToRgb.setInput(inAlloc);
                scriptYuvToRgb.forEach(outAlloc);
                outAlloc.copyTo(threadLocalInputBitmap); // 转换为 RGB

                long endTime = System.currentTimeMillis();
                long duration = endTime - startTime;
                yuvToRgbTimeMs.set(duration);
                Log.i(TAG_TIME, "YUV->RGB: " + duration + " ms (F:" + frameNum + ")");

                // 将转换后的 RGB 位图的副本放入两个队列
                // Create one master copy from the thread-local bitmap
                Bitmap rgbFrameMasterCopy = threadLocalInputBitmap.copy(threadLocalInputBitmap.getConfig(), false);

                // rgbFrameQueueForUpsample gets a reference to the master copy.
                // Consumers should not recycle this bitmap if it's shared.
                // The current upsampleLoop passes it to OpenGL or makes a copy if placeholder is needed.
                TaggedData<Bitmap> tdUpsample = bitmapTaggedDataPool.acquire();
                if (tdUpsample == null) tdUpsample = new TaggedData<>();
                tdUpsample.set(rgbFrameMasterCopy, frameNum);
                rgbFrameQueueForUpsample.put(tdUpsample);

                // rgbFrameQueueForTfInput gets its own distinct copy to ensure isolation,
                // as TFLite pre-processing might involve transformations or different lifecycles.
                TaggedData<Bitmap> tdTfInput = bitmapTaggedDataPool.acquire();
                if (tdTfInput == null) tdTfInput = new TaggedData<>();
                tdTfInput.set(rgbFrameMasterCopy, frameNum); // Share the same Bitmap instance
                rgbFrameQueueForTfInput.put(tdTfInput);


            } catch (InterruptedException e) {
                Log.w(TAG_YUV_RGB, "Loop interrupted.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(TAG_YUV_RGB, "Error: " + e.getMessage(), e);
            }
        }
        // 释放 RenderScript 资源
        if (inAlloc != null) inAlloc.destroy();
        if (outAlloc != null) outAlloc.destroy();
        if (scriptYuvToRgb != null) scriptYuvToRgb.destroy();
        if (threadRsYuvToRgb != null) threadRsYuvToRgb.destroy();
        if (threadLocalInputBitmap != null && !threadLocalInputBitmap.isRecycled()) threadLocalInputBitmap.recycle();
        Log.i(TAG_YUV_RGB, "Loop Finished");
    }

    private void prepareTfInputLoop() {
        Log.i(TAG_TF_INPUT, "Loop Started");
        Rect srcRect = new Rect();
        Rect dstRect = new Rect(0, 0, TF_INPUT_W, TF_INPUT_H);

        // 线程局部位图对象
        Bitmap threadLocalModelInputBitmap = Bitmap.createBitmap(TF_INPUT_W, TF_INPUT_H, Bitmap.Config.ARGB_8888);
        Canvas threadLocalModelInputCanvas = new Canvas(threadLocalModelInputBitmap);
        Paint threadLocalModelInputPaint = new Paint(Paint.FILTER_BITMAP_FLAG);

        // Reuse TensorImage object
        TensorImage modelInputTensor = new TensorImage(DataType.FLOAT32);

        while (processingRunning) {
            try {
                long loopStart = System.currentTimeMillis();

                long takeStart = System.currentTimeMillis();
                TaggedData<Bitmap> taggedRgbFrameForTfInput = rgbFrameQueueForTfInput.take();
                Bitmap rgbFrame = taggedRgbFrameForTfInput.getData();
                long frameNum = taggedRgbFrameForTfInput.getFrameNumber();
                // Release TaggedData<Bitmap> to pool
                taggedRgbFrameForTfInput.reset();
                bitmapTaggedDataPool.release(taggedRgbFrameForTfInput);
                long takeEnd = System.currentTimeMillis();
                Log.d(TAG_TF_INPUT, "Proc frame " + frameNum); // Use Log.d for per-frame logs

                long processingStart = System.currentTimeMillis();
                int cropX = tile_index[0] * TF_INPUT_W;
                int cropY = tile_index[1] * TF_INPUT_H;

                if (cropX + TF_INPUT_W > VIDEO_INPUT_W || cropY + TF_INPUT_H > VIDEO_INPUT_H) {
                    Log.e(TAG_TF_INPUT, "Crop dimensions exceed input. Skip frame " + frameNum);
                    // 可能会放置一个占位符或跳过将数据放入 modelInputQueue
                    // 目前，我们将继续，但这可能会在 createBitmap/drawBitmap 失败时崩溃
                    continue; // 跳过当前帧
                }

                srcRect.set(cropX, cropY, cropX + TF_INPUT_W, cropY + TF_INPUT_H);
                threadLocalModelInputCanvas.drawBitmap(rgbFrame, srcRect, dstRect, threadLocalModelInputPaint); // 裁剪到线程局部位图

                // modelInputTensor is now reused, just load new data.
                // Assuming modelInputTensor.load() does not modify the input Bitmap,
                // the copy (threadLocalModelInputBitmap1) is not necessary.
                modelInputTensor.load(threadLocalModelInputBitmap);
                TensorImage processedModelInputTensor = imageProcessorTFLiteInput.process(modelInputTensor);
                // Note: imageProcessorTFLiteInput.process might return a new TensorImage instance or modify in-place.
                // Assuming it returns a new instance or we are fine with modelInputTensor being modified.
                // If it returns a new instance, then modelInputTensor = processedModelInputTensor; is correct.
                // If it modifies in-place, then just use modelInputTensor.
                // For safety and clarity, let's assume process returns the instance to use.
                TaggedData<TensorImage> tdTensorImage = tensorImageTaggedDataPool.acquire();
                if (tdTensorImage == null) tdTensorImage = new TaggedData<>();
                tdTensorImage.set(processedModelInputTensor, frameNum);
                modelInputQueue.put(tdTensorImage); // 放入 TFLite 输入队列

                long processingEnd = System.currentTimeMillis(); // 正确的位置
                long takeDuration = takeEnd - takeStart;
                long processingDuration = processingEnd - processingStart;
                prepareTfInputTakeTimeMs.set(takeDuration);
                prepareTfInputProcessingTimeMs.set(processingDuration);
                // 记录详细时间信息
                long loopEnd = System.currentTimeMillis();
                Log.i(TAG_TIME, "TFLite Input - F:" + frameNum + " Total: " + (loopEnd - loopStart) + "ms | Take: " + takeDuration + "ms | Proc: " + processingDuration + "ms");

            } catch (InterruptedException e) {
                // The previous line was: modelInputQueue.put(new TaggedData<>(modelInputTensor, frameNum));
                // This has been replaced by the logic above using processedModelInputTensor.
                // Ensure the correct TensorImage instance is queued.
                // The 'modelInputQueue.put' is now part of the 'else' block above.
                // The original 'catch' block starts here.
                Log.w(TAG_TF_INPUT, "Loop interrupted.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(TAG_TF_INPUT, "Error: " + e.getMessage(), e);
            }
        }
        if (threadLocalModelInputBitmap != null && !threadLocalModelInputBitmap.isRecycled()) threadLocalModelInputBitmap.recycle();
        Log.i(TAG_TF_INPUT, "Loop Finished");
    }

    private void upsampleLoop() {
        Log.i(TAG_UPSAMPLE, "Loop Started (OpenGL ES)");

        // OpenGLImageProcessor instance is created in mainProcess.
        // Setup is called here on the upsampleThread.
        if (openGLImageProcessor != null && !openGLImageProcessorIsSetup) {
            Log.i(TAG_UPSAMPLE, "Attempting to setup OpenGLImageProcessor.");
            if (openGLImageProcessor.setup(video_output_shape.getWidth(), video_output_shape.getHeight())) {
                openGLImageProcessorIsSetup = true;
                Log.i(TAG_UPSAMPLE, "OpenGLImageProcessor setup successful.");
            } else {
                Log.e(TAG_UPSAMPLE, "Failed to setup OpenGLImageProcessor. Upsampling will be skipped.");
                // openGLImageProcessor = null; // Or handle more gracefully, for now, it will be skipped in the loop.
                // We can't show a Toast from a background thread directly.
                // Consider sending a message to handler if UI feedback is needed.
            }
        }

        while (processingRunning) {
            try {
                long taketime = System.currentTimeMillis();
                TaggedData<Bitmap> taggedRgbFrameForUpsample = rgbFrameQueueForUpsample.take();
                Bitmap rgbFrame = taggedRgbFrameForUpsample.getData();
                long frameNum = taggedRgbFrameForUpsample.getFrameNumber();
                // Release TaggedData<Bitmap> from upsample queue
                taggedRgbFrameForUpsample.reset();
                bitmapTaggedDataPool.release(taggedRgbFrameForUpsample);

                if (rgbFrame == null || rgbFrame.isRecycled()) {
                    Log.e(TAG_UPSAMPLE, "Null/recycled frame " + frameNum + ". Skipping.");
                    continue;
                }

                if (rgbFrame.getWidth() != VIDEO_INPUT_W || rgbFrame.getHeight() != VIDEO_INPUT_H) {
                    Log.e(TAG_UPSAMPLE, "Unexpected frame " + frameNum + " dims. Expected " +
                            VIDEO_INPUT_W + "x" + VIDEO_INPUT_H + ", got " +
                            rgbFrame.getWidth() + "x" + rgbFrame.getHeight() + ". Skipping.");
                    continue;
                }

                Log.d(TAG_UPSAMPLE, "Proc frame " + frameNum + " (OpenGL ES)"); // Use Log.d

                long startTime = System.currentTimeMillis();
                Bitmap upscaledBitmap = null;
                if (openGLImageProcessor != null && openGLImageProcessorIsSetup) {
                     upscaledBitmap = openGLImageProcessor.process(rgbFrame);
                } else {
                    Log.e(TAG_UPSAMPLE, "GL processor null/not setup. Skip frame " + frameNum);
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
                long takeDuration = startTime - taketime;
                long processingDuration = endTime - startTime;
                upsampleTakeTimeMs.set(takeDuration);
                upsampleProcessingTimeMs.set(processingDuration);
                Log.i(TAG_TIME, "OpenGL Upscale - F:" + frameNum + " Proc: " + processingDuration + " ms | Take: " + takeDuration + " ms");

                Bitmap frameToQueue;
                String logSuffix;

                // upscaledBitmap is expected to be non-null (either real data or placeholder from GL processor)
                if (upscaledBitmap == null) {
                    // This case implies a critical failure in OpenGLImageProcessor's placeholder logic.
                    Log.e(TAG_UPSAMPLE, "OpenGLImageProcessor.process() returned null! Critical error for frame: " + frameNum + ". Attempting to use original frame copy as extreme fallback.");
                    if (rgbFrame != null && !rgbFrame.isRecycled()) {
                        frameToQueue = rgbFrame.copy(rgbFrame.getConfig(), false); // Fallback to 1080p copy
                        logSuffix = " (Original - CRITICAL FALLBACK)";
                    } else {
                        Log.e(TAG_UPSAMPLE, "Original rgbFrame is also null/recycled for frame " + frameNum + ". Cannot queue anything. Skipping frame.");
                        continue; // Skip this frame entirely
                    }
                } else {
                    frameToQueue = upscaledBitmap; // This is (almost) always a 4K bitmap (real or placeholder from GL)
                    logSuffix = " (Processed by GL)";
                }
                TaggedData<Bitmap> tdBiSR = bitmapTaggedDataPool.acquire();
                if (tdBiSR == null) tdBiSR = new TaggedData<>();
                tdBiSR.set(frameToQueue, frameNum);
                biSROutputQueue.put(tdBiSR);
                // Log.d(TAG_UPSAMPLE, "Queued frame " + frameNum + logSuffix); // Optional detailed log
                // The input rgbFrame is a copy from yuvToRgbLoop.
                // It's used by openGLImageProcessor.process(). After that, this specific copy
                // is no longer needed by this loop. It will be garbage collected.
                // No explicit recycle here for rgbFrame to avoid complexity with its lifecycle outside this take().

            } catch (InterruptedException e) {
                Log.w(TAG_UPSAMPLE, "Loop interrupted.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(TAG_UPSAMPLE, "Error: " + e.getMessage(), e);
            }
        }
        // OpenGLImageProcessor resources are released in MainActivity's onDestroy
        Log.i(TAG_UPSAMPLE, "Loop Finished (OpenGL ES)");
    }
    private void inferenceLoop() {
        Log.i(TAG_INFERENCE, "Loop Started");
        while (processingRunning) {
            try {
                long loopStart = System.currentTimeMillis();

                long takeStart = System.currentTimeMillis();
                TaggedData<TensorImage> taggedModelInput = modelInputQueue.take();
                TensorImage modelInput = taggedModelInput.getData();
                long frameNum = taggedModelInput.getFrameNumber();
                // Release TaggedData<TensorImage>
                taggedModelInput.reset();
                tensorImageTaggedDataPool.release(taggedModelInput);
                long takeEnd = System.currentTimeMillis();
                Log.d(TAG_INFERENCE, "Proc frame " + frameNum); // Use Log.d

                long inferenceStart = System.currentTimeMillis();
                // 将 TF_OUTPUT_W 和 TF_OUTPUT_H 传递给 superResolution
                // InferenceTFLite 需要 [宽度, 高度] 作为其 tf_output_shape 参数
                TensorBuffer modelOutput = srTFLite.superResolution(modelInput, TF_OUTPUT_SHAPE);
                long inferenceEnd = System.currentTimeMillis();

                // 记录详细时间信息
                long loopEnd = System.currentTimeMillis();
                long takeDuration = takeEnd - takeStart;
                long inferenceDuration = inferenceEnd - inferenceStart;
                inferenceTakeTimeMs.set(takeDuration);
                inferenceTimeMs.set(inferenceDuration);
                Log.i(TAG_TIME, "Inference - F:" + frameNum + " Total: " + (loopEnd - loopStart) + "ms | Take: " + takeDuration + "ms | Infer: " + inferenceDuration + "ms");

                TaggedData<TensorBuffer> tdTensorBuffer = tensorBufferTaggedDataPool.acquire();
                if (tdTensorBuffer == null) tdTensorBuffer = new TaggedData<>();
                tdTensorBuffer.set(modelOutput, frameNum);
                modelOutputQueue.put(tdTensorBuffer); // 如果队列已满则阻塞

            } catch (InterruptedException e) {
                Log.w(TAG_INFERENCE, "Loop interrupted.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(TAG_INFERENCE, "Error: " + e.getMessage(), e);
            }
        }
        Log.i(TAG_INFERENCE, "Loop Finished");
    }

    private void afterProcessLoop() {
        Log.i(TAG_AFTER_PROC, "Loop Started");
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
                long overallLoopStart = System.currentTimeMillis(); // Renamed for clarity
                long syncStartTime = System.currentTimeMillis();

                // Synchronize frames
                while (processingRunning && (bicubicFrameToProcess == null || modelFrameToProcess == null)) {
                    if (pendingBiSRFrames.isEmpty()) {
                        if (!processingRunning) break;
                        TaggedData<Bitmap> takenBiSR = biSROutputQueue.take();
                        pendingBiSRFrames.addLast(takenBiSR);
                    }
                    if (pendingModelFrames.isEmpty()) {
                        if (!processingRunning) break;
                        TaggedData<TensorBuffer> takenModel = modelOutputQueue.take();
                        pendingModelFrames.addLast(takenModel);
                    }

                    TaggedData<Bitmap> headBiSR = pendingBiSRFrames.peekFirst();
                    TaggedData<TensorBuffer> headModel = pendingModelFrames.peekFirst();

                    if (headBiSR == null || headModel == null) { // Should only happen if interrupted during take
                        if (!processingRunning) break;
                        Log.e(TAG_AFTER_PROC, "Peeked null from pending queues. Continuing.");
                        Thread.sleep(5); // Small delay before retrying to avoid busy wait on error
                        continue;
                    }

                    long frameNumBiSR = headBiSR.getFrameNumber();
                    long frameNumModel = headModel.getFrameNumber();

                    if (frameNumBiSR == frameNumModel) {
                        bicubicFrameToProcess = pendingBiSRFrames.removeFirst();
                        modelFrameToProcess = pendingModelFrames.removeFirst();
                        currentFrameNumForProcessing = frameNumModel; // Use matched frame number
                        Log.d(TAG_AFTER_PROC, "Matched frame " + currentFrameNumForProcessing); // Use Log.d
                        // Note: bicubicFrameToProcess and modelFrameToProcess will be released after use later
                    } else if (frameNumBiSR < frameNumModel) {
                        Log.w(TAG_AFTER_PROC, "Discarding stale BiSR frame " + frameNumBiSR + " (Model at " + frameNumModel + ")");
                        TaggedData<Bitmap> staleBiSR = pendingBiSRFrames.removeFirst();
                        // Recycle Bitmap data if necessary and safe
                        if (staleBiSR.getData() != null && !staleBiSR.getData().isRecycled()) {
                           // It's from the OpenGL pool or a placeholder copy, manage its lifecycle there/or let GC handle if it's a copy
                           // For now, just release the TaggedData wrapper
                        }
                        staleBiSR.reset();
                        bitmapTaggedDataPool.release(staleBiSR);
                    } else { // frameNumBiSR > frameNumModel
                        Log.w(TAG_AFTER_PROC, "Discarding stale Model frame " + frameNumModel + " (BiSR at " + frameNumBiSR + ")");
                        TaggedData<TensorBuffer> staleModel = pendingModelFrames.removeFirst();
                        // TensorBuffer data typically doesn't need explicit recycling like Bitmaps
                        staleModel.reset();
                        tensorBufferTaggedDataPool.release(staleModel);
                    }
                } // End of synchronization loop
                long syncEndTime = System.currentTimeMillis();
                afterProcessSyncTimeMs.set(syncEndTime - syncStartTime);

                if (!processingRunning || bicubicFrameToProcess == null || modelFrameToProcess == null) {
                    if (processingRunning) { // If not shutting down but still no match, log error
                        Log.e(TAG_AFTER_PROC, "Failed to get matched frame pair. Skipping.");
                    }
                    continue; // Skip this processing cycle
                }

                // Now we have matched frames: bicubicFrameToProcess and modelFrameToProcess
                // Their frame numbers are currentFrameNumForProcessing.
                Bitmap pooledBitmap = bicubicFrameToProcess.getData();
                TensorBuffer hwcOutputTensorBuffer = modelFrameToProcess.getData();

                // Release the TaggedData wrappers now that we have the data
                bicubicFrameToProcess.reset();
                bitmapTaggedDataPool.release(bicubicFrameToProcess);
                modelFrameToProcess.reset();
                tensorBufferTaggedDataPool.release(modelFrameToProcess);

                // Check if the pooledBitmap is a placeholder (original size) or truly upscaled
                boolean isPlaceholder = pooledBitmap.getWidth() == VIDEO_INPUT_W && pooledBitmap.getHeight() == VIDEO_INPUT_H;
                if (isPlaceholder) {
                    Log.w(TAG_AFTER_PROC, "Frame " + currentFrameNumForProcessing + " BiSR is placeholder. SR patch may not align.");
                }

                // 1. 将 TFLite TensorBuffer 转换为 ARGB int[] patch
                long tensorProcessingStart = System.currentTimeMillis();
                float[] hwcOutputData = hwcOutputTensorBuffer.getFloatArray();

                int patchH = hwcOutputTensorBuffer.getShape()[1]; // Expected: TF_OUTPUT_H
                int patchW = hwcOutputTensorBuffer.getShape()[2]; // Expected: TF_OUTPUT_W
                int channels = hwcOutputTensorBuffer.getShape()[3]; // Expected: 3 (RGB)

                // Ensure sr_patch_pixels is correctly sized
                // This logic remains in Java. The native method will fill this array.
                if (sr_patch_pixels == null || sr_patch_pixels.length != patchW * patchH) {
                    // Log if the dimensions from tensor differ from configured TF_OUTPUT_W/H,
                    // as sr_patch_pixels is usually initialized based on TF_OUTPUT_W/H.
                    if (patchH != TF_OUTPUT_H || patchW != TF_OUTPUT_W) {
                        Log.w(TAG_AFTER_PROC, "SR patch dims from tensor (" + patchW + "x" + patchH +
                                ") != configured TF_OUTPUT (" + TF_OUTPUT_W + "x" + TF_OUTPUT_H +
                                "). Resizing sr_patch_pixels array to tensor dimensions.");
                    }
                    sr_patch_pixels = new int[patchW * patchH];
                }

                // Call the native method to convert float RGB to int ARGB and fill sr_patch_pixels
                if (channels == 3) {
                    convertFloatToArgbPixels(hwcOutputData, sr_patch_pixels, patchW, patchH, channels);
                } else {
                    Log.e(TAG_AFTER_PROC, "Unexpected channel count from TFLite output: " + channels +
                                          ". Expected 3 (RGB). Skipping native pixel conversion for frame " + currentFrameNumForProcessing + ".");
                    // Optionally, fill sr_patch_pixels with a default color (e.g., magenta) to indicate error
                    // java.util.Arrays.fill(sr_patch_pixels, 0xFFFF00FF); // Magenta for error
                }

                long tensorProcessingEnd = System.currentTimeMillis();
                afterProcessTensorProcessingTimeMs.set(tensorProcessingEnd - tensorProcessingStart);

                // 2. 将 SR patch放置在双三次上采样的基础图像上
                long compositionStart = System.currentTimeMillis();
                int offsetX = tile_index[0] * patchW; // Offset by output patch width
                int offsetY = tile_index[1] * patchH; // Offset by output patch height

                if (offsetX + patchW > pooledBitmap.getWidth() || offsetY + patchH > pooledBitmap.getHeight()) {
                    Log.e(TAG_AFTER_PROC, "SR patch placement exceeds bitmap bounds. Skip setPixels.");
                } else {
                    // Apply SR patch directly to the pooledBitmap
                    pooledBitmap.setPixels(sr_patch_pixels, 0, patchW, offsetX, offsetY, patchW, patchH);
                }
                long compositionEnd = System.currentTimeMillis();
                afterProcessCompositionTimeMs.set(compositionEnd - compositionStart);

                // 3. Handle display matrix transformation and prepare final bitmap for UI
                long displayStart = System.currentTimeMillis();
                Bitmap bitmapForUI;
                Bitmap tempBitmapAfterMatrix = pooledBitmap; // Start with the (now patched) pooled bitmap

                if (!displayMatrix.isIdentity()) {
                    // Matrix transformation is needed. Create a new bitmap for this.
                    tempBitmapAfterMatrix = Bitmap.createBitmap(pooledBitmap, 0, 0, pooledBitmap.getWidth(), pooledBitmap.getHeight(), displayMatrix, true);
                    // The original pooledBitmap is no longer directly needed for the display path, release it.
                    if (openGLImageProcessor != null) {
                        openGLImageProcessor.releaseBitmapToPool(pooledBitmap);
                    }
                }

                // Create a final copy for the ImageView.
                // tempBitmapAfterMatrix holds the image data (either original pooled & patched, or transformed & patched).
                bitmapForUI = tempBitmapAfterMatrix.copy(tempBitmapAfterMatrix.getConfig(), false);

                // If tempBitmapAfterMatrix was the original pooled bitmap (i.e., matrix was identity),
                // it needs to be released now that UI has its copy.
                if (tempBitmapAfterMatrix == pooledBitmap) { // True if displayMatrix was identity
                    if (openGLImageProcessor != null) {
                        openGLImageProcessor.releaseBitmapToPool(pooledBitmap);
                    }
                } else {
                    // If tempBitmapAfterMatrix was the result of createBitmap (matrix transform),
                    // it's an intermediate temporary bitmap that can be recycled since bitmapForUI is its copy.
                    if (tempBitmapAfterMatrix != null && !tempBitmapAfterMatrix.isRecycled()) {
                        tempBitmapAfterMatrix.recycle();
                    }
                }

                final Bitmap finalBitmapForImageView = bitmapForUI;
                handler.post(() -> {
                    if (imageView != null) { // Check if imageView is still valid
                        imageView.setImageBitmap(finalBitmapForImageView);
                    } else {
                        // If imageView is null (e.g. activity destroying), recycle the bitmap meant for UI
                        if (finalBitmapForImageView != null && !finalBitmapForImageView.isRecycled()){
                            finalBitmapForImageView.recycle();
                        }
                    }
                });
                long displayEnd = System.currentTimeMillis();
                afterProcessDisplayPrepTimeMs.set(displayEnd - displayStart);

                // 记录详细时间信息
                long overallLoopEnd = System.currentTimeMillis(); // Renamed for clarity
                afterProcessTotalLoopTimeMs.set(overallLoopEnd - overallLoopStart);
                long displayCost = displayEnd - displayStart; // Already captured by afterProcessDisplayPrepTimeMs

                // Log.i(TAG_TIME, "AfterProc - Total: " + (overallLoopEnd - overallLoopStart) + "ms | Tensor: " + (tensorProcessingEnd - tensorProcessingStart) + "ms | Composite: " + (compositionEnd - compositionStart) + "ms | Display: " + displayCost + "ms");
                // 上一行日志中的 takeMiddle 和 takeEnd 未定义，暂时注释掉。
                // 打印部分关键耗时
                Log.i(TAG_TIME, "AfterProc - F:" + currentFrameNumForProcessing +
                        " | Sync:" + (syncEndTime - syncStartTime) +
                        " | TensorP:" + (tensorProcessingEnd - tensorProcessingStart) +
                        " | Comp:" + (compositionEnd - compositionStart) +
                        " | DispP:" + displayCost +
                        " | Total:" + (overallLoopEnd - overallLoopStart) + "ms");

                updateTextView(); // Call without arguments, it will read AtomicLongs
                Log.i(TAG_AFTER_PROC, "Queue sizes: MO=" + modelOutputQueue.size() + " BiSR=" + biSROutputQueue.size() + // Use Log.d
                        " YUV=" + yuvBytesQueue.size() + " RGBIn=" + rgbFrameQueueForTfInput.size() +
                        " RGBUp=" + rgbFrameQueueForUpsample.size());
 
 
            } catch (InterruptedException e) {
                Log.w(TAG_AFTER_PROC, "Loop interrupted.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(TAG_AFTER_PROC, "Error: " + e.getMessage(), e);
            }
        }
        Log.i(TAG_AFTER_PROC, "Loop Finished");
    }
 
    public void updateTextView() {
        runOnUiThread(() -> {
            if (fpsTextView != null) {
                String displayText = String.format(java.util.Locale.US,
                        "YUV->RGB: % 3d ms\n" +
                        "TFInput Take: % 3d ms,   Proc: % 3d ms\n" +
                        "Upsample Take: % 3d ms,  Proc: % 3d ms\n" +
                        "Inference Take: % 3d ms, Infer: % 3d ms\n" +
                        "AfterProc Sync: % 3d ms, Tensor: % 3d ms\n" +
                        "Composite: % 3d ms,      DispPrep: % 3d ms\n" +
                        "AfterProc Total: % 3d ms",
                        yuvToRgbTimeMs.get(),
                        prepareTfInputTakeTimeMs.get(), prepareTfInputProcessingTimeMs.get(),
                        upsampleTakeTimeMs.get(), upsampleProcessingTimeMs.get(),
                        inferenceTakeTimeMs.get(), inferenceTimeMs.get(),
                        afterProcessSyncTimeMs.get(), afterProcessTensorProcessingTimeMs.get(),
                        afterProcessCompositionTimeMs.get(), afterProcessDisplayPrepTimeMs.get(),
                        afterProcessTotalLoopTimeMs.get());
                fpsTextView.setText(displayText);
            }
        });
    }
 
    // 从 JNI 调用
    public static void putData(byte[] data) {
        try {
            long currentFrameNumber = frameCounter.getAndIncrement();
            TaggedData<byte[]> td = yuvTaggedDataPool.acquire();
            if (td == null) {
                td = new TaggedData<>();
            }
            td.set(data, currentFrameNumber);
            yuvBytesQueue.put(td);
        } catch (InterruptedException e) {
            Log.e(TAG_JNI, "putData interrupted for frame ");
            Thread.currentThread().interrupt();
        }
    }
 
    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.i(TAG_MAIN, "onDestroy: Shutting down threads/resources.");
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
            Log.w(TAG_MAIN, "Interrupted while joining threads.");
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
 
        Log.i(TAG_MAIN, "onDestroy finished.");
    }
 
    // 本地方法声明
    public native void mainDecoder(String url);
    private native void convertFloatToArgbPixels(float[] floatArray, int[] intArray, int width, int height, int channels);
}

