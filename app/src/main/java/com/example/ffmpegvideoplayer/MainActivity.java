




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
// import androidx.core.util.Pools; // TaggedData Pools are removed
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
    private static BlockingQueue<byte[]> yuvBytesQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<Bitmap> rgbFrameQueueForTfInput = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<Bitmap> rgbFrameQueueForUpsample = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TensorImage> modelInputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TensorBuffer> modelOutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<Bitmap> biSROutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);

    // TaggedData Pools - REMOVED
    // private static final Pools.SimplePool<TaggedData<byte[]>> yuvTaggedDataPool = new Pools.SimplePool<>(QUEUE_CAPACITY + 5);
    // @SuppressWarnings("unchecked") // Suppressing for generic array creation if we were to use array of pools
    // private static final Pools.SimplePool<TaggedData<Bitmap>> bitmapTaggedDataPool = new Pools.SimplePool<>(QUEUE_CAPACITY * 2 + 10); // Bitmap is used in a few queues
    // private static final Pools.SimplePool<TaggedData<TensorImage>> tensorImageTaggedDataPool = new Pools.SimplePool<>(QUEUE_CAPACITY + 5);
    // private static final Pools.SimplePool<TaggedData<TensorBuffer>> tensorBufferTaggedDataPool = new Pools.SimplePool<>(QUEUE_CAPACITY + 5);
 
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
    private Bitmap mDisplayBuffer1;
    private Bitmap mDisplayBuffer2;
    private Canvas mDisplayCanvas1; // Canvas for mDisplayBuffer1
    private Canvas mDisplayCanvas2; // Canvas for mDisplayBuffer2
    private volatile Bitmap mCurrentDisplayFrontBuffer; // The one ImageView should use
    private final Object mDisplayBufferLock = new Object();

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

        // Initialize display buffers
        // Ensure video_output_shape is initialized before this point. It is a member variable.
        // The config ARGB_8888 is common for display.
        if (video_output_shape.getWidth() > 0 && video_output_shape.getHeight() > 0) {
            mDisplayBuffer1 = Bitmap.createBitmap(video_output_shape.getWidth(), video_output_shape.getHeight(), Bitmap.Config.ARGB_8888);
            mDisplayCanvas1 = new Canvas(mDisplayBuffer1);
            mDisplayBuffer2 = Bitmap.createBitmap(video_output_shape.getWidth(), video_output_shape.getHeight(), Bitmap.Config.ARGB_8888);
            mDisplayCanvas2 = new Canvas(mDisplayBuffer2);
            mCurrentDisplayFrontBuffer = mDisplayBuffer1; // Initialize front buffer
        } else {
            Log.e(TAG_MAIN, "video_output_shape is not valid for creating display buffers.");
            // Handle error: perhaps set a flag and don't proceed with display, or use default small buffers
        }

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
                byte[] yuvData = yuvBytesQueue.take();
                // No TaggedData to release
                Log.i(TAG_YUV_RGB, "Processing YUV data"); // Use Log.d for per-frame logs

                long startTime = System.currentTimeMillis();
                inAlloc.copyFrom(yuvData);
                scriptYuvToRgb.setInput(inAlloc);
                scriptYuvToRgb.forEach(outAlloc);
                outAlloc.copyTo(threadLocalInputBitmap); // 转换为 RGB

                long endTime = System.currentTimeMillis();
                long duration = endTime - startTime;
                yuvToRgbTimeMs.set(duration);
                Log.i(TAG_TIME, "YUV->RGB: " + duration + " ms");

                // 将转换后的 RGB 位图的副本放入两个队列
                // Create one master copy from the thread-local bitmap
                Bitmap rgbFrameMasterCopy = threadLocalInputBitmap.copy(threadLocalInputBitmap.getConfig(), false);

                // rgbFrameQueueForUpsample gets a reference to the master copy.
                // Consumers should not recycle this bitmap if it's shared.
                // The current upsampleLoop passes it to OpenGL or makes a copy if placeholder is needed.
                rgbFrameQueueForUpsample.put(rgbFrameMasterCopy);

                // rgbFrameQueueForTfInput gets its own distinct copy to ensure isolation,
                // as TFLite pre-processing might involve transformations or different lifecycles.
                // Bitmap rgbFrameTfInputCopy = rgbFrameMasterCopy.copy(rgbFrameMasterCopy.getConfig(), false); // If a true copy is needed
                // For now, we continue to share the same Bitmap instance as per original logic for tdTfInput.set(rgbFrameMasterCopy, frameNum)
                // Creating a new copy for TfInput queue to ensure data integrity if rgbFrameMasterCopy is modified elsewhere or recycled.
                Bitmap rgbFrameTfInputCopy = rgbFrameMasterCopy.copy(rgbFrameMasterCopy.getConfig(), false);
                rgbFrameQueueForTfInput.put(rgbFrameTfInputCopy);


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
                Bitmap rgbFrame = rgbFrameQueueForTfInput.take();
                // No TaggedData to release
                long takeEnd = System.currentTimeMillis();
                Log.d(TAG_TF_INPUT, "Processing RGB frame for TF input"); // Use Log.d for per-frame logs

                long processingStart = System.currentTimeMillis();
                int cropX = tile_index[0] * TF_INPUT_W;
                int cropY = tile_index[1] * TF_INPUT_H;

                if (cropX + TF_INPUT_W > VIDEO_INPUT_W || cropY + TF_INPUT_H > VIDEO_INPUT_H) {
                    Log.e(TAG_TF_INPUT, "Crop dimensions exceed input. Skipping this frame.");
                    // 可能会放置一个占位符或跳过将数据放入 modelInputQueue
                    // 目前，我们将继续，但这可能会在 createBitmap/drawBitmap 失败时崩溃
                    if (rgbFrame != null && !rgbFrame.isRecycled()) rgbFrame.recycle(); // Recycle the taken frame
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
                modelInputQueue.put(processedModelInputTensor); // 放入 TFLite 输入队列
                if (rgbFrame != null && !rgbFrame.isRecycled()) { // Recycle the original rgbFrame after use
                    rgbFrame.recycle();
                }

                long processingEnd = System.currentTimeMillis(); // 正确的位置
                long takeDuration = takeEnd - takeStart;
                long processingDuration = processingEnd - processingStart;
                prepareTfInputTakeTimeMs.set(takeDuration);
                prepareTfInputProcessingTimeMs.set(processingDuration);
                // 记录详细时间信息
                long loopEnd = System.currentTimeMillis();
                Log.i(TAG_TIME, "TFLite Input - Total: " + (loopEnd - loopStart) + "ms | Take: " + takeDuration + "ms | Proc: " + processingDuration + "ms");

            } catch (InterruptedException e) {
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
            // Pass input dimensions (VIDEO_INPUT_W, VIDEO_INPUT_H) and output dimensions
            if (openGLImageProcessor.setup(VIDEO_INPUT_W, VIDEO_INPUT_H, video_output_shape.getWidth(), video_output_shape.getHeight())) {
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
                Bitmap rgbFrame = rgbFrameQueueForUpsample.take();
                // No TaggedData to release

                if (rgbFrame == null || rgbFrame.isRecycled()) {
                    Log.e(TAG_UPSAMPLE, "Null/recycled frame. Skipping.");
                    continue;
                }

                if (rgbFrame.getWidth() != VIDEO_INPUT_W || rgbFrame.getHeight() != VIDEO_INPUT_H) {
                    Log.e(TAG_UPSAMPLE, "Unexpected frame dims. Expected " +
                            VIDEO_INPUT_W + "x" + VIDEO_INPUT_H + ", got " +
                            rgbFrame.getWidth() + "x" + rgbFrame.getHeight() + ". Skipping.");
                    if (rgbFrame != null && !rgbFrame.isRecycled()) rgbFrame.recycle();
                    continue;
                }

                Log.d(TAG_UPSAMPLE, "Processing frame for upsample (OpenGL ES)");

                long startTime = System.currentTimeMillis();
                Bitmap upscaledBitmap = null;
                if (openGLImageProcessor != null && openGLImageProcessorIsSetup) {
                     upscaledBitmap = openGLImageProcessor.process(rgbFrame);
                } else {
                    Log.e(TAG_UPSAMPLE, "GL processor null/not setup. Skipping frame.");
                    if (rgbFrame != null && !rgbFrame.isRecycled()) {
                        rgbFrame.recycle();
                    }
                    continue;
                }

                long endTime = System.currentTimeMillis();
                long takeDuration = startTime - taketime;
                long processingDuration = endTime - startTime;
                upsampleTakeTimeMs.set(takeDuration);
                upsampleProcessingTimeMs.set(processingDuration);
                Log.i(TAG_TIME, "OpenGL Upscale - Proc: " + processingDuration + " ms | Take: " + takeDuration + " ms");

                Bitmap frameToQueue;
                String logSuffix;

                if (upscaledBitmap == null) {
                    Log.e(TAG_UPSAMPLE, "OpenGLImageProcessor.process() returned null! Critical error. Attempting to use original frame copy as extreme fallback.");
                    if (rgbFrame != null && !rgbFrame.isRecycled()) {
                        frameToQueue = rgbFrame.copy(rgbFrame.getConfig(), false);
                        logSuffix = " (Original - CRITICAL FALLBACK)";
                    } else {
                        Log.e(TAG_UPSAMPLE, "Original rgbFrame is also null/recycled. Cannot queue anything. Skipping frame.");
                        if (rgbFrame != null && !rgbFrame.isRecycled()) rgbFrame.recycle();
                        continue;
                    }
                } else {
                    frameToQueue = upscaledBitmap;
                    logSuffix = " (Processed by GL)";
                }
                biSROutputQueue.put(frameToQueue);
                // Log.d(TAG_UPSAMPLE, "Queued frame" + logSuffix);
                
                // Manage recycling of the input rgbFrame
                // If upscaledBitmap is a new bitmap, the original rgbFrame can be recycled.
                // If upscaledBitmap is the same as rgbFrame (e.g., GL processed in-place or returned input on error), don't recycle.
                // If frameToQueue is a copy of rgbFrame (fallback), original rgbFrame can be recycled.
                if (rgbFrame != null && !rgbFrame.isRecycled()) {
                    if (upscaledBitmap != null && rgbFrame != upscaledBitmap) { // upscaledBitmap is new
                        rgbFrame.recycle();
                    } else if (upscaledBitmap == null && rgbFrame != frameToQueue) { // frameToQueue is a copy of rgbFrame
                        rgbFrame.recycle();
                    }
                    // If rgbFrame == upscaledBitmap OR rgbFrame == frameToQueue (no copy in fallback), it's queued, so don't recycle here.
                }

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
                TensorImage modelInput = modelInputQueue.take();
                // No TaggedData to release
                long takeEnd = System.currentTimeMillis();
                Log.d(TAG_INFERENCE, "Processing model input"); // Use Log.d

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
                Log.i(TAG_TIME, "Inference - Total: " + (loopEnd - loopStart) + "ms | Take: " + takeDuration + "ms | Infer: " + inferenceDuration + "ms");

                modelOutputQueue.put(modelOutput); // 如果队列已满则阻塞

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

        // Deques and Pair-based variables are removed.
        // We will take directly from queues.

        while (processingRunning) {
            Bitmap pooledBitmap = null;
            TensorBuffer hwcOutputTensorBuffer = null;

            try {
                long overallLoopStart = System.currentTimeMillis();
                long syncStartTime = System.currentTimeMillis();

                // Take from BiSR output queue
                pooledBitmap = biSROutputQueue.take();
                if (!processingRunning) {
                    if (pooledBitmap != null && !pooledBitmap.isRecycled() && openGLImageProcessor != null) {
                        openGLImageProcessor.releaseBitmapToPool(pooledBitmap);
                    }
                    break; // Exit loop if processing stopped during take
                }

                // Take from model output queue
                hwcOutputTensorBuffer = modelOutputQueue.take();
                if (!processingRunning) {
                    if (pooledBitmap != null && !pooledBitmap.isRecycled() && openGLImageProcessor != null) {
                        openGLImageProcessor.releaseBitmapToPool(pooledBitmap);
                    }
                    // hwcOutputTensorBuffer (TensorBuffer) doesn't require explicit recycling here
                    break; // Exit loop
                }

                long syncEndTime = System.currentTimeMillis();
                afterProcessSyncTimeMs.set(syncEndTime - syncStartTime);
                Log.d(TAG_AFTER_PROC, "Processing a pair of BiSR and Model frames.");

                // Check if frames are valid (take() should block, but good practice)
                if (pooledBitmap == null || hwcOutputTensorBuffer == null) {
                    Log.e(TAG_AFTER_PROC, "Failed to get a valid frame pair after take(). Skipping.");
                    if (pooledBitmap != null && !pooledBitmap.isRecycled() && openGLImageProcessor != null) {
                        openGLImageProcessor.releaseBitmapToPool(pooledBitmap);
                    }
                    continue;
                }

                // Check if the pooledBitmap is a placeholder (original size) or truly upscaled
                boolean isPlaceholder = pooledBitmap.getWidth() == VIDEO_INPUT_W && pooledBitmap.getHeight() == VIDEO_INPUT_H;
                if (isPlaceholder) {
                    Log.w(TAG_AFTER_PROC, "BiSR frame is placeholder. SR patch may not align.");
                }

                // 1. 将 TFLite TensorBuffer 转换为 ARGB int[] patch
                long tensorProcessingStart = System.currentTimeMillis();
                float[] hwcOutputData = hwcOutputTensorBuffer.getFloatArray();

                int patchH = hwcOutputTensorBuffer.getShape()[1]; // Expected: TF_OUTPUT_H
                int patchW = hwcOutputTensorBuffer.getShape()[2]; // Expected: TF_OUTPUT_W
                int channels = hwcOutputTensorBuffer.getShape()[3]; // Expected: 3 (RGB)

                if (sr_patch_pixels == null || sr_patch_pixels.length != patchW * patchH) {
                    if (patchH != TF_OUTPUT_H || patchW != TF_OUTPUT_W) {
                        Log.w(TAG_AFTER_PROC, "SR patch dims from tensor (" + patchW + "x" + patchH +
                                ") != configured TF_OUTPUT (" + TF_OUTPUT_W + "x" + TF_OUTPUT_H +
                                "). Resizing sr_patch_pixels array to tensor dimensions.");
                    }
                    sr_patch_pixels = new int[patchW * patchH];
                }

                if (channels == 3) {
                    convertFloatToArgbPixels(hwcOutputData, sr_patch_pixels, patchW, patchH, channels);
                } else {
                    Log.e(TAG_AFTER_PROC, "Unexpected channel count from TFLite output: " + channels +
                                          ". Expected 3 (RGB). Skipping native pixel conversion.");
                    // java.util.Arrays.fill(sr_patch_pixels, 0xFFFF00FF); // Magenta for error
                }

                long tensorProcessingEnd = System.currentTimeMillis();
                afterProcessTensorProcessingTimeMs.set(tensorProcessingEnd - tensorProcessingStart);

                // 2. 将 SR patch放置在双三次上采样的基础图像上
                long compositionStart = System.currentTimeMillis();
                int offsetX = tile_index[0] * patchW;
                int offsetY = tile_index[1] * patchH;

                if (offsetX + patchW > pooledBitmap.getWidth() || offsetY + patchH > pooledBitmap.getHeight()) {
                    Log.e(TAG_AFTER_PROC, "SR patch placement exceeds bitmap bounds. Skip setPixels.");
                } else {
                    pooledBitmap.setPixels(sr_patch_pixels, 0, patchW, offsetX, offsetY, patchW, patchH);
                }
                long compositionEnd = System.currentTimeMillis();
                afterProcessCompositionTimeMs.set(compositionEnd - compositionStart);

                // 3. Handle display matrix transformation and prepare final bitmap for UI
                long displayStart = System.currentTimeMillis();
                Bitmap targetDisplayBuffer;
                Canvas targetDisplayCanvas;
                Bitmap sourceBitmapForDisplay = pooledBitmap;

                if (!displayMatrix.isIdentity()) {
                    sourceBitmapForDisplay = Bitmap.createBitmap(pooledBitmap, 0, 0, pooledBitmap.getWidth(), pooledBitmap.getHeight(), displayMatrix, true);
                    if (openGLImageProcessor != null && pooledBitmap != null && !pooledBitmap.isRecycled()) {
                        openGLImageProcessor.releaseBitmapToPool(pooledBitmap);
                    }
                }

                synchronized (mDisplayBufferLock) {
                    if (mCurrentDisplayFrontBuffer == mDisplayBuffer1) {
                        targetDisplayBuffer = mDisplayBuffer2;
                        targetDisplayCanvas = mDisplayCanvas2;
                    } else {
                        targetDisplayBuffer = mDisplayBuffer1;
                        targetDisplayCanvas = mDisplayCanvas1;
                    }

                    if (targetDisplayCanvas != null && sourceBitmapForDisplay != null && !sourceBitmapForDisplay.isRecycled()) {
                        targetDisplayCanvas.drawBitmap(sourceBitmapForDisplay, 0, 0, null);
                    }
                    mCurrentDisplayFrontBuffer = targetDisplayBuffer;
                }
                
                if (sourceBitmapForDisplay == pooledBitmap) {
                    if (openGLImageProcessor != null && sourceBitmapForDisplay != null && !sourceBitmapForDisplay.isRecycled()) {
                        openGLImageProcessor.releaseBitmapToPool(sourceBitmapForDisplay);
                    }
                } else {
                    if (sourceBitmapForDisplay != null && !sourceBitmapForDisplay.isRecycled()) {
                        sourceBitmapForDisplay.recycle();
                    }
                }

                final Bitmap finalBitmapForImageView = mCurrentDisplayFrontBuffer;
                if (finalBitmapForImageView != null) {
                    handler.post(() -> {
                        if (imageView != null && finalBitmapForImageView != null && !finalBitmapForImageView.isRecycled()) {
                            imageView.setImageBitmap(finalBitmapForImageView);
                        }
                    });
                }
                long displayEnd = System.currentTimeMillis();
                afterProcessDisplayPrepTimeMs.set(displayEnd - displayStart);

                long overallLoopEnd = System.currentTimeMillis();
                afterProcessTotalLoopTimeMs.set(overallLoopEnd - overallLoopStart);
                long displayCost = displayEnd - displayStart;

                Log.i(TAG_TIME, "AfterProc - " +
                        " | Sync:" + (syncEndTime - syncStartTime) +
                        " | TensorP:" + (tensorProcessingEnd - tensorProcessingStart) +
                        " | Comp:" + (compositionEnd - compositionStart) +
                        " | DispP:" + displayCost +
                        " | Total:" + (overallLoopEnd - overallLoopStart) + "ms");

                updateTextView();
                Log.i(TAG_AFTER_PROC, "Queue sizes: MO=" + modelOutputQueue.size() + " BiSR=" + biSROutputQueue.size() +
                        " YUV=" + yuvBytesQueue.size() + " RGBIn=" + rgbFrameQueueForTfInput.size() +
                        " RGBUp=" + rgbFrameQueueForUpsample.size());
 
            } catch (InterruptedException e) {
                Log.w(TAG_AFTER_PROC, "Loop interrupted.");
                if (pooledBitmap != null && !pooledBitmap.isRecycled() && openGLImageProcessor != null) {
                    openGLImageProcessor.releaseBitmapToPool(pooledBitmap);
                }
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(TAG_AFTER_PROC, "Error: " + e.getMessage(), e);
                if (pooledBitmap != null && !pooledBitmap.isRecycled() && openGLImageProcessor != null) {
                    openGLImageProcessor.releaseBitmapToPool(pooledBitmap);
                }
            }
        }
        Log.i(TAG_AFTER_PROC, "Loop Finished");
    }

    public void updateTextView() {
        runOnUiThread(() -> {
            if (fpsTextView != null) {
                String displayText = String.format(java.util.Locale.US,
                        "yuv2rgb % 3d; " +
                        "TFInput: % 3d | % 3d ; " +
                        "Upsample: % 3d | % 3d ; " +
                        "Inference: % 3d | % 3d ms; " +
                        "AfterProc: % 3d | % 3d ms | " +
                        "% 3d | % 3d ms; " +
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
            // long currentFrameNumber = frameCounter.getAndIncrement(); // Removed
            yuvBytesQueue.put(data);
        } catch (InterruptedException e) {
            Log.e(TAG_JNI, "putData interrupted");
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
        // frameCounter.set(0); // Removed

        // 释放 TFLite 模型
        if (srTFLite != null) {
            srTFLite.close(); // 假设 InferenceTFLite 有一个 close() 方法
        }

        if (openGLImageProcessor != null) {
            openGLImageProcessor.release();
            openGLImageProcessor = null;
        }

        // Recycle display buffers
        synchronized (mDisplayBufferLock) {
            if (mDisplayBuffer1 != null && !mDisplayBuffer1.isRecycled()) {
                mDisplayBuffer1.recycle();
                mDisplayBuffer1 = null;
            }
            if (mDisplayBuffer2 != null && !mDisplayBuffer2.isRecycled()) {
                mDisplayBuffer2.recycle();
                mDisplayBuffer2 = null;
            }
            mDisplayCanvas1 = null;
            mDisplayCanvas2 = null;
            mCurrentDisplayFrontBuffer = null;
        }
 
        Log.i(TAG_MAIN, "onDestroy finished.");
    }
 
    // 本地方法声明
    public native void mainDecoder(String url);
    private native void convertFloatToArgbPixels(float[] floatArray, int[] intArray, int width, int height, int channels);
}

