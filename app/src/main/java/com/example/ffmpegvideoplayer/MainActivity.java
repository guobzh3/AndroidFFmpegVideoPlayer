




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

import java.nio.ByteBuffer;
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
    private static BlockingQueue<SharedByteBuffer> rgbFrameQueueForTfInput = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<SharedByteBuffer> rgbFrameQueueForUpsample = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TensorImage> modelInputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TensorBuffer> modelOutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<Bitmap> biSROutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);


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
    private boolean isPICO = true;

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

        // Initialize the shared buffer pool
        SharedByteBuffer.initialize(VIDEO_INPUT_W * VIDEO_INPUT_H * 4);

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

        mainProcess();

    }

    public void mainProcess() {
        processingRunning = true;

        openGLImageProcessor = new OpenGLImageProcessor(getApplicationContext());
        java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(5);

        yuvToRgbThread = new Thread(() -> yuvToRgbLoop(latch), "YuvToRgbThread");
        yuvToRgbThread.start();

        prepareTfInputThread = new Thread(() -> prepareTfInputLoop(latch), "PrepareTfInputThread");
        prepareTfInputThread.start();

        upsampleThread = new Thread(() -> upsampleLoop(latch), "UpsampleThread");
        upsampleThread.start();

        inferenceThread = new Thread(() -> inferenceLoop(latch), "InferenceThread");
        inferenceThread.start();

        afterProcessThread = new Thread(() -> afterProcessLoop(latch), "AfterProcessThread");
        afterProcessThread.start();

        new Thread(() -> {
            try {
                Log.i(TAG_MAIN, "Waiting for processing threads to initialize...");
                latch.await(); // Wait for all threads to be ready
                Log.i(TAG_MAIN, "All processing threads are ready. Starting decoder.");
                mainDecoder(getString(R.string.video_url));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                Log.e(TAG_MAIN, "Decoder thread interrupted while waiting for processing threads.", e);
            }
        }, "DecoderThread").start();
    }

    private void yuvToRgbLoop(java.util.concurrent.CountDownLatch latch) {
        Log.i(TAG_YUV_RGB, "Loop Started");
        RenderScript threadRsYuvToRgb = RenderScript.create(getApplicationContext());
        Type.Builder yuvTypeBuilder = new Type.Builder(threadRsYuvToRgb, Element.U8(threadRsYuvToRgb))
                .setX(VIDEO_INPUT_W).setY(VIDEO_INPUT_H).setYuvFormat(ImageFormat.YV12);
        Allocation inAlloc = Allocation.createTyped(threadRsYuvToRgb, yuvTypeBuilder.create(), Allocation.USAGE_SCRIPT);
        Type.Builder rgbaTypeBuilder = new Type.Builder(threadRsYuvToRgb, Element.RGBA_8888(threadRsYuvToRgb))
                .setX(VIDEO_INPUT_W).setY(VIDEO_INPUT_H);
        Allocation outAlloc = Allocation.createTyped(threadRsYuvToRgb, rgbaTypeBuilder.create(), Allocation.USAGE_SCRIPT);
        ScriptIntrinsicYuvToRGB scriptYuvToRgb = ScriptIntrinsicYuvToRGB.create(threadRsYuvToRgb, Element.RGBA_8888(threadRsYuvToRgb));

        latch.countDown();
        Log.i(TAG_YUV_RGB, "Initialized and waiting for data.");
        // This thread no longer needs its own bitmap or buffer. It gets them from the pool.
        while (processingRunning) {
            SharedByteBuffer sharedBuffer = null;
            try {
                byte[] yuvData = yuvBytesQueue.take();
                Log.i(TAG_YUV_RGB, "Processing YUV data");

                long startTime = System.currentTimeMillis();
                inAlloc.copyFrom(yuvData);
                scriptYuvToRgb.setInput(inAlloc);
                scriptYuvToRgb.forEach(outAlloc);

                sharedBuffer = SharedByteBuffer.obtain();
                if (sharedBuffer == null) {
                    Log.e(TAG_YUV_RGB, "Failed to obtain a shared buffer. Skipping frame.");
                    continue;
                }

                Bitmap tempBitmap = Bitmap.createBitmap(VIDEO_INPUT_W, VIDEO_INPUT_H, Bitmap.Config.ARGB_8888);
                outAlloc.copyTo(tempBitmap);
                sharedBuffer.buffer.position(0);
                tempBitmap.copyPixelsToBuffer(sharedBuffer.buffer);
                tempBitmap.recycle();
                sharedBuffer.buffer.position(0);

                long endTime = System.currentTimeMillis();
                long duration = endTime - startTime;
                yuvToRgbTimeMs.set(duration);
                Log.i(TAG_TIME, "YUV->RGB to SharedByteBuffer: " + duration + " ms");

                // Add a reference for each queue it's being added to.
                sharedBuffer.addRef(); // For upsample queue
                sharedBuffer.addRef(); // For TF input queue
                rgbFrameQueueForUpsample.put(sharedBuffer);
                rgbFrameQueueForTfInput.put(sharedBuffer);
                
                // The buffer is now owned by the consumers. Do not release here.

            } catch (InterruptedException e) {
                if (sharedBuffer != null) {
                    sharedBuffer.release(); // Release if interrupted before putting into queues
                }
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
        // No bitmap to recycle here anymore
        Log.i(TAG_YUV_RGB, "Loop Finished");
    }

    private void prepareTfInputLoop(java.util.concurrent.CountDownLatch latch) {
        Log.i(TAG_TF_INPUT, "Loop Started");
        Rect srcRect = new Rect();
        Rect dstRect = new Rect(0, 0, TF_INPUT_W, TF_INPUT_H);

        // 线程局部位图对象
        Bitmap threadLocalModelInputBitmap = Bitmap.createBitmap(TF_INPUT_W, TF_INPUT_H, Bitmap.Config.ARGB_8888);
        Canvas threadLocalModelInputCanvas = new Canvas(threadLocalModelInputBitmap);
        Paint threadLocalModelInputPaint = new Paint(Paint.FILTER_BITMAP_FLAG);

        // Reuse TensorImage object
        TensorImage modelInputTensor = new TensorImage(DataType.FLOAT32);

        latch.countDown();
        Log.i(TAG_TF_INPUT, "Initialized and waiting for data.");
        while (processingRunning) {
            SharedByteBuffer sharedBuffer = null;
            try {
                long loopStart = System.currentTimeMillis();

                long takeStart = System.currentTimeMillis();
                sharedBuffer = rgbFrameQueueForTfInput.take();
                ByteBuffer rgbBuffer = sharedBuffer.buffer;
                long takeEnd = System.currentTimeMillis();
                Log.d(TAG_TF_INPUT, "Processing RGB buffer for TF input");

                long processingStart = System.currentTimeMillis();
                int cropX = tile_index[0] * TF_INPUT_W;
                int cropY = tile_index[1] * TF_INPUT_H;

                if (cropX + TF_INPUT_W > VIDEO_INPUT_W || cropY + TF_INPUT_H > VIDEO_INPUT_H) {
                    Log.e(TAG_TF_INPUT, "Crop dimensions exceed input. Skipping this frame.");
                    continue;
                }
                
                Bitmap fullFrameBitmap = Bitmap.createBitmap(VIDEO_INPUT_W, VIDEO_INPUT_H, Bitmap.Config.ARGB_8888);
                rgbBuffer.position(0);
                fullFrameBitmap.copyPixelsFromBuffer(rgbBuffer);

                srcRect.set(cropX, cropY, cropX + TF_INPUT_W, cropY + TF_INPUT_H);
                threadLocalModelInputCanvas.drawBitmap(fullFrameBitmap, srcRect, dstRect, threadLocalModelInputPaint);
                
                fullFrameBitmap.recycle();

                modelInputTensor.load(threadLocalModelInputBitmap);
                TensorImage processedModelInputTensor = imageProcessorTFLiteInput.process(modelInputTensor);

                modelInputQueue.put(processedModelInputTensor);

                long processingEnd = System.currentTimeMillis();
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
            } finally {
                if (sharedBuffer != null) {
                    sharedBuffer.release();
                }
            }
        }
        if (threadLocalModelInputBitmap != null && !threadLocalModelInputBitmap.isRecycled()) threadLocalModelInputBitmap.recycle();
        Log.i(TAG_TF_INPUT, "Loop Finished");
    }

    private void upsampleLoop(java.util.concurrent.CountDownLatch latch) {
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

        latch.countDown();
        Log.i(TAG_UPSAMPLE, "Initialized and waiting for data.");
        while (processingRunning) {
            SharedByteBuffer sharedBuffer = null;
            try {
                long taketime = System.currentTimeMillis();
                sharedBuffer = rgbFrameQueueForUpsample.take();
                ByteBuffer rgbBuffer = sharedBuffer.buffer;


                if (rgbBuffer == null) {
                    Log.e(TAG_UPSAMPLE, "Null buffer. Skipping.");
                    continue;
                }

                Log.d(TAG_UPSAMPLE, "Processing buffer for upsample (OpenGL ES)");

                long startTime = System.currentTimeMillis();
                Bitmap upscaledBitmap = null;
                if (openGLImageProcessor != null && openGLImageProcessorIsSetup) {
                    rgbBuffer.position(0);
                    upscaledBitmap = openGLImageProcessor.process(rgbBuffer, VIDEO_INPUT_W, VIDEO_INPUT_H);
                } else {
                    Log.e(TAG_UPSAMPLE, "GL processor null/not setup. Skipping frame.");
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
                    Log.e(TAG_UPSAMPLE, "OpenGLImageProcessor.process() returned null! Critical error.");
                    // Since we no longer have the original Bitmap, we can't do a fallback here.
                    // We will just skip this frame.
                    continue;
                } else {
                    frameToQueue = upscaledBitmap;
                    logSuffix = " (Processed by GL)";
                }
                biSROutputQueue.put(frameToQueue);
                // Log.d(TAG_UPSAMPLE, "Queued frame" + logSuffix);

                // ByteBuffer does not need to be recycled here. It will be reused.
                // The bitmap `upscaledBitmap` is now in the queue and will be handled by the afterProcessLoop.

            } catch (InterruptedException e) {
                Log.w(TAG_UPSAMPLE, "Loop interrupted.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(TAG_UPSAMPLE, "Error: " + e.getMessage(), e);
            } finally {
                if (sharedBuffer != null) {
                    sharedBuffer.release();
                }
            }
        }
        // OpenGLImageProcessor resources are released in MainActivity's onDestroy
        Log.i(TAG_UPSAMPLE, "Loop Finished (OpenGL ES)");
    }
    private void inferenceLoop(java.util.concurrent.CountDownLatch latch) {
        Log.i(TAG_INFERENCE, "Loop Started");
        latch.countDown();
        Log.i(TAG_INFERENCE, "Initialized and waiting for data.");
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

    private void afterProcessLoop(java.util.concurrent.CountDownLatch latch) {
        Log.i(TAG_AFTER_PROC, "Loop Started");
        Matrix displayMatrix = new Matrix();
        if (!isPICO) { // 手机显示的原始逻辑
            // matrix.postRotate(0);
            // displayMatrix.postRotate(90);
        }

        // Deques and Pair-based variables are removed.
        // We will take directly from queues.

        latch.countDown();
        Log.i(TAG_AFTER_PROC, "Initialized and waiting for data.");
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

