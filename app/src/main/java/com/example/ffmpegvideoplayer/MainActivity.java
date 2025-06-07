




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
import java.nio.ByteOrder;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.Deque;
import java.util.LinkedList;
// import androidx.core.util.Pools; // TaggedData Pools are removed
import com.example.ffmpegvideoplayer.analysis.InferenceTFLite;
import android.util.Size;
import org.tensorflow.lite.DataType;
// import org.tensorflow.lite.support.common.ops.NormalizeOp; // No longer needed
// import org.tensorflow.lite.support.image.ImageProcessor; // No longer needed
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;

import com.example.ffmpegvideoplayer.OpenGLImageProcessor;

public class MainActivity extends AppCompatActivity {

    private static final int QUEUE_CAPACITY = 64;
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
    private final AtomicLong upsampleProcessingLastTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessSyncTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessTensorProcessingTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessCompositionTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessDisplayPrepTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessTotalLoopTimeMs = new AtomicLong(0);


 
    // private final static String mytag = "MyNativeCode"; // Replaced by specific tags
    // private final static String time_tag = "time"; // Replaced by specific tags and integrated messages
    private static final String TAG_MAIN = "PlayerActivity";
    private static final String TAG_DECODER = "DecoderThread";
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
    private volatile Bitmap mLastDisplayedBitmap = null;

    private TextView fpsTextView;
    private boolean isPICO = true;

    private final static String deligater="gpu";

    private InferenceTFLite srTFLite;

    private int[] sr_patch_pixels;

    // private ImageProcessor imageProcessorTFLiteInput; // No longer needed

    private volatile boolean processingRunning = true;
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

        // imageProcessorTFLiteInput is no longer needed as normalization is done manually.

        initModel();

        // Initialize the shared buffer pool
        SharedByteBuffer.initialize(VIDEO_INPUT_W * VIDEO_INPUT_H * 4);

        // Display buffers are no longer needed; we will display from the bitmap pool directly.

        mainProcess();

    }

    public void mainProcess() {
        processingRunning = true;

        openGLImageProcessor = new OpenGLImageProcessor(getApplicationContext());
        java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(4);

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


    private void prepareTfInputLoop(java.util.concurrent.CountDownLatch latch) {
        Log.i(TAG_TF_INPUT, "Loop Started");

        // Reuse TensorBuffer and its ByteBuffer to avoid allocations in the loop.
        TensorBuffer inputTensorBuffer = TensorBuffer.createFixedSize(new int[]{TF_INPUT_H, TF_INPUT_W, 3}, DataType.FLOAT32);
        ByteBuffer modelInputByteBuffer = inputTensorBuffer.getBuffer();

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

                // Reset buffers for the native call
                rgbBuffer.position(0);
                modelInputByteBuffer.rewind();

                // Call the native function for high-performance processing
                cropAndNormalizeRgbaToRgbFloat(rgbBuffer, modelInputByteBuffer, cropX, cropY, TF_INPUT_W, TF_INPUT_H, VIDEO_INPUT_W);

                // Create a new TensorImage for each frame to avoid race conditions.
                TensorImage modelInputTensor = new TensorImage(DataType.FLOAT32);
                modelInputTensor.load(inputTensorBuffer);
                modelInputQueue.put(modelInputTensor);

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
        Log.i(TAG_TF_INPUT, "Loop Finished");
    }

    private void upsampleLoop(java.util.concurrent.CountDownLatch latch) {
        Log.i(TAG_UPSAMPLE, "Loop Started (OpenGL ES)");

        if (openGLImageProcessor != null && !openGLImageProcessorIsSetup) {
            Log.i(TAG_UPSAMPLE, "Attempting to setup OpenGLImageProcessor.");
            if (openGLImageProcessor.setup(VIDEO_INPUT_W, VIDEO_INPUT_H, video_output_shape.getWidth(), video_output_shape.getHeight())) {
                openGLImageProcessorIsSetup = true;
                Log.i(TAG_UPSAMPLE, "OpenGLImageProcessor setup successful.");
            } else {
                Log.e(TAG_UPSAMPLE, "Failed to setup OpenGLImageProcessor. Upsampling will be skipped.");
            }
        }

        latch.countDown();
        Log.i(TAG_UPSAMPLE, "Initialized and waiting for data.");
        while (processingRunning) {


            // Next, try to queue a new frame for processing
            SharedByteBuffer sharedBuffer = null;
            try {
                // Poll instead of take, to keep the loop non-blocking.
                // A small timeout prevents a busy-wait loop while allowing responsiveness.
                sharedBuffer = rgbFrameQueueForUpsample.poll(5, java.util.concurrent.TimeUnit.MILLISECONDS);
                if (sharedBuffer != null) {
                    ByteBuffer rgbBuffer = sharedBuffer.buffer;
                    if (rgbBuffer != null && openGLImageProcessor != null && openGLImageProcessorIsSetup) {
                        rgbBuffer.position(0);
                        openGLImageProcessor.queueFrame(rgbBuffer, VIDEO_INPUT_W, VIDEO_INPUT_H);
                    }
                }
                else{
                    // Log.i(TAG_UPSAMPLE, "Didn't put a frame. Move on to next loop.");
                }
                // If sharedBuffer is null, it's fine, we just loop again and check for a processed frame.
            } catch (InterruptedException e) {
                Log.w(TAG_UPSAMPLE, "Loop interrupted while polling for a new frame.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(TAG_UPSAMPLE, "Error in upsample loop: " + e.getMessage(), e);
            } finally {
                if (sharedBuffer != null) {
                    sharedBuffer.release();
                }
            }

            // First, try to get a processed frame without blocking
            if (openGLImageProcessor != null && openGLImageProcessorIsSetup) {
                Bitmap processedBitmap = openGLImageProcessor.getProcessedFrame();
                if (processedBitmap != null) {
                    try {
                        biSROutputQueue.put(processedBitmap);
                    } catch (InterruptedException e) {
                        Log.w(TAG_UPSAMPLE, "Interrupted while queueing processed bitmap.");
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }


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

                if (pooledBitmap == null || hwcOutputTensorBuffer == null) {
                    Log.e(TAG_AFTER_PROC, "Failed to get a valid frame pair after take(). Skipping.");
                    if (pooledBitmap != null && !pooledBitmap.isRecycled() && openGLImageProcessor != null) {
                        openGLImageProcessor.releaseBitmapToPool(pooledBitmap);
                    }
                    continue;
                }

                // CRITICAL FIX: The bitmap from the zero-copy GL pipeline (HardwareBuffer) is IMMUTABLE.
                // We cannot use setPixels() on it. We must create a mutable copy to draw the SR patch on.
                Bitmap mutableBitmapForDisplay = pooledBitmap.copy(Bitmap.Config.ARGB_8888, true);

                // We are done with the original immutable bitmap from the pool, so we release it immediately.
                if (openGLImageProcessor != null) {
                    openGLImageProcessor.releaseBitmapToPool(pooledBitmap);
                }

                // Check if the base frame is a placeholder (original size) or truly upscaled
                boolean isPlaceholder = mutableBitmapForDisplay.getWidth() == VIDEO_INPUT_W && mutableBitmapForDisplay.getHeight() == VIDEO_INPUT_H;
                if (isPlaceholder) {
                    Log.w(TAG_AFTER_PROC, "BiSR frame is placeholder. SR patch may not align.");
                }

                // 1. 将 TFLite TensorBuffer 转换为 ARGB int[] patch
                long tensorProcessingStart = System.currentTimeMillis();
                ByteBuffer hwcOutputBuffer = hwcOutputTensorBuffer.getBuffer();
                int patchH = hwcOutputTensorBuffer.getShape()[1]; // Expected: TF_OUTPUT_H
                int patchW = hwcOutputTensorBuffer.getShape()[2]; // Expected: TF_OUTPUT_W
                int channels = hwcOutputTensorBuffer.getShape()[3]; // Expected: 3 (RGB)

                if (sr_patch_pixels == null || sr_patch_pixels.length != patchW * patchH) {
                     sr_patch_pixels = new int[patchW * patchH];
                }

                if (channels == 3) {
                    convertFloatBufferToArgbPixels(hwcOutputBuffer, sr_patch_pixels, patchW, patchH, channels);
                } else {
                    Log.e(TAG_AFTER_PROC, "Unexpected channel count from TFLite output: " + channels);
                }
                long tensorProcessingEnd = System.currentTimeMillis();
                afterProcessTensorProcessingTimeMs.set(tensorProcessingEnd - tensorProcessingStart);

                // 2. 将 SR patch放置在可变的副本图像上
                long compositionStart = System.currentTimeMillis();
                int offsetX = tile_index[0] * patchW;
                int offsetY = tile_index[1] * patchH;

                if (offsetX + patchW > mutableBitmapForDisplay.getWidth() || offsetY + patchH > mutableBitmapForDisplay.getHeight()) {
                    Log.e(TAG_AFTER_PROC, "SR patch placement exceeds bitmap bounds. Skip setPixels.");
                } else {
                    mutableBitmapForDisplay.setPixels(sr_patch_pixels, 0, patchW, offsetX, offsetY, patchW, patchH);
                }
                long compositionEnd = System.currentTimeMillis();
                afterProcessCompositionTimeMs.set(compositionEnd - compositionStart);

                // 3. Post the final composited bitmap to the UI thread for display
                long displayStart = System.currentTimeMillis();
                final Bitmap finalBitmapToDisplay = mutableBitmapForDisplay;

                handler.post(() -> {
                    if (imageView != null && finalBitmapToDisplay != null && !finalBitmapToDisplay.isRecycled()) {
                        if (!displayMatrix.isIdentity()) {
                            imageView.setImageMatrix(displayMatrix);
                        }
                        imageView.setImageBitmap(finalBitmapToDisplay);

                        // The previous bitmap was also a temporary mutable copy. It's not in the pool, so we must RECYCLE it.
                        if (mLastDisplayedBitmap != null && !mLastDisplayedBitmap.isRecycled()) {
                            if (mLastDisplayedBitmap != finalBitmapToDisplay) {
                                mLastDisplayedBitmap.recycle();
                            }
                        }
                        // Keep track of the new bitmap so it can be recycled on the next frame.
                        mLastDisplayedBitmap = finalBitmapToDisplay;
                    } else {
                        // If we cannot display this bitmap, we must recycle it to avoid a leak.
                        if (finalBitmapToDisplay != null && !finalBitmapToDisplay.isRecycled()) {
                           finalBitmapToDisplay.recycle();
                        }
                    }
                });
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
                Log.i(TAG_AFTER_PROC, "Queue sizes: MO=" + modelOutputQueue.size() + " BiSR=" + biSROutputQueue.size() + " RGBIn=" + rgbFrameQueueForTfInput.size() + " RGBUp=" + rgbFrameQueueForUpsample.size());
 
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
 
    // 从 JNI 调用, now receives RGBA data
    public static void putData(byte[] rgbaData) {
        SharedByteBuffer sharedBuffer = null;
        try {
            sharedBuffer = SharedByteBuffer.obtain();
            if (sharedBuffer == null) {
                Log.e(TAG_JNI, "Failed to obtain a shared buffer for RGBA data. Skipping frame.");
                return;
            }

            sharedBuffer.buffer.position(0);
            sharedBuffer.buffer.put(rgbaData);
            sharedBuffer.buffer.position(0);

            // Add a reference for each queue it's being added to.
            sharedBuffer.addRef(2);
            rgbFrameQueueForUpsample.put(sharedBuffer);
            rgbFrameQueueForTfInput.put(sharedBuffer);

        } catch (InterruptedException e) {
            if (sharedBuffer != null) {
                sharedBuffer.release(); // Release if interrupted before putting into queues
            }
            Log.e(TAG_JNI, "putData interrupted");
            Thread.currentThread().interrupt();
        } catch (Exception e) {
            if (sharedBuffer != null) {
                sharedBuffer.release();
            }
            Log.e(TAG_JNI, "Error in putData: " + e.getMessage(), e);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.i(TAG_MAIN, "onDestroy: Shutting down threads/resources.");
        processingRunning = false; // Signal loops to stop
 
        // 中断线程以使其脱离阻塞队列操作
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
            if (prepareTfInputThread != null) prepareTfInputThread.join(1000);
            if (upsampleThread != null) upsampleThread.join(1000);
            if (inferenceThread != null) inferenceThread.join(1000);
            if (afterProcessThread != null) afterProcessThread.join(1000);
        } catch (InterruptedException e) {
            Log.w(TAG_MAIN, "Interrupted while joining threads.");
            Thread.currentThread().interrupt();
        }
 
        // 清空队列（可选，如果线程未完全排空，有助于垃圾回收）
        // Recycle our manually created composition buffers

        
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
 
        // The last displayed bitmap is a temporary mutable copy, so it must be recycled, not released to the pool.
        if (mLastDisplayedBitmap != null && !mLastDisplayedBitmap.isRecycled()) {
            mLastDisplayedBitmap.recycle();
            mLastDisplayedBitmap = null;
        }

        if (openGLImageProcessor != null) {
            openGLImageProcessor.release();
            openGLImageProcessor = null;
        }
 
        Log.i(TAG_MAIN, "onDestroy finished.");
    }
 
    // 本地方法声明
    public native void mainDecoder(String url);
    private native void cropAndNormalizeRgbaToRgbFloat(ByteBuffer input, ByteBuffer output, int cropX, int cropY, int cropW, int cropH, int inputW);
    private native void convertFloatBufferToArgbPixels(ByteBuffer floatBuffer, int[] intArray, int width, int height, int channels);
}

