




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

    private static final int QUEUE_CAPACITY = 16;
    private static BlockingQueue<SharedByteBuffer> rgbFrameQueueForTfInput = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<SharedByteBuffer> rgbFrameQueueForUpsample = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TensorImage> modelInputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TensorBuffer> modelOutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<InferenceResult> inferenceResultQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<ByteBuffer> safePatchBufferPool = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<InferenceResult> inferenceResultPool = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static final ArrayBlockingQueue<Bitmap> displayQueue = new ArrayBlockingQueue<>(1);
    private static BlockingQueue<TensorImage> tensorImagePool = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
 
 
    // AtomicLongs for storing processing times of different stages
    private final AtomicLong yuvToRgbTimeMs = new AtomicLong(0);
    private final AtomicLong prepareTfInputTakeTimeMs = new AtomicLong(0);
    private final AtomicLong prepareTfInputProcessingTimeMs = new AtomicLong(0);
    private final AtomicLong inferenceTakeTimeMs = new AtomicLong(0);
    private final AtomicLong inferenceTimeMs = new AtomicLong(0);
    private final AtomicLong upsampleTakeTimeMs = new AtomicLong(0);
    private final AtomicLong upsamplePassTimeMs = new AtomicLong(0);
    private final AtomicLong compositePassTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessSyncTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessTensorProcessingTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessCompositionTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessDisplayPrepTimeMs = new AtomicLong(0);
    private final AtomicLong afterProcessTotalLoopTimeMs = new AtomicLong(0);
    private final AtomicLong gpuTotalTimeMs = new AtomicLong(0); // New timer for combined GPU work
 
 
    // A simple data class to hold the results of the inference thread
    private static class InferenceResult {
        ByteBuffer patchBuffer;
        int patchWidth;
        int patchHeight;
        float[] patchRect;
    }
 
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
    private final Runnable displayer = () -> {
        final Bitmap bitmapToDisplay = displayQueue.poll();
        if (bitmapToDisplay == null) return;

        if (imageView != null && !bitmapToDisplay.isRecycled()) {
            Bitmap oldBitmap = mLastDisplayedBitmap;
            imageView.setImageBitmap(bitmapToDisplay);
            mLastDisplayedBitmap = bitmapToDisplay;

            if (oldBitmap != null && oldBitmap != bitmapToDisplay) {
                openGLImageProcessor.releaseBitmapToPool(oldBitmap);
            }
        } else if (openGLImageProcessor != null) {
            openGLImageProcessor.releaseBitmapToPool(bitmapToDisplay);
        }
    };

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

        // Initialize pools to avoid allocation in the loops.
        int patchBufferSize = TF_OUTPUT_W * TF_OUTPUT_H * 3 * 4; // W*H*RGB*sizeof(float)
        TensorBuffer primeBuffer = TensorBuffer.createFixedSize(new int[]{TF_INPUT_H, TF_INPUT_W, 3}, DataType.FLOAT32);
        for (int i = 0; i < QUEUE_CAPACITY; i++) {
            // Pool for patch data buffers
            safePatchBufferPool.offer(ByteBuffer.allocateDirect(patchBufferSize).order(ByteOrder.nativeOrder()));
            // Pool for the data structure that holds the patch data and metadata
            InferenceResult res = new InferenceResult();
            res.patchRect = new float[4];
            inferenceResultPool.offer(res);

            TensorImage ti = new TensorImage(DataType.FLOAT32);
            ti.load(primeBuffer);
            tensorImagePool.offer(ti);
        }

        java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(4);

        prepareTfInputThread = new Thread(() -> prepareTfInputLoop(latch), "PrepareTfInputThread");
        prepareTfInputThread.start();

        upsampleThread = new Thread(() -> upsampleLoop(latch), "UpsampleThread");
        upsampleThread.start();

        inferenceThread = new Thread(() -> inferenceLoop(latch), "InferenceThread");
        inferenceThread.start();
 
        afterProcessThread = new Thread(() -> patchDataLoop(latch), "PatchDataThread"); // Renamed
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

                // Get a pre-allocated TensorImage from the pool.
                TensorImage modelInputTensor = tensorImagePool.take();
                ByteBuffer modelInputByteBuffer = modelInputTensor.getBuffer();

                // Reset buffers for the native call
                rgbBuffer.position(0);
                modelInputByteBuffer.rewind();

                // Call the native function for high-performance processing, writing directly into the TensorImage's buffer.
                cropAndNormalizeRgbaToRgbFloat(rgbBuffer, modelInputByteBuffer, cropX, cropY, TF_INPUT_W, TF_INPUT_H, VIDEO_INPUT_W);

                // The TensorImage is now ready, queue it for the inference thread.
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
            } catch (Throwable t) {
                Log.e(TAG_TF_INPUT, "FATAL Error: " + t.getMessage(), t);
                processingRunning = false;
            } finally {
                if (sharedBuffer != null) {
                    sharedBuffer.release();
                }
            }
        }
        Log.i(TAG_TF_INPUT, "Loop Finished");
    }

    private void upsampleLoop(java.util.concurrent.CountDownLatch latch) {
        Log.i(TAG_UPSAMPLE, "Loop Started (OpenGL ES Rendering Thread)");
 
        if (openGLImageProcessor != null && !openGLImageProcessorIsSetup) {
            Log.i(TAG_UPSAMPLE, "Attempting to setup OpenGLImageProcessor.");
            if (openGLImageProcessor.setup(VIDEO_INPUT_W, VIDEO_INPUT_H, video_output_shape.getWidth(), video_output_shape.getHeight())) {
                openGLImageProcessorIsSetup = true;
                Log.i(TAG_UPSAMPLE, "OpenGLImageProcessor setup successful.");
            } else {
                Log.e(TAG_UPSAMPLE, "Failed to setup OpenGLImageProcessor. Rendering will be skipped.");
                processingRunning = false; // Stop processing if GL fails to set up.
            }
        }
 
        latch.countDown();
        Log.i(TAG_UPSAMPLE, "Initialized and waiting for data.");
        while (processingRunning) {
            SharedByteBuffer sharedBuffer = null;
            InferenceResult inferenceResult = null;
            Bitmap finalCompositedBitmap = null;
 
            try {
                // 1. Synchronize and take from both queues. This is the new pipeline join point.
                sharedBuffer = rgbFrameQueueForUpsample.take();
                inferenceResult = inferenceResultQueue.take();
 
                if (openGLImageProcessor != null && openGLImageProcessorIsSetup) {
                    long gpuWorkStart = System.currentTimeMillis();
 
                    // 2. Perform Pass 1: Upscale
                    ByteBuffer rgbBuffer = sharedBuffer.buffer;
                    rgbBuffer.position(0);
                    long pass1Start = System.currentTimeMillis();
                    openGLImageProcessor.performUpscalePass(rgbBuffer, VIDEO_INPUT_W, VIDEO_INPUT_H);
                    long pass1End = System.currentTimeMillis();
                    upsamplePassTimeMs.set(pass1End - pass1Start);
 
                    // 3. Perform Pass 2: Composite
                    long pass2Start = System.currentTimeMillis();
                    finalCompositedBitmap = openGLImageProcessor.performCompositePass(
                            inferenceResult.patchBuffer,
                            inferenceResult.patchWidth,
                            inferenceResult.patchHeight,
                            inferenceResult.patchRect);
                    long pass2End = System.currentTimeMillis();

                    // CRITICAL: Return the buffer and the result container to their respective pools.
                    if (inferenceResult.patchBuffer != null) {
                        safePatchBufferPool.offer(inferenceResult.patchBuffer);
                    }
                    inferenceResultPool.offer(inferenceResult);
                    compositePassTimeMs.set(pass2End - pass2Start);
 
                    long gpuWorkEnd = System.currentTimeMillis();
                    gpuTotalTimeMs.set(gpuWorkEnd - gpuWorkStart);
 
                    // 5. Post the final bitmap to the UI thread for display
                    if (finalCompositedBitmap != null) {
                        displayQueue.clear();
                        displayQueue.offer(finalCompositedBitmap);
                        handler.post(displayer);
                    }
                }
            } catch (InterruptedException e) {
                Log.w(TAG_UPSAMPLE, "Loop interrupted.");
                Thread.currentThread().interrupt();
                break;
            } catch (Throwable t) {
                Log.e(TAG_UPSAMPLE, "FATAL Error in rendering loop: " + t.getMessage(), t);
                processingRunning = false;
            } finally {
                // IMPORTANT: Release the shared buffer regardless of what happens.
                if (sharedBuffer != null) {
                    sharedBuffer.release();
                }
            }
        }
        Log.i(TAG_UPSAMPLE, "Loop Finished (OpenGL ES Rendering Thread)");
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

                // Return the TensorImage to the pool after inference is done.
                tensorImagePool.offer(modelInput);

            } catch (InterruptedException e) {
                Log.w(TAG_INFERENCE, "Loop interrupted.");
                Thread.currentThread().interrupt();
                break;
            } catch (Throwable t) {
                Log.e(TAG_INFERENCE, "FATAL Error: " + t.getMessage(), t);
                processingRunning = false;
            }
        }
        Log.i(TAG_INFERENCE, "Loop Finished");
    }

    private void patchDataLoop(java.util.concurrent.CountDownLatch latch) {
        Log.i(TAG_AFTER_PROC, "Loop Started (Patch Data Prep)");
        latch.countDown();
        Log.i(TAG_AFTER_PROC, "Initialized and waiting for data.");
 
        while (processingRunning) {
            try {
                // 1. Get the output from the TFLite model.
                TensorBuffer hwcOutputTensorBuffer = modelOutputQueue.take();

                // 2. Get a safe buffer from the pool to copy the inference result into.
                // This avoids both race conditions and per-frame allocations.
                ByteBuffer safeBuffer = safePatchBufferPool.take();
                safeBuffer.clear();

                ByteBuffer originalBuffer = hwcOutputTensorBuffer.getBuffer();
                originalBuffer.rewind();

                // Defensively check capacity, though it should always be sufficient.
                if (safeBuffer.capacity() < originalBuffer.remaining()) {
                    Log.e(TAG_AFTER_PROC, "Pooled buffer is too small! Releasing and skipping frame.");
                    safePatchBufferPool.offer(safeBuffer); // Release it back
                    continue;
                }

                safeBuffer.put(originalBuffer);
                safeBuffer.flip();

                // 3. Get a result container from the pool and populate it.
                InferenceResult result = inferenceResultPool.take();
                result.patchBuffer = safeBuffer;
                result.patchWidth = hwcOutputTensorBuffer.getShape()[2];
                result.patchHeight = hwcOutputTensorBuffer.getShape()[1];


                float rectW = (float)TF_OUTPUT_W / video_output_shape.getWidth();
                float rectH = (float)TF_OUTPUT_H / video_output_shape.getHeight();
                // User confirmed that * 2 is correct for their setup.
                float topLeftX = (float)(tile_index[0] * TF_INPUT_W) / video_output_shape.getWidth() * 2;
                float topLeftY = (float)(tile_index[1] * TF_INPUT_H) / video_output_shape.getHeight() * 2;

                result.patchRect[0] = topLeftX;
                result.patchRect[1] = topLeftY;
                result.patchRect[2] = rectW;
                result.patchRect[3] = rectH;
 
                // 3. Queue the result for the rendering thread
                inferenceResultQueue.put(result);
 
            } catch (InterruptedException e) {
                Log.w(TAG_AFTER_PROC, "Loop interrupted.");
                Thread.currentThread().interrupt();
                break;
            } catch (Throwable t) {
                Log.e(TAG_AFTER_PROC, "FATAL Error in patch data loop: " + t.getMessage(), t);
                processingRunning = false;
            }
        }
        Log.i(TAG_AFTER_PROC, "Loop Finished (Patch Data Prep)");
    }
    

    public void updateTextView() {
        runOnUiThread(() -> {
            if (fpsTextView != null) {
                String displayText = String.format(java.util.Locale.US,
                        "yuv2rgb % 3d; " +
                        "TFInput: % 3d | % 3d ; " +
                        "GPU Pass 1 (Upsample): % 3d; " +
                        "GPU Pass 2 (Composite): % 3d; " +
                        "GPU Total: % 3d; " +
                        "Inference: % 3d | % 3d ms; ",
                        yuvToRgbTimeMs.get(),
                        prepareTfInputTakeTimeMs.get(), prepareTfInputProcessingTimeMs.get(),
                        upsamplePassTimeMs.get(),
                        compositePassTimeMs.get(),
                        gpuTotalTimeMs.get(),
                        inferenceTakeTimeMs.get(), inferenceTimeMs.get());
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
        inferenceResultQueue.clear();
        safePatchBufferPool.clear();
        inferenceResultPool.clear();
        tensorImagePool.clear();
 
        // 释放 TFLite 模型
        if (srTFLite != null) {
            srTFLite.close(); // 假设 InferenceTFLite 有一个 close() 方法
        }
 
        // The last displayed bitmap is from our pool and must be returned to it.
        if (mLastDisplayedBitmap != null && openGLImageProcessor != null) {
            openGLImageProcessor.releaseBitmapToPool(mLastDisplayedBitmap);
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

