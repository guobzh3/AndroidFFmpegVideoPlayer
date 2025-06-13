




package com.example.ffmpegvideoplayer;

import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.renderscript.ScriptIntrinsicResize;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.util.Log;
import android.view.TextureView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import android.view.Choreographer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Timer;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicLong;
import java.util.Locale;
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

    private static final int QUEUE_CAPACITY = 12;
    private static BlockingQueue<SharedByteBuffer> rgbFrameQueueForTfInput = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<SharedByteBuffer> rgbFrameQueueForUpsample = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TensorImage> modelInputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TensorBuffer> modelOutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<InferenceResult> inferenceResultQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<InferenceResult> inferenceResultPool = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<ByteBuffer> rgbaPatchBufferPool = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static final ArrayBlockingQueue<Bitmap> displayQueue = new ArrayBlockingQueue<>(1);
    private static BlockingQueue<TensorImage> tensorImagePool = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
 
 
    // AtomicLongs for storing processing times of different stages
    // --- Performance Counters ---
    // C++ Timers (updated via JNI)
    private static final AtomicLong decoderLoopTimeUs = new AtomicLong(0);
    private static final AtomicLong decoderReadTimeUs = new AtomicLong(0);
    private static final AtomicLong yuvToRgbTimeUs = new AtomicLong(0);

    // Java Timers (instantaneous values)
    private final AtomicLong prepareTfInputLoopTimeMs = new AtomicLong(0);
    private final AtomicLong prepareTfInputTakeTimeMs = new AtomicLong(0);
    private final AtomicLong prepareTfInputProcessingTimeMs = new AtomicLong(0);
    private final AtomicLong inferenceLoopTimeMs = new AtomicLong(0);
    private final AtomicLong inferenceTakeTimeMs = new AtomicLong(0);
    private final AtomicLong patchDataLoopTimeMs = new AtomicLong(0);
    private final AtomicLong patchDataTakeTimeMs = new AtomicLong(0);
    private final AtomicLong patchDataProcessTimeMs = new AtomicLong(0);
    private final AtomicLong upsampleLoopTimeMs = new AtomicLong(0);
    private final AtomicLong upsampleTakeTimeMs = new AtomicLong(0);

    // Deques for averaging display values (contain nanoseconds)
    private final Deque<Long> recentInferenceTimes = new ConcurrentLinkedDeque<>();
    private final Deque<Long> recentUpsamplePassTimes = new ConcurrentLinkedDeque<>();
    private final Deque<Long> recentCompositePassTimes = new ConcurrentLinkedDeque<>();
    private static final Deque<Long> recentYuvToRgbTimes = new ConcurrentLinkedDeque<>();
    private final Deque<Long> recentPrepareTfInputProcessingTimes = new ConcurrentLinkedDeque<>();
    private final Deque<Long> recentPatchDataProcessingTimes = new ConcurrentLinkedDeque<>();

    // Atomics for holding 1-second averages (in microseconds)
    private final AtomicLong avgInferenceTimeUs = new AtomicLong(0);
    private final AtomicLong avgUpsamplePassTimeUs = new AtomicLong(0);
    private final AtomicLong avgCompositePassTimeUs = new AtomicLong(0);
    private final AtomicLong avgYuvToRgbTimeUs = new AtomicLong(0);
    private final AtomicLong avgPrepareTfInputProcessingTimeUs = new AtomicLong(0);
    private final AtomicLong avgPatchDataProcessingTimeUs = new AtomicLong(0);
 
 
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

    private TextureView textureView;
    private Handler handler; // For UI updates
    private Handler logHandler; // For background logging
    private HandlerThread logThread;
    private static ByteBuffer videoFrameBuffer;

    private TextView fpsTextView;
    private boolean isPICO = true;

    private final static String deligater="gpu";

    private InferenceTFLite srTFLite;


    // private ImageProcessor imageProcessorTFLiteInput; // No longer needed

    private volatile boolean processingRunning = true;
    private Thread prepareTfInputThread;
    private Thread upsampleThread;
    private Thread inferenceThread;
    private Thread afterProcessThread;
    private OpenGLImageProcessor openGLImageProcessor;
    private volatile boolean openGLImageProcessorIsSetup = false;
    private final List<Long> reusableList = new LinkedList<>();
    private final Runnable logRunnable = new Runnable() {
        @Override
        public void run() {
            if (!processingRunning) return;
            
            logPerformanceAndQueueSizes(); // This runs on the logThread
            handler.post(MainActivity.this::updateTextView); // Post UI update back to main thread
            
            if (processingRunning) {
                logHandler.postDelayed(this, 1000); // Schedule next run on logThread
            }
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
            this.srTFLite.initialBufferPool(QUEUE_CAPACITY);
        } catch (Exception e) {
            Log.e(TAG_ERROR, "Model init error: " + e.getMessage(), e);
            Toast.makeText(this, "Model Initialization Failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textureView = findViewById(R.id.textureView);
        handler = new Handler(Looper.getMainLooper()); // Main thread handler for UI
        
        // Create a dedicated thread for logging
        logThread = new HandlerThread("LoggingThread");
        logThread.start();
        logHandler = new Handler(logThread.getLooper()); // logHandler is now on the background thread

        fpsTextView = findViewById(R.id.inference_time);

        textureView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
            @Override
            public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
                if (openGLImageProcessor != null) {
                    openGLImageProcessor.setSurface(surface, width, height);
                }
            }

            @Override
            public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
                if (openGLImageProcessor != null) {
                    openGLImageProcessor.onSurfaceSizeChanged(width, height);
                }
            }

            @Override
            public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
                if (openGLImageProcessor != null) {
                    openGLImageProcessor.setSurface(null, 0, 0);
                }
                return true;
            }

            @Override
            public void onSurfaceTextureUpdated(SurfaceTexture surface) {
                // Invoked every time a new frame is available
            }
        });

        initModel();

        // Initialize the shared buffer pool
        SharedByteBuffer.initialize(VIDEO_INPUT_W * VIDEO_INPUT_H * 4);

        // Display buffers are no longer needed; we will display from the bitmap pool directly.

        mainProcess();
        logHandler.post(logRunnable);

    }

    public void mainProcess() {
        processingRunning = true;

        openGLImageProcessor = new OpenGLImageProcessor(getApplicationContext());

        // Initialize pools to avoid allocation in the loops.
        TensorBuffer primeBuffer = TensorBuffer.createFixedSize(new int[]{TF_INPUT_H, TF_INPUT_W, 3}, DataType.FLOAT32);
        int patchBufferSize = TF_OUTPUT_W * TF_OUTPUT_H * 4;
        for (int i = 0; i < QUEUE_CAPACITY; i++) {
            // Pool for the data structure that holds the patch data and metadata
            InferenceResult res = new InferenceResult();
            res.patchRect = new float[4];
            inferenceResultPool.offer(res);

            TensorImage ti = new TensorImage(DataType.FLOAT32);
            ti.load(primeBuffer);
            tensorImagePool.offer(ti);

            rgbaPatchBufferPool.offer(ByteBuffer.allocateDirect(patchBufferSize).order(ByteOrder.nativeOrder()));
        }

        videoFrameBuffer = ByteBuffer.allocateDirect(VIDEO_INPUT_W * VIDEO_INPUT_H * 4).order(ByteOrder.nativeOrder());

        java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(4);

        prepareTfInputThread = new Thread(() -> prepareTfInputLoop(latch), "PrepareTfInputThread");
        prepareTfInputThread.start();

        upsampleThread = new Thread(() -> upsampleLoop(latch), "UpsampleThread");
        upsampleThread.start();

        inferenceThread = new Thread(() -> inferenceLoop(latch), "InferenceThread");
        inferenceThread.start();
 
        afterProcessThread = new Thread(() -> patchDataLoop(latch), "PatchDataThread"); // Renamed
        afterProcessThread.start();

        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.e(TAG_MAIN, "Initialization interrupted while waiting for processing threads.", e);
            return;
        }
        System.gc();

 
        new Thread(() -> {
            try {
                Thread.sleep(3000);
                Log.i(TAG_MAIN, "All processing threads are ready. Starting decoder.");
                mainDecoder(getString(R.string.video_url), videoFrameBuffer);
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
                long loopStart = System.nanoTime();

                long takeStart = System.nanoTime();
                sharedBuffer = rgbFrameQueueForTfInput.take();
                long takeEnd = System.nanoTime();
                prepareTfInputTakeTimeMs.set((takeEnd - takeStart) / 1_000_000);

                ByteBuffer rgbBuffer = sharedBuffer.buffer;

                long processingStart = System.nanoTime();
                int cropX = tile_index[0] * TF_INPUT_W;
                int cropY = tile_index[1] * TF_INPUT_H;

                if (cropX + TF_INPUT_W > VIDEO_INPUT_W || cropY + TF_INPUT_H > VIDEO_INPUT_H) {
                    Log.e(TAG_TF_INPUT, "Crop dimensions exceed input. Skipping this frame.");
                    continue;
                }

                TensorImage modelInputTensor = tensorImagePool.take();
                ByteBuffer modelInputByteBuffer = modelInputTensor.getBuffer();

                rgbBuffer.position(0);
                modelInputByteBuffer.rewind();

                cropAndNormalizeRgbaToRgbFloat(rgbBuffer, modelInputByteBuffer, cropX, cropY, TF_INPUT_W, TF_INPUT_H, VIDEO_INPUT_W);

                modelInputQueue.put(modelInputTensor);

                long processingEnd = System.nanoTime();
                prepareTfInputProcessingTimeMs.set((processingEnd - processingStart) / 1_000_000);

                long loopEnd = System.nanoTime();
                prepareTfInputLoopTimeMs.set((loopEnd - loopStart) / 1_000_000);
                recentPrepareTfInputProcessingTimes.add(processingEnd - processingStart);

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
            try {
                long loopStart = System.nanoTime();

                // 1. Synchronize and take from both queues. This is the new pipeline join point.
                long takeStart = System.nanoTime();
                sharedBuffer = rgbFrameQueueForUpsample.take();
                inferenceResult = inferenceResultQueue.take();
                long takeEnd = System.nanoTime();
                upsampleTakeTimeMs.set((takeEnd - takeStart) / 1_000_000);

                if (openGLImageProcessor != null && openGLImageProcessorIsSetup) {
                    long gpuWorkStart = System.nanoTime();

                    // 2. Perform Pass 1: Upscale (bi)
                    ByteBuffer rgbBuffer = sharedBuffer.buffer;
                    rgbBuffer.position(0);
                    long pass1Start = System.nanoTime();
                    openGLImageProcessor.performUpscalePass(rgbBuffer, VIDEO_INPUT_W, VIDEO_INPUT_H);
                    long pass1End = System.nanoTime();
                    recentUpsamplePassTimes.add(pass1End - pass1Start);

                    // 3. Perform Pass 2: Composite
                    long pass2Start = System.nanoTime();
                    openGLImageProcessor.performCompositePass(
                            inferenceResult.patchBuffer,
                            inferenceResult.patchWidth,
                            inferenceResult.patchHeight,
                            inferenceResult.patchRect);
                    long pass2End = System.nanoTime();
                    recentCompositePassTimes.add(pass2End - pass2Start);

                    // CRITICAL: Return the result container and its buffer to their respective pools.
                    rgbaPatchBufferPool.offer(inferenceResult.patchBuffer);
                    inferenceResultPool.offer(inferenceResult);
                    inferenceResult = null; // Avoid accidental reuse

                    long gpuWorkEnd = System.nanoTime();
                    
                    // 5. Update the UI - This is now handled by the dedicated log thread
                    // updateTextView();
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
        // Clean up OpenGL resources on the correct thread.
        if (openGLImageProcessor != null) {
            openGLImageProcessor.release();
            openGLImageProcessor = null;
        }
        Log.i(TAG_UPSAMPLE, "Loop Finished and GL resources released (OpenGL ES Rendering Thread)");
    }

    private void inferenceLoop(java.util.concurrent.CountDownLatch latch) {
        Log.i(TAG_INFERENCE, "Loop Started");
        latch.countDown();
        Log.i(TAG_INFERENCE, "Initialized and waiting for data.");
        while (processingRunning) {
            try {
                long loopStart = System.nanoTime();

                long takeStart = System.nanoTime();
                TensorImage modelInput = modelInputQueue.take();
                long takeEnd = System.nanoTime();
                inferenceTakeTimeMs.set((takeEnd - takeStart) / 1_000_000);

                long inferenceStart = System.nanoTime();
                TensorBuffer modelOutput = srTFLite.superResolution(modelInput);
                long inferenceEnd = System.nanoTime();
                recentInferenceTimes.add(inferenceEnd - inferenceStart);

                if (modelOutput != null) {
                    modelOutputQueue.put(modelOutput);
                }

                tensorImagePool.offer(modelInput);

                long loopEnd = System.nanoTime();
                inferenceLoopTimeMs.set((loopEnd - loopStart) / 1_000_000);

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
            TensorBuffer hwcOutputTensorBuffer = null;
            try {
                long loopStart = System.nanoTime();

                // 1. Get the output from the TFLite model.
                long takeStart = System.nanoTime();
                hwcOutputTensorBuffer = modelOutputQueue.take();
                long takeEnd = System.nanoTime();
                patchDataTakeTimeMs.set((takeEnd - takeStart) / 1_000_000);

                long processStart = System.nanoTime();

                // 2. Get the original buffer and a destination buffer for conversion.
                ByteBuffer originalBuffer = hwcOutputTensorBuffer.getBuffer();
                originalBuffer.rewind();

                ByteBuffer rgbaBuffer = rgbaPatchBufferPool.take();
                rgbaBuffer.rewind();

                // 3. Perform the conversion from Float RGB to Uint8 RGBA.
                convertFloatRgbToRgbaUint8(originalBuffer, rgbaBuffer, TF_OUTPUT_W, TF_OUTPUT_H);
                rgbaBuffer.rewind();

                // 4. Get a result container from the pool and populate it.
                InferenceResult result = inferenceResultPool.take();
                result.patchBuffer = rgbaBuffer; // Use the newly converted buffer
                result.patchWidth = TF_OUTPUT_W;
                result.patchHeight = TF_OUTPUT_H;

                float rectW = (float)TF_OUTPUT_W / video_output_shape.getWidth();
                float rectH = (float)TF_OUTPUT_H / video_output_shape.getHeight();
                float topLeftX = (float)(tile_index[0] * TF_INPUT_W) / video_output_shape.getWidth() * 2;
                float topLeftY = (float)(tile_index[1] * TF_INPUT_H) / video_output_shape.getHeight() * 2;

                result.patchRect[0] = topLeftX;
                result.patchRect[1] = topLeftY;
                result.patchRect[2] = rectW;
                result.patchRect[3] = rectH;

                // 5. Queue the result for the rendering thread
                inferenceResultQueue.put(result);

                long processEnd = System.nanoTime();
                patchDataProcessTimeMs.set((processEnd - processStart) / 1_000_000);

                long loopEnd = System.nanoTime();
                patchDataLoopTimeMs.set((loopEnd - loopStart) / 1_000_000);
                recentPatchDataProcessingTimes.add(processEnd - processStart);

            } catch (InterruptedException e) {
                Log.w(TAG_AFTER_PROC, "Loop interrupted.");
                Thread.currentThread().interrupt();
                break;
            } catch (Throwable t) {
                Log.e(TAG_AFTER_PROC, "FATAL Error in patch data loop: " + t.getMessage(), t);
                processingRunning = false;
            } finally {
                if (hwcOutputTensorBuffer != null) {
                    srTFLite.releaseBuffer(hwcOutputTensorBuffer);
                }
            }
        }
        Log.i(TAG_AFTER_PROC, "Loop Finished (Patch Data Prep)");
    }
    

    public void updateTextView() {
        runOnUiThread(() -> {
            if (fpsTextView != null) {
                // Values are in microseconds, convert to ms for display
                double avgSrMs = avgInferenceTimeUs.get() / 1000.0;
                double avgBiMs = avgUpsamplePassTimeUs.get() / 1000.0;
                double avgCompMs = avgCompositePassTimeUs.get() / 1000.0;
                double avgYuvMs = avgYuvToRgbTimeUs.get() / 1000.0;
                double avgTfInProcMs = avgPrepareTfInputProcessingTimeUs.get() / 1000.0; // Changed to ProcessingTime
                double avgPatchProcMs = avgPatchDataProcessingTimeUs.get() / 1000.0; // Changed to ProcessingTime

                String displayText = String.format(Locale.US,
                        "YUV: %.2f | TFIn: %.2f | SR: %.2f | BI: %.2f | Patch: %.2f | Comp: %.2f", // Reordered
                        avgYuvMs, avgTfInProcMs, avgSrMs, avgBiMs, avgPatchProcMs, avgCompMs // Reordered
                );
                fpsTextView.setText(displayText);
            }
        });
    }

    private long calculateAndClearAverage(Deque<Long> dataQueue, List<Long> reusableList) {
        if (dataQueue.isEmpty()) {
            return 0;
        }
        long sum = 0;
        int count = 0;
        // Drain the queue to the reusable list to avoid allocation
        reusableList.clear();
        while (!dataQueue.isEmpty()) {
            reusableList.add(dataQueue.poll());
        }

        for (Long nanoTime : reusableList) {
            sum += nanoTime;
            count++;
        }
        // Return average in microseconds
        return (sum / count) / 1000;
    }

    private void logPerformanceAndQueueSizes() {
        // Calculate averages for the last second using the reusable list
        avgInferenceTimeUs.set(calculateAndClearAverage(recentInferenceTimes, reusableList));
        avgUpsamplePassTimeUs.set(calculateAndClearAverage(recentUpsamplePassTimes, reusableList));
        avgCompositePassTimeUs.set(calculateAndClearAverage(recentCompositePassTimes, reusableList));
        avgYuvToRgbTimeUs.set(calculateAndClearAverage(recentYuvToRgbTimes, reusableList));
        avgPrepareTfInputProcessingTimeUs.set(calculateAndClearAverage(recentPrepareTfInputProcessingTimes, reusableList));
        avgPatchDataProcessingTimeUs.set(calculateAndClearAverage(recentPatchDataProcessingTimes, reusableList));

        String logMsg = String.format(Locale.US,
                """
                        TICK --- Loops(ms): [Dec: %.2f, TFIn: %.2f, Infer: %d, Patch: %.2f, GL: %d] ---\s
                        Takes(ms): [Read: %.2f, TFIn: %d, Infer: %d, Patch: %d, GL: %d] ---\s
                        Process(ms): [YUV: %.2f, TFIn: %d, SR: %.2f, Patch: %d, BI: %.2f, Comp: %.2f] ---\s
                        Queues: [tfIn: %d, upsample: %d, modelIn: %d, modelOut: %d, result: %d]""",
                // Loop Times
                decoderLoopTimeUs.get() / 1000.0,
                avgPrepareTfInputProcessingTimeUs.get() / 1000.0, // Changed to avg
                inferenceLoopTimeMs.get(),
                avgPatchDataProcessingTimeUs.get() / 1000.0, // Changed to avg
                upsampleLoopTimeMs.get(),
                // Take Times
                decoderReadTimeUs.get() / 1000.0,
                prepareTfInputTakeTimeMs.get(),
                inferenceTakeTimeMs.get(),
                patchDataTakeTimeMs.get(),
                upsampleTakeTimeMs.get(),
                // Processing Times
                avgYuvToRgbTimeUs.get() / 1000.0, // Changed to avg
                prepareTfInputProcessingTimeMs.get(),
                avgInferenceTimeUs.get() / 1000.0,
                patchDataProcessTimeMs.get(),
                avgUpsamplePassTimeUs.get() / 1000.0,
                avgCompositePassTimeUs.get() / 1000.0,
                // Queue Sizes
                rgbFrameQueueForTfInput.size(),
                rgbFrameQueueForUpsample.size(),
                modelInputQueue.size(),
                modelOutputQueue.size(),
                inferenceResultQueue.size()
        );
        Log.i(TAG_TIME, logMsg);
    }
 
    // --- JNI Callbacks ---
    public static void updateDecoderTimings(long loopTimeUs, long readTimeUs) {
        decoderLoopTimeUs.set(loopTimeUs);
        decoderReadTimeUs.set(readTimeUs);
    }

    public static void updateYuvToRgbTime(long durationUs) {
        yuvToRgbTimeUs.set(durationUs);
        // Assuming durationUs is in microseconds, convert to nanoseconds for the Deque
        // or ensure consistency in units. The Deques are currently in nanoseconds.
        // Let's convert to nanoseconds for consistency.
        // The original recent*Times deques store nanoseconds, so let's convert durationUs to nanoseconds.
        recentYuvToRgbTimes.add(durationUs * 1000);
    }

    // JNI calls this when a frame is ready in the pre-allocated buffer
    public static void onFrameReady() {
        SharedByteBuffer sharedBuffer = null;
        try {
            sharedBuffer = SharedByteBuffer.obtain();
            if (sharedBuffer == null) {
                Log.e(TAG_JNI, "Failed to obtain a shared buffer. Skipping frame.");
                return;
            }

            videoFrameBuffer.position(0);
            sharedBuffer.buffer.position(0);
            sharedBuffer.buffer.put(videoFrameBuffer);
            sharedBuffer.buffer.position(0);

            // Add a reference for each queue it's being added to.
            sharedBuffer.addRef(2);
            rgbFrameQueueForUpsample.put(sharedBuffer);
            rgbFrameQueueForTfInput.put(sharedBuffer);

        } catch (InterruptedException e) {
            if (sharedBuffer != null) {
                sharedBuffer.release();
            }
            Log.e(TAG_JNI, "onFrameReady interrupted");
            Thread.currentThread().interrupt();
        } catch (Exception e) {
            if (sharedBuffer != null) {
                sharedBuffer.release();
            }
            Log.e(TAG_JNI, "Error in onFrameReady: " + e.getMessage(), e);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.i(TAG_MAIN, "onDestroy: Shutting down threads/resources.");
        processingRunning = false; // Signal loops to stop
 
        if (logThread != null) {
            logHandler.removeCallbacks(logRunnable);
            logThread.quitSafely(); // Safely quit the looper
            try {
                logThread.join(500); // Wait for the thread to finish
            } catch (InterruptedException e) {
                Log.e(TAG_MAIN, "Interrupted while joining log thread.", e);
            }
        }

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
        inferenceResultPool.clear();
        tensorImagePool.clear();
 
        // 释放 TFLite 模型
        if (srTFLite != null) {
            srTFLite.close(); // 假设 InferenceTFLite 有一个 close() 方法
        }
 
        // With TextureView, there's no mLastDisplayedBitmap to manage.

        // The release of openGLImageProcessor is now handled within the upsampleLoop
        // to ensure it happens on the correct GL thread.
 
        Log.i(TAG_MAIN, "onDestroy finished.");
    }
 
    // 本地方法声明
    public native void mainDecoder(String url, ByteBuffer buffer);
    private native void cropAndNormalizeRgbaToRgbFloat(ByteBuffer input, ByteBuffer output, int cropX, int cropY, int cropW, int cropH, int inputW);
    private native void convertFloatRgbToRgbaUint8(ByteBuffer floatRgbInput, ByteBuffer rgbaUint8Output, int width, int height);
}

