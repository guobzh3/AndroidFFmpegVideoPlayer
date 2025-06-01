//计划：
// 1 把preprocess分开。
// 1.5 换一种bi。
// 2 inference能不能放在dsp上？




package com.example.ffmpegvideoplayer;

import android.graphics.Bitmap;
// import android.graphics.BitmapFactory; // Not used
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Canvas;
import android.graphics.Paint; // Keep if model_input_bitmap drawing needs it, though direct drawBitmap might not.
// import android.graphics.BitmapShader; // Not used
// import android.graphics.Shader; // Not used
// import android.graphics.RectF; // Not used
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
// import androidx.core.graphics.Insets; // For EdgeToEdge, keep if that's used elsewhere
// import androidx.core.view.ViewCompat; // For EdgeToEdge
// import androidx.core.view.WindowInsetsCompat; // For EdgeToEdge

// import java.io.InputStream; // Not used
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import com.example.ffmpegvideoplayer.analysis.InferenceTFLite;
import android.util.Size;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
// import org.tensorflow.lite.support.image.ops.ResizeOp; // Will be removed from MainActivity's ImageProcessor
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
// import android.renderscript.ScriptC; // Not used directly

// import android.opengl.GLES31; // For GLSurfaceView, keep
// import android.opengl.GLSurfaceView; // For GLSurfaceView, keep
// import android.opengl.GLES20; // For GLSurfaceView, keep

import com.example.ffmpegvideoplayer.analysis.BiTFLite; // Keep, initialized

public class MainActivity extends AppCompatActivity {

    private static final int QUEUE_CAPACITY = 64;
    private static BlockingQueue<byte[]> yuvBytesQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TensorImage> modelInputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    private static BlockingQueue<TensorBuffer> modelOutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);
    // private static BlockingQueue<int[]> viewOutQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY); // Unused in provided logic
    private static BlockingQueue<Bitmap> biSROutputQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);

    private final static String mytag = "MyNativeCode";
    private final static String time_tag = "time";

    // Define input/output dimensions clearly
    // Video input dimensions (from decoder)
    private static final int VIDEO_INPUT_W = 1920; // Original: video_input_shape[0]
    private static final int VIDEO_INPUT_H = 1024; // Original: video_input_shape[1]

    // TFLite model input patch dimensions (cropped from video input)
    private static final int TF_INPUT_W = 480;    // Original: tf_input_shape[0]
    private static final int TF_INPUT_H = 270;    // Original: tf_input_shape[1]

    // TFLite model output patch dimensions (super-resolved)
    private static final int TF_OUTPUT_W = 960;   // Original: tf_output_shape[0]
    private static final int TF_OUTPUT_H = 540;   // Original: tf_output_shape[1]

    // Final video output dimensions (for display, where SR patch is placed)
    private final Size video_output_shape = new Size(3840, 2048); // Width, Height

    // Tile processing (original logic preserved)
    private static final int[] tile_index = new int[]{1, 1}; // {tile_x_index, tile_y_index}
    // private static int[] tile_split = new int[] {VIDEO_INPUT_W / 2 , VIDEO_INPUT_H / 2}; // SRx4 example
    // private static int[] tile_split = new int[] {VIDEO_INPUT_W / 4 , VIDEO_INPUT_H / 4}; // SRx2 example (original, but unused in crop)

    static {
        System.loadLibrary("ffmpegvideoplayer");
    }

    private SurfaceView surfaceView;
    // private SurfaceHolder surfaceHolder; // Initialized but not directly used in mainProcess
    private ImageView imageView;
    private Handler handler;
    // private GLSurfaceView mGLSurfaceView; // Initialized but not directly used in mainProcess

    private TextView frameSizeTextView;
    private TextView fpsTextView;
    private boolean isPICO = false; // Configuration flag

    private final static String deligater="qnn";

    private InferenceTFLite srTFLite;
    private BiTFLite biTFLite; // Initialized, but its inference not in mainProcess

    // Reusable Bitmaps
    private Bitmap inputBitmap;        // For YUV->RGB conversion output (full fra me)
    private Bitmap model_input_bitmap; // For TFLite input (cropped patch), reused
    private Bitmap bicubic_output_bitmap; // For RenderScript bicubic upscale output (full frame), reused

    // Reusable array for TFLite output patch pixels
    private int[] sr_patch_pixels;

    // RenderScript objects
    private RenderScript mRsYuvToRgb;
    private Allocation inAllocYuvToRgb;
    private Allocation outAllocYuvToRgb;
    private ScriptIntrinsicYuvToRGB scriptYuvToRgb;

    private RenderScript mRsResize;
    private Allocation inAllocResize;
    private Allocation outAllocResize;
    private ScriptIntrinsicResize scriptResize;

    private ImageProcessor imageProcessorTFLiteInput;

    // Thread management
    private volatile boolean processingRunning = true;
    private Thread preProcessThread;
    private Thread inferenceThread;
    private Thread afterProcessThread;

    private Canvas modelInputCanvas; // For drawing cropped region onto model_input_bitmap
    private Paint modelInputPaint;   // Optional, for drawing options

    private void initModel() {
        try {
            this.srTFLite = new InferenceTFLite();
            this.biTFLite = new BiTFLite(); // Original initialization
            if (deligater.equals("qnn")){
                this.srTFLite.addQNNDelegate(this);
                this.biTFLite.addQNNDelegate(this); // Original
            }
            else if (deligater.equals("gpu")) {
                this.srTFLite.addGPUDelegate();
                this.biTFLite.addGPUDelegate(); // Original
                Log.i(mytag, "Using GPU Delegate for TFLite (PICO configuration)");
            } else {
                this.srTFLite.addNNApiDelegate();
                this.biTFLite.addNNApiDelegate(); // Original
                Log.i(mytag, "Using NNAPI Delegate for TFLite");
            }
            this.srTFLite.initialModel(this);
            this.biTFLite.initialModel(this); // Original
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
        // surfaceHolder = surfaceView.getHolder(); // Original
        imageView = findViewById(R.id.imageView);
        // mGLSurfaceView = findViewById(R.id.glSurfaceView); // Assuming an ID if it exists
        handler = new Handler(Looper.getMainLooper());

        frameSizeTextView = findViewById(R.id.frame_size);
        fpsTextView = findViewById(R.id.inference_time);

        // Initialize reusable Bitmaps
        inputBitmap = Bitmap.createBitmap(VIDEO_INPUT_W, VIDEO_INPUT_H, Bitmap.Config.ARGB_8888);
        model_input_bitmap = Bitmap.createBitmap(TF_INPUT_W, TF_INPUT_H, Bitmap.Config.ARGB_8888);
        bicubic_output_bitmap = Bitmap.createBitmap(video_output_shape.getWidth(), video_output_shape.getHeight(), Bitmap.Config.ARGB_8888);

        // Canvas for model_input_bitmap
        modelInputCanvas = new Canvas(model_input_bitmap);
        modelInputPaint = new Paint(Paint.FILTER_BITMAP_FLAG); // For quality if scaling down, though crop is 1:1 here

        // Initialize reusable pixel array for SR patch
        // This will be sized based on actual model output later, but for now, use configured TF_OUTPUT sizes
        sr_patch_pixels = new int[TF_OUTPUT_W * TF_OUTPUT_H];


        // Init RenderScript for YUV to RGB
        mRsYuvToRgb = RenderScript.create(getApplicationContext());
        Type.Builder yuvTypeBuilder = new Type.Builder(mRsYuvToRgb, Element.U8(mRsYuvToRgb))
                .setX(VIDEO_INPUT_W).setY(VIDEO_INPUT_H).setYuvFormat(ImageFormat.YV12); // Assuming YV12 from JNI
        inAllocYuvToRgb = Allocation.createTyped(mRsYuvToRgb, yuvTypeBuilder.create(), Allocation.USAGE_SCRIPT);

        Type.Builder rgbaTypeBuilder = new Type.Builder(mRsYuvToRgb, Element.RGBA_8888(mRsYuvToRgb))
                .setX(VIDEO_INPUT_W).setY(VIDEO_INPUT_H);
        outAllocYuvToRgb = Allocation.createTyped(mRsYuvToRgb, rgbaTypeBuilder.create(), Allocation.USAGE_SCRIPT);
        scriptYuvToRgb = ScriptIntrinsicYuvToRGB.create(mRsYuvToRgb, Element.RGBA_8888(mRsYuvToRgb));

        // Init RenderScript for Bicubic Resize
        mRsResize = RenderScript.create(getApplicationContext());
        inAllocResize = Allocation.createFromBitmap(mRsResize, inputBitmap); // Input for resize is the full RGB frame
        outAllocResize = Allocation.createFromBitmap(mRsResize, bicubic_output_bitmap); // Output is the large bicubic upscaled bitmap
        scriptResize = ScriptIntrinsicResize.create(mRsResize);

        // ImageProcessor for TFLite input (NO ResizeOp here, as model_input_bitmap is already correct size)
        imageProcessorTFLiteInput = new ImageProcessor.Builder()
                .add(new NormalizeOp(0f, 255f))
                // Add QuantizeOp and CastOp here if srTFLite model is UINT8 and IS_INT8 flag in InferenceTFLite is true
                // This needs to align with InferenceTFLite's IS_INT8 logic and model requirements.
                // For now, assuming FLOAT32 model or IS_INT8=false in InferenceTFLite.
                .build();

        initModel();

        // Start decoder thread (original JNI call)
        new Thread(() -> {
            Log.i(mytag, "Decoder thread started.");
            mainDecoder(getString(R.string.video_url)); // Ensure R.string.video_url is defined
            Log.i(mytag, "Decoder thread finished.");
        }).start();

        mainProcess();
    }

    public void mainProcess() {
        processingRunning = true; // Set flag before starting threads

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

                byte[] yuvData = yuvBytesQueue.take(); // Blocks if empty

                // 1. YUV to RGB using RenderScript
                long startTime = System.currentTimeMillis();
                inAllocYuvToRgb.copyFrom(yuvData);
                scriptYuvToRgb.setInput(inAllocYuvToRgb);
                scriptYuvToRgb.forEach(outAllocYuvToRgb);
                outAllocYuvToRgb.copyTo(inputBitmap); // inputBitmap now holds full RGB frame
                long endTime = System.currentTimeMillis();
                Log.i(time_tag, "preProcess - YUV to RGB: " + (endTime - startTime) + " ms");

                // 2. Prepare TFLite model input (cropping and processing)
                startTime = System.currentTimeMillis();
                int cropX = tile_index[0] * TF_INPUT_W;
                int cropY = tile_index[1] * TF_INPUT_H;

                // Ensure crop area is within bounds of inputBitmap
                if (cropX + TF_INPUT_W > VIDEO_INPUT_W || cropY + TF_INPUT_H > VIDEO_INPUT_H) {
                    Log.e(mytag, "Crop dimensions exceed inputBitmap bounds. Skipping frame for TFLite.");
                    // Potentially put a placeholder or skip putting to modelInputQueue
                    // For now, we'll proceed, but this could crash if createBitmap/drawBitmap fails
                }

                srcRect.set(cropX, cropY, cropX + TF_INPUT_W, cropY + TF_INPUT_H);
                modelInputCanvas.drawBitmap(inputBitmap, srcRect, dstRect, modelInputPaint); // model_input_bitmap now holds cropped patch

                TensorImage modelInputTensor = new TensorImage(DataType.FLOAT32); // Assuming FLOAT32, adjust if model is quantized
                modelInputTensor.load(model_input_bitmap);
                modelInputTensor = imageProcessorTFLiteInput.process(modelInputTensor);
                endTime = System.currentTimeMillis();
                Log.i(time_tag, "preProcess - Create TFLite Input: " + (endTime - startTime) + " ms");

                modelInputQueue.put(modelInputTensor); // Blocks if full

                // 3. Perform Bicubic Upscale for the rest of the image using RenderScript
                startTime = System.currentTimeMillis();
                // Update inAllocResize if inputBitmap content changed and it's not auto-synced.
                // createFromBitmap links the Allocation to the Bitmap. If Bitmap pixels change,
                // inAllocResize needs to be notified or recreated if the link is not live.
                // For safety, or if issues arise, one might re-copy:
                inAllocResize.copyFrom(inputBitmap); // Ensure fresh data from inputBitmap
                scriptResize.setInput(inAllocResize);
                scriptResize.forEach_bicubic(outAllocResize);
                outAllocResize.copyTo(bicubic_output_bitmap); // bicubic_output_bitmap now has full upscaled image
                endTime = System.currentTimeMillis();
                Log.i(time_tag, "preProcess - Bicubic Upscale: " + (endTime - startTime) + " ms");

                // Put a COPY of the bicubic_output_bitmap into the queue
                // This is important if afterProcessLoop modifies the bitmap it receives
                Bitmap biSrOutputCopy = bicubic_output_bitmap.copy(bicubic_output_bitmap.getConfig(), true);
                biSROutputQueue.put(biSrOutputCopy); // Blocks if full

                endTimeFullLoop = System.currentTimeMillis();
                costTimeFullLoop = endTimeFullLoop - startTimeFullLoop;
                Log.i(time_tag, "preProcess - Full Loop Time: " + costTimeFullLoop + " ms. Queue sizes: YUV=" + yuvBytesQueue.size() + " ModelIn=" + modelInputQueue.size() + " BiSR=" + biSROutputQueue.size());


            } catch (InterruptedException e) {
                Log.w(mytag, "PreProcessing Loop interrupted. Exiting.");
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                Log.e(mytag, "Error in PreProcessing Loop: " + e.getMessage(), e);
                // Continue loop or break based on error severity
            }
        }
        Log.i(mytag, "PreProcessing Loop Finished");
    }

    private void inferenceLoop() {
        Log.i(mytag, "Inference Loop Started");
        while (processingRunning) {
            try {
                TensorImage modelInput = modelInputQueue.take(); // Blocks if empty

                long startTime = System.currentTimeMillis();
                // Pass TF_OUTPUT_W and TF_OUTPUT_H to superResolution
                // InferenceTFLite expects [Width, Height] for its tf_output_shape parameter
                TensorBuffer modelOutput = srTFLite.superResolution(modelInput, new int[]{TF_OUTPUT_W, TF_OUTPUT_H});
                long endTime = System.currentTimeMillis();
                Log.i(time_tag, "Inference - SR Time: " + (endTime - startTime) + " ms");
                // Log.i(mytag, "Input: " + modelInput.getWidth() + "x" + modelInput.getHeight());
                // Log.i(mytag, "Output shape: " + Arrays.toString(modelOutput.getShape()));

                modelOutputQueue.put(modelOutput); // Blocks if full

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
        if (!isPICO) { // Original logic for phone display
            // matrix.postRotate(0); // Original was 0, if 90 is needed:
            // displayMatrix.postRotate(90);
        }

        while (processingRunning) {
            try {
                long startTimeFullLoop = System.currentTimeMillis();
                TensorBuffer hwcOutputTensorBuffer = modelOutputQueue.take(); // Blocks if empty
                Bitmap bicubicBaseBitmap = biSROutputQueue.take(); // Blocks if empty

                // 1. Convert TFLite TensorBuffer to ARGB int[] patch
                long startTime = System.currentTimeMillis();
                float[] hwcOutputData = hwcOutputTensorBuffer.getFloatArray();

                // Get actual dimensions from the tensor buffer (should match TF_OUTPUT_W, TF_OUTPUT_H if model is consistent)
                // Shape is typically [Batch, Height, Width, Channels] for TFLite image models (NHWC)
                // Or [Batch, Width, Height, Channels] if InferenceTFLite constructed it that way.
                // Based on InferenceTFLite's previous logic: shape was [1, H_param, W_param, 3]
                // where H_param was tf_output_shape[1] and W_param was tf_output_shape[0] from MainActivity.
                // So, if MainActivity sends [TF_OUTPUT_W, TF_OUTPUT_H], then inside InferenceTFLite:
                // H_param = TF_OUTPUT_H, W_param = TF_OUTPUT_W.
                // Resulting TensorBuffer shape: [1, TF_OUTPUT_H, TF_OUTPUT_W, 3]
                int batch = hwcOutputTensorBuffer.getShape()[0]; // Should be 1
                int patchH = hwcOutputTensorBuffer.getShape()[1]; // Expected: TF_OUTPUT_H
                int patchW = hwcOutputTensorBuffer.getShape()[2]; // Expected: TF_OUTPUT_W
                int channels = hwcOutputTensorBuffer.getShape()[3]; // Expected: 3

                if (patchH != TF_OUTPUT_H || patchW != TF_OUTPUT_W) {
                    Log.w(mytag, "Warning: SR output patch dimensions (" + patchW + "x" + patchH +
                            ") differ from configured TF_OUTPUT (" + TF_OUTPUT_W + "x" + TF_OUTPUT_H + "). Resizing sr_patch_pixels array.");
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

                // 2. Place SR patch onto the bicubic upscaled base image
                startTime = System.currentTimeMillis();
                int offsetX = tile_index[0] * patchW; // Offset by output patch width
                int offsetY = tile_index[1] * patchH; // Offset by output patch height

                if (offsetX + patchW > bicubicBaseBitmap.getWidth() || offsetY + patchH > bicubicBaseBitmap.getHeight()) {
                    Log.e(mytag, "SR patch placement exceeds bicubicBaseBitmap bounds. Skipping setPixels.");
                } else {
                    // bicubicBaseBitmap is the one taken from biSROutputQueue, which is a copy.
                    // So, modifying it here is safe.
                    bicubicBaseBitmap.setPixels(sr_patch_pixels, 0, patchW, offsetX, offsetY, patchW, patchH);
                }
                endTime = System.currentTimeMillis();
                Log.i(time_tag, "afterProcess - Composite Patch: " + (endTime - startTime) + " ms");

                // 3. Apply matrix transformation (if any) and display
                startTime = System.currentTimeMillis();
                Bitmap finalBitmapToDisplay = bicubicBaseBitmap; // Default
                if (!displayMatrix.isIdentity()) { // Only create new bitmap if transformation is needed
                    finalBitmapToDisplay = Bitmap.createBitmap(bicubicBaseBitmap, 0, 0,
                            bicubicBaseBitmap.getWidth(), bicubicBaseBitmap.getHeight(), displayMatrix, true);
                }

                final Bitmap displayBitmap = finalBitmapToDisplay; // Effectively final for lambda
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

    public void updateTextView(String time) { // Changed param name for clarity
        runOnUiThread(() -> {
            if (fpsTextView != null) { // Check for null in case view is not ready/gone
                fpsTextView.setText(time);
            }
            // frameSizeTextView.setText(...); // Original didn't update this one here
        });
    }

    // Called from JNI
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

        // Interrupt threads to break them out of blocking queue operations
        if (preProcessThread != null) {
            preProcessThread.interrupt();
        }
        if (inferenceThread != null) {
            inferenceThread.interrupt();
        }
        if (afterProcessThread != null) {
            afterProcessThread.interrupt();
        }

        // Wait for threads to finish (optional, with timeout)
        try {
            if (preProcessThread != null) preProcessThread.join(1000);
            if (inferenceThread != null) inferenceThread.join(1000);
            if (afterProcessThread != null) afterProcessThread.join(1000);
        } catch (InterruptedException e) {
            Log.w(mytag, "Interrupted while joining threads.");
            Thread.currentThread().interrupt();
        }

        // Clear queues (optional, helps GC if threads didn't fully drain them)
        yuvBytesQueue.clear();
        modelInputQueue.clear();
        modelOutputQueue.clear();
        biSROutputQueue.clear();


        // Release RenderScript resources
        if (inAllocYuvToRgb != null) inAllocYuvToRgb.destroy();
        if (outAllocYuvToRgb != null) outAllocYuvToRgb.destroy();
        if (scriptYuvToRgb != null) scriptYuvToRgb.destroy();
        if (mRsYuvToRgb != null) mRsYuvToRgb.destroy();

        if (inAllocResize != null) inAllocResize.destroy();
        if (outAllocResize != null) outAllocResize.destroy();
        if (scriptResize != null) scriptResize.destroy();
        if (mRsResize != null) mRsResize.destroy();

        // Release TFLite models
        if (srTFLite != null) {
            srTFLite.close(); // Assuming InferenceTFLite has a close() method
        }
        if (biTFLite != null) {
            // biTFLite.close(); // Assuming BiTFLite also has a close() method
        }

        // Recycle reusable bitmaps if they are not managed by Allocations that are destroyed
        // Bitmaps linked to Allocations (e.g. via createFromBitmap) might be managed by RS.
        // Standalone bitmaps should be recycled.
        if (inputBitmap != null && !inputBitmap.isRecycled()) inputBitmap.recycle();
        if (model_input_bitmap != null && !model_input_bitmap.isRecycled()) model_input_bitmap.recycle();
        if (bicubic_output_bitmap != null && !bicubic_output_bitmap.isRecycled()) bicubic_output_bitmap.recycle();


        Log.i(mytag, "onDestroy finished.");
    }

    // Native method declaration (as in original)
    public native void mainDecoder(String url);
}