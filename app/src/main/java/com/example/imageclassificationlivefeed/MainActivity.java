package com.example.imageclassificationlivefeed;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraManager;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

public class MainActivity extends AppCompatActivity implements ImageReader.OnImageAvailableListener {
    private static final String TAG = "MainActivity";
    private static final int CAMERA_PERMISSION_REQUEST = 121;
    private static final String[] REQUIRED_PERMISSIONS = new String[]{Manifest.permission.CAMERA};
    private static final String MODEL_FILE = "mobilenet_v1_1.0_224.tflite";
    private static final String LABEL_FILE = "mobilenet_v1_1.0_224.txt";
    private static final int INPUT_SIZE = 224;  // Adjust based on your model's input size

    private int sensorOrientation;
    private int previewHeight = 0, previewWidth = 0;
    private boolean isProcessingFrame = false;
    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    private Bitmap rgbFrameBitmap;
    private Classifier classifier;
    private TextView resultTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        resultTextView = findViewById(R.id.result_text);
        
        try {
            classifier = new Classifier(getAssets(), MODEL_FILE, LABEL_FILE, INPUT_SIZE);
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize classifier", e);
            Toast.makeText(this, "Failed to initialize classifier", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        if (allPermissionsGranted()) {
            setFragment();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, CAMERA_PERMISSION_REQUEST);
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (allPermissionsGranted()) {
                setFragment();
            } else {
                Toast.makeText(this, "Camera permission is required to run this app", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    protected void setFragment() {
        try {
            final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
            String cameraId = manager.getCameraIdList()[0];

            CameraConnectionFragment camera2Fragment = CameraConnectionFragment.newInstance(
                    new CameraConnectionFragment.ConnectionCallback() {
                        @Override
                        public void onPreviewSizeChosen(final Size size, final int rotation) {
                            previewHeight = size.getHeight();
                            previewWidth = size.getWidth();
                            Log.d(TAG, "Preview size: " + previewWidth + "x" + previewHeight + " rotation: " + rotation);
                            sensorOrientation = rotation - getScreenOrientation();
                        }
                    },
                    this,
                    R.layout.camera_fragment,
                    new Size(640, 480));

            camera2Fragment.setCamera(cameraId);
            getSupportFragmentManager().beginTransaction()
                    .replace(R.id.container, camera2Fragment)
                    .commit();
        } catch (CameraAccessException e) {
            Log.e(TAG, "Error accessing camera", e);
            Toast.makeText(this, "Error accessing camera", Toast.LENGTH_LONG).show();
            finish();
        }
    }

    //TODO getting frames of live camera footage and passing them to model
    @Override
    public void onImageAvailable(ImageReader reader) {
        // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return;
        }
        if (rgbBytes == null) {
            rgbBytes = new int[previewWidth * previewHeight];
        }
        try {
            final Image image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            if (isProcessingFrame) {
                image.close();
                return;
            }
            isProcessingFrame = true;
            final Image.Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            imageConverter =
                    new Runnable() {
                        @Override
                        public void run() {
                            ImageUtils.convertYUV420ToARGB8888(
                                    yuvBytes[0],
                                    yuvBytes[1],
                                    yuvBytes[2],
                                    previewWidth,
                                    previewHeight,
                                    yRowStride,
                                    uvRowStride,
                                    uvPixelStride,
                                    rgbBytes);
                        }
                    };

            postInferenceCallback =
                    new Runnable() {
                        @Override
                        public void run() {
                            image.close();
                            isProcessingFrame = false;
                        }
                    };

            processImage();

        } catch (final Exception e) {

            return;
        }

    }

    private void processImage() {
        imageConverter.run();
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
        
        // Run classification on the bitmap
        if (classifier != null) {
            List<Classifier.Recognition> results = classifier.recognizeImage(rgbFrameBitmap);
            if (results != null && !results.isEmpty()) {
                StringBuilder resultText = new StringBuilder();
                for (Classifier.Recognition result : results) {
                    resultText.append(result.title).append(", ").append(result.confidence).append("\n");
                }
                runOnUiThread(() -> resultTextView.setText(resultText.toString()));
            }
        }
        
        postInferenceCallback.run();
    }

    protected void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (classifier != null) {
            classifier = null;
        }

    }
}