package com.example.liveface

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.Window
import android.view.WindowManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.*
import org.opencv.android.CameraActivity
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.io.IOException

class CameraActivity : Activity(), CameraBridgeViewBase.CvCameraViewListener2 {
    companion object {
        private const val TAG = "MainActivity"
        private const val MY_PERMISSIONS_REQUEST_CAMERA = 0
    }

    private var mRgba: Mat? = null
    private var mGray: Mat? = null
    private var mOpenCvCameraView: CameraBridgeViewBase? = null
    // call java class
    // private lateinit var facialExpressionRecognition: FacialExpressionRecognition

    private val mLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.i(TAG, "OpenCv Is loaded")
//                    if(mOpenCvCameraView?.isEnabled == true) {
                        //Log.i(TAG, "YES ENABLED")
                        mOpenCvCameraView?.enableView()

                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // if camera permission is not given it will ask for it on device
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), MY_PERMISSIONS_REQUEST_CAMERA)
        }

        setContentView(R.layout.activity_camera)
        Log.i(TAG, "Activity camera called")
        mOpenCvCameraView = findViewById(R.id.frame_Surface)
        if(mOpenCvCameraView != null){
            Log.d(TAG, "Camera view found!")
        }else{
            Log.d(TAG, "Camera view not found!")
        }
        mOpenCvCameraView?.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView?.setCvCameraViewListener(this)

        // this will load cascade classifier and model
        // this only happen one time when you start CameraActivity
//        try {
//            // input size of model is 48
//            val inputSize = 48
//            Log.i(TAG, "called facial recognition OKKKKKKKKKK")
//           // facialExpressionRecognition = FacialExpressionRecognition(assets, this, "model300.tflite", inputSize)
//        } catch (e: IOException) {
//            Log.i(TAG,"THIS IS ERROR")
//            e.printStackTrace()
//        }
    }

    override fun onResume() {
        super.onResume()
        if (OpenCVLoader.initDebug()) {
            //if load success
            Log.d(TAG, "Opencv initialization is done")
            //mOpenCvCameraView?.enableView()
        } else {
            //if not loaded
            Log.d(TAG, "Opencv is not loaded. try again")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback)
        }
    }

    override fun onPause() {
        super.onPause()
        mOpenCvCameraView?.let {
            it.disableView()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mOpenCvCameraView?.let {
            it.disableView()
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        Log.i(TAG, "OonCameraViewStarted called")
        mRgba = Mat(height, width, CvType.CV_8UC4)
        mGray = Mat(height, width, CvType.CV_8UC1)
    }

    override fun onCameraViewStopped() {
        mRgba?.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat? {
        mRgba = inputFrame?.rgba() ?: mRgba
        mGray = inputFrame?.gray() ?: mGray
        //mRgba = mRgba?.let { facialExpressionRecognition.recognizeImage(it) }
        return mRgba
    }
}