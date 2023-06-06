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
import com.example.liveface.databinding.ActivityMainBinding
import org.opencv.android.*
import org.opencv.android.CameraActivity
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import java.io.IOException


class MainActivity : Activity(), CameraBridgeViewBase.CvCameraViewListener2 {
    companion object {
        private const val TAG = "MainActivity"
        private const val MY_PERMISSIONS_REQUEST_CAMERA = 0
    }
    private lateinit var binding: ActivityMainBinding
    private lateinit var mRgba: Mat
    private lateinit var mGray: Mat
    private var mOpenCvCameraView: CameraBridgeViewBase? = null
    private lateinit var facialExpressionRecognition: FacialExpressionRecognition
    private var mCameraId:Int =0

    private val mLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.i(TAG, "OpenCv Is loaded")

                    activateOpenCVCameraView()
                    try {
                        // input size of model is 48
                        val inputSize = 48
                        Log.i(TAG, "called facial recognition OKKKKKKKKKK")
                        facialExpressionRecognition = FacialExpressionRecognition(assets, this@MainActivity, "model300.tflite", inputSize)
                    } catch (e: IOException) {
                        Log.i(TAG,"THIS IS ERROR")
                        e.printStackTrace()
                    }
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), MY_PERMISSIONS_REQUEST_CAMERA)
        }
        binding.flipCamera.setOnClickListener{
            swapCamera()
        }
        activateOpenCVCameraView()
    }

    override fun onResume() {
        super.onResume()
        if (OpenCVLoader.initDebug()) {
            //if load success
            Log.d(TAG, "Opencv initialization is done")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        } else {
            //if not loaded
            Log.d(TAG, "Opencv is not loaded. try again")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallback)
        }
    }
    private fun activateOpenCVCameraView() {
        // everything needed to start a camera preview
        mOpenCvCameraView = binding.frameSurface
        Log.i(TAG, "Activity camera called")
        if(mOpenCvCameraView != null){
            Log.d(TAG, "Camera view found!")
        }else{
            Log.d(TAG, "Camera view not found!")
        }
        mOpenCvCameraView?.setCameraPermissionGranted()
        mOpenCvCameraView?.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_ANY)
        mOpenCvCameraView?.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView?.setCvCameraViewListener(this)
        mOpenCvCameraView?.enableView()
    }

    override fun onPause() {
        super.onPause()
        mOpenCvCameraView?.disableView()
    }

    override fun onDestroy() {
        super.onDestroy()
        mOpenCvCameraView?.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        Log.i(TAG, "OonCameraViewStarted called")
        mRgba = Mat(height, width, CvType.CV_8UC4)
        mGray = Mat(height, width, CvType.CV_8UC1)
    }

    override fun onCameraViewStopped() {
        mRgba.release()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat? {
        mRgba = inputFrame.rgba() ?: mRgba
        mGray = inputFrame.gray() ?: mGray
//        val rotMat = Imgproc.getRotationMatrix2D(Point(mRgba.cols() / 2.0, mRgba.rows() / 2.0), 90.0, 1.0)
//        var rotated = Mat()
//        Imgproc.warpAffine(mRgba, rotated, rotMat, mRgba.size())
        Core.rotate(mRgba, mRgba, Core.ROTATE_90_CLOCKWISE)
        if(mCameraId == 1){
            Core.flip(mRgba,mRgba,-1)
            Core.flip(mGray,mGray,-1)
        }
        mRgba = facialExpressionRecognition.recognizeImage(mRgba)
        //mRgba = mRgba?.let { facialExpressionRecognition.recognizeImage(it) }
        return mRgba
    }

    private fun swapCamera(){
        mCameraId = mCameraId xor 1
        mOpenCvCameraView?.disableView()
        mOpenCvCameraView?.setCameraIndex(mCameraId)
        mOpenCvCameraView?.enableView()
    }
}