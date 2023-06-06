package com.example.liveface

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class FacialExpressionRecognition(assetManager: AssetManager, context: Context, modelPath: String, inputSize: Int) {
    private var interpreter: Interpreter
    lateinit var emo:String
    private val INPUT_SIZE: Int
    private var height = 0
    private var width = 0
    private var gpuDelegate: Delegate? = null

    // now define cascadeClassifier for face detection
    private lateinit var cascadeClassifier: CascadeClassifier

    init {
        INPUT_SIZE = inputSize
        // set GPU for the interpreter
        val options = Interpreter.Options()
        gpuDelegate = GpuDelegate()
        // add gpuDelegate to option
        options.addDelegate(gpuDelegate)
        options.setNumThreads(4)

        interpreter = Interpreter(loadModelFile(assetManager, modelPath), options)
        Log.d("facial_Expression", "Model is loaded")

        // now we will load haarcascade classifier
        try {
            // define input stream to read classifier
            val input:InputStream = context.resources.openRawResource(R.raw.haarcascade_frontalface_alt)
            Log.i("MainACTIVITY", "Resource opened successfully: ${input.available()} bytes available")
            // create a folder
            val cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE)
            // now create a new file in that folder
            val mCascadeFile = File(cascadeDir, "haarcascade_frontalface_alt")
            // now define output stream to transfer data to file we created
            val os = FileOutputStream(mCascadeFile)
            // now create buffer to store byte
            val buffer = ByteArray(4096)
            var byteRead: Int

            while (input.read(buffer).also { byteRead = it } != -1) {
                // writing on mCascade file
                os.write(buffer, 0, byteRead)
            }
            // close input and output stream
            input.close()
            os.close()
           // System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
            cascadeClassifier = CascadeClassifier(mCascadeFile.absolutePath)
            // if cascade file is loaded print
            Log.d("facial_Expression", "Classifier is loaded")

        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
    ////////////
    fun recognizeImage(matImage: Mat): Mat {
        Log.i("facial_Expression","Now Recognize Image")
        //Core.flip(matImage.t(), matImage,1) // Rotate matImage by 90 degrees
        val grayscaleImage = Mat()
        Imgproc.cvtColor(matImage, grayscaleImage, Imgproc.COLOR_RGBA2GRAY)
        height = grayscaleImage.height()
        width = grayscaleImage.width()
        // Define minimum height of face in original image, below this size no face in original image will show
        val absoluteFaceSize = (height * 0.1).toInt()
        // Create MatOfRect to store face
        val faces = MatOfRect()
        // Check if cascadeClassifier is loaded or not
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(
                grayscaleImage,
                faces,
                1.1,
                2,
                2,
                Size(absoluteFaceSize.toDouble(), absoluteFaceSize.toDouble()),
                Size()
            )
        }
        // Convert it to array
        val faceArray = faces.toArray()
        // Loop through each face
        for (i in faceArray.indices) {

            Imgproc.rectangle(
                matImage,
                faceArray[i].tl(),
                faceArray[i].br(),
                Scalar(0.0, 255.0, 0.0, 255.0),
                2
            )
            // Crop face from original frame and grayscaleImage
            // Starting x coordinate, Starting y coordinate
            val roi = Rect(
                faceArray[i].tl().x.toInt(),
                faceArray[i].tl().y.toInt(),
                (faceArray[i].br().x - faceArray[i].tl().x).toInt(),
                (faceArray[i].br().y - faceArray[i].tl().y).toInt()
            )
            val croppedRgba = Mat(matImage, roi)
            var bitmap: Bitmap? = null
            bitmap =
                Bitmap.createBitmap(croppedRgba.cols(), croppedRgba.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(croppedRgba, bitmap)
            // Resize bitmap to (48,48)
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 48, 48, false)
            // Convert scaledBitmap to byteBuffer
            val byteBuffer = convertBitmapToByteBuffer(scaledBitmap)
            // Create an object to hold output
            val emotion = Array(1) { FloatArray(1) }
            // Predict with byteBuffer as an input and emotion as an output
            interpreter.run(byteBuffer, emotion)
            // If emotion is recognized, print value of it
            // Define float value of emotion
            val emotion_v = emotion[0][0]
            Log.d("facial_expression", "Output: $emotion_v")
            // Create a function that returns text emotion
            val emotion_s = get_emotion_text(emotion_v)

            Imgproc.putText(
                matImage,
                "${emotion_s} (${emotion_v})",
                Point(
                    (faceArray[i].tl().x.toInt() + 10).toDouble(),
                    (faceArray[i].tl().y.toInt() + 20).toDouble()
                ),
                1,
                1.5,
                Scalar(0.0, 0.0, 255.0, 150.0),
                2
            )
        }

        return matImage;
    }
    private fun get_emotion_text(emotion_v: Float): String {
        var value = ""
    if (emotion_v >= 0 && emotion_v < 0.5) {
        value = "Surprise"
        emo = value
    } else if (emotion_v >= 0.5 && emotion_v < 1.0) {
        value = "Fear"
        emo = value
    } else if (emotion_v >= 1.0 && emotion_v < 2.0) {
        value = "Angry"
        emo = value
    } else if (emotion_v >= 2.0 && emotion_v < 2.5) {
        value = "Neutral"
        emo = value
    } else if (emotion_v >= 2.5 && emotion_v < 3.5) {
        value = "Sad"
        emo = value
    } else if (emotion_v >= 4.5 && emotion_v < 5.5) {
        value = "Happy"
        emo = value
    } else {
        value = "Happy"
        emo = value
    }
    return value
}
    fun convertBitmapToByteBuffer(scaledBitmap: Bitmap): ByteBuffer {
                val byteBuffer: ByteBuffer
                val size_image = INPUT_SIZE //48

                byteBuffer = ByteBuffer.allocateDirect(4 * 1 * size_image * size_image * 3)
                // 4 is multiplied for float input
                // 3 is multiplied for rgb
                byteBuffer.order(ByteOrder.nativeOrder())
                val intValues = IntArray(size_image * size_image)
                scaledBitmap.getPixels(
                    intValues,
                    0,
                    scaledBitmap.width,
                    0,
                    0,
                    scaledBitmap.width,
                    scaledBitmap.height
                )
                var pixel = 0
                for (i in 0 until size_image) {
                    for (j in 0 until size_image) {
                        val `val` = intValues[pixel++]
                        // now put float value to bytebuffer
                        // scale image to convert image from 0-255 to 0-1
                        byteBuffer.putFloat(((`val` shr 16) and 0xFF) / 255.0f)
                        byteBuffer.putFloat(((`val` shr 8) and 0xFF) / 255.0f)
                        byteBuffer.putFloat((`val` and 0xFF) / 255.0f)
                    }
                }
                return byteBuffer
                // check one more time it is important else you will get error
            }

            @Throws(IOException::class)
            fun loadModelFile(
                assetManager: AssetManager,
                modelPath: String
            ): MappedByteBuffer {
                // this will give description of file
                val assetFileDescriptor: AssetFileDescriptor = assetManager.openFd(modelPath)
                // create a inputsteam to read file
                val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
                val fileChannel = inputStream.channel

                val startOffset = assetFileDescriptor.startOffset
                val declaredLength = assetFileDescriptor.declaredLength
                return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            }
        }