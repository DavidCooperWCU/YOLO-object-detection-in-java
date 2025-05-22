package sample;

import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.utils.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

//import com.streambase.com.gs.collections.impl.Counter;

import java.util.ArrayList;
import java.util.List;

import java.io.*;	

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

public class yolo {

	private static List<String> classNames;

	private static List<String> namesList(String filePath) {
        List<String> names = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                names.add(line.trim());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return names;
    }

	private static List<String> getOutputNames(Net net) {
		List<String> names = new ArrayList<>();

		List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
		List<String> layersNames = net.getLayerNames();

		outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));// unfold and create R-CNN layers from the
																			// loaded YOLO model//
		return names;
	}

	private static void drawPrediction(int classId, float conf, Rect box, Mat frame, double x, double y, double width, double height) {
/*
 * MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
				Rect2d[] boxesArray = rects.toArray(new Rect2d[0]);
				MatOfRect2d boxes = new MatOfRect2d(boxesArray);
				MatOfInt indices = new MatOfInt();
				Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices); // We draw the bounding boxes for
																						// objects here//

				int[] ind = indices.toArray();
				int j = 0;
				for (int i = 0; i < ind.length; ++i) {
					int idx = ind[i];
					Rect2d box = boxesArray[idx];
					Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(0, 0, 255), 2);
					// i=j;

					//System.out.println(idx);
				}
 */


		Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(0, 255, 0));

    String label = String.format("%.2f", conf);
    if (classId < classNames.size())
    {
        label = classNames.get(classId) + ": " + label;
    } else {
		label =  "UNKNOWN: " + label;
	}
	label = label + String.format(",x=%.2f,y=%.2f,w=%.2f,h=%.2f",x ,y,width,height);

    int[] baseLine = new int[1];
    Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);

    double top = Math.max(box.y, labelSize.height);
    Imgproc.rectangle(frame, new Point(box.x, top - labelSize.height),
              new Point(box.x + labelSize.width, top + baseLine[0]), Scalar.all(255), Imgproc.FILLED);
    Imgproc.putText(frame, label, new Point(box.x, top), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0,0,255));
}


	public static void main(String[] args) throws InterruptedException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		// System.load("opencv_java490.dylib"); // Load the openCV 4.0 dll //
		String modelONNX = "data/yolov8n.onnx";
		String modelWeights = "data/yolov3.weights"; // Download and load only wights for YOLO , this is obtained from
														// official YOLO site//
		String modelConfiguration = "data/yolov3.cfg";// Download and load cfg file for YOLO , can be obtained from
														// official site//
		classNames = namesList("data/coco.names");

		// String filePath = "D:\\cars.mp4"; //My video file to be analysed//
		VideoCapture cap = new VideoCapture(0);// Load video using the videocapture method//
		Mat frame = new Mat(); // define a matrix to extract and store pixel info from video//
		Mat dst = new Mat();
		// cap.read(frame);
		JFrame jframe = new JFrame("Video"); // the lines below create a frame to display the resultant video with
												// object detection and localization//
		JLabel vidpanel = new JLabel();
		jframe.setContentPane(vidpanel);
		jframe.setSize(1000, 800);
		jframe.setVisible(true);// we instantiate the frame here//

		Net net = Dnn.readNetFromONNX(modelONNX);
		//Net net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights); // OpenCV DNN supports models trained from
																			// various frameworks like Caffe and
																			// TensorFlow. It also supports various
																			// networks architectures based on YOLO//
		// Thread.sleep(5000);

		// Mat image =
		// Imgcodecs.imread("D:\\yolo-object-detection\\yolo-object-detection\\images\\soccer.jpg");
		//Size sz = new Size(288, 288);
		Size sz = new Size(640, 640);

		List<Mat> result = new ArrayList<>();
		List<String> outBlobNames = getOutputNames(net);
		Scalar scaleFactor = new Scalar(0.00392);
			scaleFactor = new Scalar(1/255.0);
		Image2BlobParams imgParams = new Image2BlobParams(scaleFactor,sz,new Scalar(0),true);

		while (cap.read(frame)) {


			
			Mat blob = Dnn.blobFromImageWithParams(frame,imgParams); // We feed one frame of video
																							// into the network at a
																							// time, we have to convert
																							// the image to a blob. A
																							// blob is a pre-processed
																							// image that serves as the
																							// input.//
			net.setInput(blob);

			net.forward(result, outBlobNames); // Feed forward the model to get output //
			//System.out.println("results: " + result.size());
			//outBlobNames.forEach(System.out::println);
			//result.forEach(System.out::println);

			float confThreshold = 0.25f; // Insert thresholding beyond which the model will detect objects//
			List<Integer> clsIds = new ArrayList<>();
			List<Float> confs = new ArrayList<>();
			List<Rect2d> rects = new ArrayList<>();
			List<double[]> sizes = new ArrayList<>();
			for (int i = 0; i < result.size(); ++i) {
				// each row is a candidate detection, the 1st 4 numbers are
				// [center_x, center_y, width, height], followed by (N-4) class probabilities
				Mat levelTrans = result.get(i);
				Mat level = new Mat();
				//System.out.println("levelTrans size:" + levelTrans.size());
				
				Core.transposeND(levelTrans, new MatOfInt(0,2,1), level);
				//System.out.println("levelTrans size:" + levelTrans.size());
				//System.out.println("level size:" + level.size());
				
				level = level.reshape(1,level.size(1));
				//System.out.println("level size:" + level.size());
				for (int j = 0; j < level.rows(); ++j) {
					Mat row = level.row(j);
					Mat scores = row.colRange(4, level.cols());
					Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
					float confidence = (float) mm.maxVal;
					Point classIdPoint = mm.maxLoc;
					if (confidence > confThreshold) {
						double centerX = (row.get(0, 0)[0] ); // scaling for drawing the bounding boxes//
						double centerY = (row.get(0, 1)[0] );
						double width = (row.get(0, 2)[0] );
						double height = (row.get(0, 3)[0] );
						double left = centerX-0.5*width;
						double top = centerY-0.5*height;
						double right = centerX+0.5*width;
						double bottom = centerY-0.5*height;

						clsIds.add((int) classIdPoint.x);
						confs.add((float) confidence);
						
						//System.out.println(classNames.get((int)classIdPoint.x)+","+confidence + ","+centerX+","+centerY
						//   + "," + width + "," + height);
						rects.add(new Rect2d(left, top, width, height));
						double[] temp = {centerX,centerY,width,height};
						sizes.add(temp);
					}
				}
			}
			float nmsThresh = 0.7f;
			//System.out.println("confs: " + confs);
			if (confs.size() > 0) {

				MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
				Rect2d[] boxesArray = rects.toArray(new Rect2d[0]);
				MatOfRect2d boxes = new MatOfRect2d(boxesArray);
				MatOfInt indices = new MatOfInt();
				Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices); // We draw the bounding boxes for
																						// objects here//

			    
				int[] ind = indices.toArray();
				int j = 0;
				for (int i = 0; i < ind.length; ++i) {
					int idx = ind[i];
					Rect2d box = boxesArray[idx];
					Rect boxRect = new Rect((int)box.x,(int)box.y,(int)box.width,(int)box.height);
					double[] temp = sizes.get(idx);
          			Rect boxCorrected = imgParams.blobRectToImageRect(boxRect, frame.size());
					//Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(0, 0, 255), 2);
					// i=j;
					drawPrediction(clsIds.get(idx),confs.get(idx),boxCorrected,frame,temp[0],temp[1],temp[2],temp[3]);
					//System.out.println(idx);
				}
			}
			// Imgcodecs.imwrite("D://out.png", image);
			// System.out.println("Image Loaded");
			ImageIcon image = new ImageIcon(Mat2bufferedImage(frame)); // setting the results into a frame and
																		// initializing it //
			vidpanel.setIcon(image);
			vidpanel.repaint();
			// System.out.println(j);
			// System.out.println("Done");

		}

	}

	// }
	private static BufferedImage Mat2bufferedImage(Mat image) { // The class described here takes in matrix and renders
																// the video to the frame //
		MatOfByte bytemat = new MatOfByte();
		Imgcodecs.imencode(".jpg", image, bytemat);
		byte[] bytes = bytemat.toArray();
		InputStream in = new ByteArrayInputStream(bytes);
		BufferedImage img = null;
		try {
			img = ImageIO.read(in);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return img;
	}
}
