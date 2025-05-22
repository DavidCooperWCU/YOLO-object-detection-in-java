package com.cooplogic.app;

import com.cooplogic.img.*;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import java.util.List;

import org.opencv.core.Size;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.videoio.VideoCapture;


public class VideoYolo {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
        VideoCapture cap = new VideoCapture(0);// Load video using the videocapture method//
		Mat frame = new Mat(); // define a matrix to extract and store pixel info from video//
		JFrame jframe = new JFrame("Video"); // the lines below create a frame to display the resultant video with
												// object detection and localization//
		JLabel vidpanel = new JLabel();
		jframe.setContentPane(vidpanel);
		jframe.setSize(1000, 800);
		jframe.setVisible(true);// we instantiate the frame here//

        Size sz = new Size(640, 640);

        Scalar scaleFactor = new Scalar(0.00392);
			scaleFactor = new Scalar(1/255.0);
		
        Yolo yolo = new Yolo("data/yolov8n.onnx","data/coco.names",sz,scaleFactor);        

        while (cap.read(frame)) {
            List<Detection> dets = yolo.lookOnce(frame);
            for(Detection d : dets) {
                yolo.drawPrediction(d, frame);
            }
            ImageIcon image = new ImageIcon(yolo.Mat2bufferedImage(frame)); // setting the results into a frame and
																		// initializing it //
			vidpanel.setIcon(image);
			vidpanel.repaint();
			
        }
    }
}
