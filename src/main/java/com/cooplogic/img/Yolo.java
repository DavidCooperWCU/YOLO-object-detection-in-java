package com.cooplogic.img;

import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.*;


import java.awt.image.BufferedImage;

import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import java.io.*;

/**
 * Author: David G. Cooper, Ph.D
 * Date: May 22, 2025
 * Purpose: A library class that does yolo detection on a single frame.
 */
public class Yolo {

    public static List<String> namesList(String filePath) {
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

    private List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();

        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));// unfold and create R-CNN layers from the
                                                                          // loaded YOLO model//
        return names;
    }

    public static BufferedImage Mat2bufferedImage(Mat image) { // The class described here takes in matrix and renders
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

    private Net net;
    private List<String> classNames;
    private Image2BlobParams imgParams;
    private List<Mat> result;
    private List<String> outBlobNames;

    public Yolo(String modelONNX, String classNamePath, Size sz, Scalar scaleFactor) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
        
        classNames = namesList("data/coco.names");
        net = Dnn.readNetFromONNX(modelONNX);
        imgParams = new Image2BlobParams(scaleFactor, sz, new Scalar(0), true);
        outBlobNames = getOutputNames(net);
        result = new ArrayList<>();
    }

    public List<Detection> lookOnce(Mat frame) {
        float confThreshold = 0.25f;
        float nmsThresh = 0.7f;
        return lookOnce(frame, confThreshold, nmsThresh);
    }

    public List<Detection> lookOnce(Mat frame, float confThreshold, float nmsThresh) {
        List<Detection> detections = new ArrayList<>();
        Mat blob = Dnn.blobFromImageWithParams(frame, imgParams); // We feed one frame of video
                                                                  // into the network at a
                                                                  // time, we have to convert
                                                                  // the image to a blob. A
                                                                  // blob is a pre-processed
                                                                  // image that serves as the
                                                                  // input.//
        net.setInput(blob);

        net.forward(result, outBlobNames); // Feed forward the model to get output //

        List<Integer> clsIds = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        List<Rect2d> rects = new ArrayList<>();
        List<double[]> sizes = new ArrayList<>();
        for (int i = 0; i < result.size(); ++i) {
            // each row is a candidate detection, the 1st 4 numbers are
            // [center_x, center_y, width, height], followed by (N-4) class probabilities
            Mat levelTrans = result.get(i);
            Mat level = new Mat();
            
            // YOLO v8 transformations
            Core.transposeND(levelTrans, new MatOfInt(0, 2, 1), level);
            level = level.reshape(1, level.size(1));
            
            for (int j = 0; j < level.rows(); ++j) {
                Mat row = level.row(j);
                Mat scores = row.colRange(4, level.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float) mm.maxVal;
                Point classIdPoint = mm.maxLoc;
                if (confidence > confThreshold) {
                    double centerX = (row.get(0, 0)[0]); 
                    double centerY = (row.get(0, 1)[0]);
                    double width = (row.get(0, 2)[0]);
                    double height = (row.get(0, 3)[0]);
                    double left = centerX - 0.5 * width;
                    double top = centerY - 0.5 * height;

                    clsIds.add((int) classIdPoint.x);
                    confs.add((float) confidence);

                    rects.add(new Rect2d(left, top, width, height));
                }
            }
        }

        if (confs.size() > 0) {

            MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
            Rect2d[] boxesArray = rects.toArray(new Rect2d[0]);
            MatOfRect2d boxes = new MatOfRect2d(boxesArray);
            MatOfInt indices = new MatOfInt();
            
            //select which boxes to use 
            Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices); 
            
            int[] ind = indices.toArray();
            for (int idx : ind) {
                
                Rect2d box = boxesArray[idx];
                Rect boxRect = new Rect((int) box.x, (int) box.y, (int) box.width, (int) box.height);
                // resize the boxes for the frame size.
                Rect boxCorrected = imgParams.blobRectToImageRect(boxRect, frame.size());
                // add detection to list of detections to return
                detections.add(new Detection(clsIds.get(idx), confs.get(idx), boxCorrected));

            }
        }
        return detections;
    }

    public void drawPrediction(Detection detection, Mat frame) {
        

		Imgproc.rectangle(frame, detection.box.tl(), detection.box.br(), new Scalar(0, 255, 0));

    String label = String.format("%.2f", detection.conf);
    if (detection.classId < classNames.size())
    {
        label = classNames.get(detection.classId) + ": " + label;
    } else {
		label =  "UNKNOWN: " + label;
	}
	
    int[] baseLine = new int[1];
    Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 1.5, 3, baseLine);

    double top = Math.max(detection.box.y, labelSize.height);
    Imgproc.rectangle(frame, new Point(detection.box.x, top - labelSize.height),
              new Point(detection.box.x + labelSize.width, top + baseLine[0]), Scalar.all(255), Imgproc.FILLED);
    Imgproc.putText(frame, label, new Point(detection.box.x, top), Imgproc.FONT_HERSHEY_SIMPLEX, 1.5, new Scalar(0,0,255),3);
}
}
