package com.cooplogic.img;
import org.opencv.core.Rect;
/**
 * Author: David G. Cooper, Ph.D
 * Date: May 22, 2025
 * Purpose: A library class that holds a single yolo detection.
 */
public class Detection {
    final int classId;
    final float conf;
    final Rect box;
    
    public Detection(int classId, float conf, Rect box) {
        this.classId = classId;
        this.conf = conf;
        this.box = box;
    }
}
