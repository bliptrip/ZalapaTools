#!/usr/bin/env python
#Author: Andrew Maule
#Modified from https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
import argparse
import cv2
import numpy as np
import pickle
import sys

gdebug=False

def debug(str):
    if(gdebug):
        print(str)

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Undistorts image using camera matrix and distortion parameters stored in a corresponding pickle file.")
    parser.add_argument('-i', '--input', action='store', required=True, help="Input image to undistort.")
    parser.add_argument('-o', '--output', action='store', required=True, help="Output image with distortions removed.")
    parser.add_argument('-p', '--params', action='store', required=True, help="Location of pickle file containing the camera lens matrix and distortion parameters.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose debugging.")
    parser.add_argument('-c', '--crop', action='store_true', help="Auto-crop image")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

if __name__ == '__main__':
    parsed           = parse_args()
    gdebug = parsed.verbose

    # termination criteria
    with open(parsed.params, 'rb') as params_fh:
        params = pickle.load(params_fh)
        params_fh.close()
        debug("Descriptor string for parameter file: " + params['descriptor'])
        img = cv2.imread(parsed.input)
        assert img is not None, "Failed to open input image %s" % (parsed.input)
        mtx = params['matrix']
        dist = params['distortion']
        h,w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
        dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
        # crop the image
        if( parsed.crop ):
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]
        cv2.imwrite(parsed.output,dst)
