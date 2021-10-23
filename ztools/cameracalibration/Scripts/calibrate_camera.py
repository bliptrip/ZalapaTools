#!/usr/bin/env python
#Author: Andrew Maule
#Modified from https://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html
import argparse
import cv2
import glob
import numpy as np
import pickle
import sys

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Generates a camera matrix and a distortion matrix for a series of checkerboard pictures taken with a camera, in order to determine camera/lens-specific distortion parameters.")
    parser.add_argument('-d', '--directory', action='store', default="./", help="Directory to search for checkerboard patterns taken with a given camera.")
    parser.add_argument('-e', '--extension', action='store', default="JPG", help="Image extension to search for when looking for checkerboard images in search directory.")
    parser.add_argument('-o', '--output_directory', action='store', default='./', help="Directory to store the output pickle file containing the camera matrix and lens distortion parameter matrix.")
    parser.add_argument('-s', '--size', action='store', type=float, default=100.0, help="Size, in mm, of each square in the checkerboard pattern.")
    parser.add_argument('-x', '--num_x', action='store', type=int, default=6, help="Number of inner corners in horizontal (x) direction of checkerboard pattern to search against, or number of columns of circles.")
    parser.add_argument('-y', '--num_y', action='store', type=int, default=8, help="Number of inner corners in vertical (y) direction of checkerboard pattern to search against, or number of rows of circles.")
    parser.add_argument('-t', '--type', action='store', default="squares", choices=['squares','circles'], help="Type of pattern to search for in camera.")
    parser.add_argument('--draw', action='store_true', help='Draw the software points and lines that connect the corresponding corners of the checkerboard tiles.')
    parser.add_argument('--draw_resize_w', action='store', default=864, help='Resize image window to specified width.')
    parser.add_argument('--draw_resize_h', action='store', default=1296, help='Resize image window to specified height.')
    parser.add_argument('--descriptor', action='store', required=True, help='Descriptor of the camera lens and focal length that parameters are generated against.')
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)


if __name__ == '__main__':
    parsed           = parse_args()
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, int(parsed.size), 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros(((parsed.num_x*parsed.num_y),3), np.float32)
    objp[:,:2] = np.mgrid[0:parsed.num_y,0:parsed.num_x].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(parsed.directory+'/*.'+parsed.extension)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if( parsed.draw ):
            cv2.namedWindow('Input Image',cv2.WINDOW_NORMAL)
            cv2.imshow('Input Image',gray)
            cv2.resizeWindow('Input Image', parsed.draw_resize_w, parsed.draw_resize_h)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

        # Find the chess board corners
        if(parsed.type == 'squares'):
            ret, corners = cv2.findChessboardCorners(gray, (parsed.num_y,parsed.num_x), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners)

                if( parsed.draw ):
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (parsed.num_y,parsed.num_x), corners, ret)
                    cv2.namedWindow('img',cv2.WINDOW_NORMAL)
                    cv2.imshow('img',img)
                    cv2.resizeWindow('img', parsed.draw_resize_w, parsed.draw_resize_h)
                    cv2.waitKey(100)
        elif(parsed.type == 'circles'):
            # Setup SimpleBlobDetector parameters -- without this, findCirclesGrid() doesn't find the circles!
            params = cv2.SimpleBlobDetector_Params()
            # Filter by Threshold
            #params.minThreshold = 200
            #params.maxThreshold = 255

            # Filter by Area.
            params.filterByArea = True
            #Max width is about 212 pixels in test pictures.  pi*(212/2)^2 = 35281 = 3.52 x 10^4
            params.maxArea = 10e6

            # Filter by circularity
            #params.filterByCircularity = True
            #params.minCircularity = 0.7
            #params.maxCircularity = 1.0

            # Create a detector with the parameters
            ver = (cv2.__version__).split('.')
            if int(ver[0]) < 3 :
                detector = cv2.SimpleBlobDetector(params)
            else: 
                detector = cv2.SimpleBlobDetector_create(params)
            if( parsed.draw ):
                keypoints = detector.detect(gray)
                im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.namedWindow('Keypoints',cv2.WINDOW_NORMAL)
                cv2.imshow("Keypoints", im_with_keypoints)
                cv2.resizeWindow('Keypoints', parsed.draw_resize_w, parsed.draw_resize_h)
                cv2.waitKey(500)
                cv2.destroyAllWindows()
            ret, corners = cv2.findCirclesGrid(gray, (parsed.num_y,parsed.num_x), flags=cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector=detector)
            #ret, corners = cv2.findCirclesGrid(gray, (parsed.num_y,parsed.num_x), flags=cv2.CALIB_CB_SYMMETRIC_GRID)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                #cv2.cornerSubPix(gray,corners,(parsed.num_y,parsed.num_x),(-1,-1),criteria)
                imgpoints.append(corners)

                if( parsed.draw ):
                    # Draw and display the corner
                    cv2.drawChessboardCorners(img, (parsed.num_y,parsed.num_x), corners, ret)
                    cv2.namedWindow('img',cv2.WINDOW_NORMAL)
                    cv2.imshow('img',img)
                    cv2.resizeWindow('img', parsed.draw_resize_w, parsed.draw_resize_h)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()

    if( parsed.draw ):
        cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    print("Total error: %.3f" % (mean_error/len(objpoints)))

    with open(parsed.output_directory+'/calibrate_params.pickle', 'wb') as pickle_fh:
        pickle.dump({"descriptor": parsed.descriptor, "matrix": mtx, "distortion": dist}, pickle_fh)
        pickle_fh.close()
