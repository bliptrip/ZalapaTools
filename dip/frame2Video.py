#!/usr/bin/env python
#Author: Andrew Maule
#Purpose: Generate a slide-show series from set of images listed in a config file.

import argparse
import csv
import cv2
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Generates a time-lapse video of images specified in a config file..")
    parser.add_argument('-d', '--directory', action='store', type=str, required=True, help="The parent directory to search for image files.")
    parser.add_argument('-c', '--config', action='store', type=str, required=True, help="A csv file with name of files and other parameters.")
    parser.add_argument('-o', '--output', action='store', type=str, required=True, help="The output file path of the newly created video.")
    parser.add_argument('-r', '--repeat', action='store', type=int, default=1, help="The number of times to repeat a frame in a video (b/c fps can't be below 1 for VideoWriter)")
    parser.add_argument('-f', '--fps', action='store', type=float, default=1.0, help="The number of frames per second in output video.")
    parser.add_argument('-s', '--resize', action='store', type=float, default=1.0, help="Resizing factor to apply to input images before generating output slideshow video.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

if __name__ == '__main__':
    images_dict = {}
    print(cv2.__version__)
    parsed = parse_args()
    parent_dir = parsed.directory
    config     = parsed.config
    with open(config,'r') as config_fh:
        config_csv = csv.DictReader(config_fh, delimiter=',', quotechar='"')
        image_files = [parent_dir+'/'+e['newfile'] for e in config_csv]
        img0 = cv2.imread(image_files[0])
        if( parsed.resize != 1.0 ):
            img0 = cv2.resize(img0, (0,0), fx=parsed.resize, fy=parsed.resize)
        (height,width,depth) = img0.shape
        vidwriter = cv2.VideoWriter(parsed.output, cv2.VideoWriter_fourcc('a','v','c','1'), parsed.fps, (width,height))
        for img_file in image_files:
            img = cv2.imread(img_file)
            if( parsed.resize != 1.0 ):
                img = cv2.resize(img, (0,0), fx=parsed.resize, fy=parsed.resize)
            (nheight,nwidth,ndepth) = img.shape
            if (nheight != height) or (nwidth != width):
                #Force a resize to the first image frame size, otherwise the encoder will silently drop the frame and not render (even if one pixel off)
                img = cv2.resize(img, (width,height))
            for i in range(0,parsed.repeat):
                print("Writing image file: %s: Shape: %dx%d" % (img_file,width,height))
                vidwriter.write(img)
        vidwriter.release()
