#!/usr/bin/env python
# Author: Andrew Maule
#
# Purpose: Take a JSON file exported from the Hitachi Automotive and Industry Lab's Semantic Segmentation Editor
#           of an image with an XRite Color Card and generate an output CSV file containing the average value for 
#           a given contour label and it's mapping to the color card expected value.  This is used as input for
#           the whitebalance script.
#
# Hitachi Automotive and Industry Lab Github Page: https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor
#
#
# NOTE: To get average of a contour, I took example from stackexchange:
# https://stackoverflow.com/questions/17936510/how-to-find-average-intensity-of-opencv-contour-in-realtime
#
# Making slight modifications to Method 3 increases execution time slightly, but gains the benefit of giving correct results regardless of contour positions. Each contour 
# is individually labeled, and the mean is calculated using only the mask for that contour.
#
#for (size_t i = 0; i < cont.size(); ++i)
#{
#        cv::drawContours(labels, cont, i, cv::Scalar(i), CV_FILLED);
#        cv::Rect roi = cv::boundingRect(cont[i]);
#        cv::Scalar mean = cv::mean(image(roi), labels(roi) == i);
#        cont_avgs[i] = mean[0];
#}
import argparse
import csv
import cv2 as cv
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Using an input JSON file and an image file containing an XRite Color Card, calculate the average intensity values of each labeled contour.  Output to a CSV file.")
    parser.add_argument('-j', '--json', action='store', required=True, help="JSON file output from the Semantic Segmentation Editor software.")
    parser.add_argument('-i', '--input', action='store', required=True, help="Input image.")
    parser.add_argument('-c', '--config', action='store', required=True, help="Input configuration file containing expected RGB/CIE Lab color values.")
    parser.add_argument('-o', '--output', action='store', required=True, help="Output CSV file.")
    parser.add_argument('-d', '--debug', action='store_true', help="Show debug plots for mapping input to output values.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

class CCColormapGenerator():
    def __init__(self, debug=False):
        self.img      = None
        self.sse_json = None
        self.conts    = None
        self.debug    = debug
        return

    def load_json(self,json_path):
        #Get a file handle for the JSON object
        sse_json_fh   = open(json_path,'r')
        #Load the JSON file into an accessible object
        self.sse_json = json.load(sse_json_fh)
        sse_json_fh.close()

    def load_img(self,img_path):
        #Open the image file
        self.img     = cv.imread(img_path)
        self.img_lab = cv.cvtColor(self.img, cv.COLOR_BGR2LAB)

    def contours(self):
        'Returns the contours in the input JSON object as a numpy array of [x,y] points.'
        #Contours are found in the JSON structure under the 'objects' lookup key
        self.objects = sse_objects = self.sse_json['objects']
        sse_conts    = [x['polygon'] for x in sse_objects]
        #Convert the contours to a contour format recognized by opencv
        #Should be a list of numpy arrays, dtype=int32, which are the x,y coordinate pairs that define the polygon contours
        conts = []
        for sse_cont in sse_conts:
            conta = np.zeros(shape=(len(sse_cont),2),dtype=np.int32)
            for i,cont in enumerate(sse_cont):
                conta[i,:] = (cont['x'],cont['y'])
            conts.append(conta)
        self.conts = conts
        return(conts)

    def means(self):
        'Calculates and returns the mean intensity levels for each b,g,r channel of the input image.'
        #Generate a single-channel labels matrix that mirrors the width and height of the loaded image.  Each element
        #contains an index corresponding to the contour index + 1.
        conts = self.conts
        labels = np.zeros(shape=(self.img.shape[0],self.img.shape[1]),dtype=np.int32)
        cont_means = np.zeros(shape=(len(conts),self.img.shape[2]),dtype=np.float)
        cont_means_lab = np.zeros(shape=(len(conts),self.img_lab.shape[2]),dtype=np.float)
        for i in range(0,len(conts)):
            labels = cv.drawContours( labels, contours=conts, contourIdx=i, color=i+1, thickness=-1 )
            roi    = cv.boundingRect(conts[i])
            x_0,y_0,width,height = roi
            x_1=x_0+width
            y_1=y_0+height
            #Calculate the BGR means
            img_roi = self.img[y_0:y_1,x_0:x_1,:]
            mask    = labels[y_0:y_1,x_0:x_1] == (i+1)
            mean   = cv.mean(img_roi,mask.astype(np.uint8))
            cont_means[i,:] = mean[0:3]
            #Calculate the Lab means
            img_lab_roi = self.img_lab[y_0:y_1,x_0:x_1,:]
            mean_lab   = cv.mean(img_lab_roi,mask.astype(np.uint8))
            cont_means_lab[i,:] = mean_lab[0:3]
        self.cont_means = cont_means
        self.cont_means_lab = cont_means_lab
        return((cont_means,cont_means_lab))

    def write(self, config_path, csv_path):
        'Write the output csv file given the average values previous computed and the json labels.'
        sse_idxs        = [x['classIndex']-1 for x in self.objects]
        sse_labels      = [x['label'] for x in self.objects]
        #Convert to canonical Lab CIE
        cont_means_lab  = self.cont_means_lab
        cont_means_lab  = cont_means_lab - [0,128.0,128.0]
        cont_means_lab  = cont_means_lab * [100/255,1,1]
        with open(config_path, 'r') as config_csv:
            map_config_csv = csv.DictReader(config_csv)
            #Copy the fieldnames over from the input config
            fieldnames = map_config_csv.fieldnames.copy()
            #Append new fieldnames for the output file
            fieldnames.extend(['b_in', 'g_in', 'r_in', 'L_cie_in', 'a_cie_in', 'b_cie_in'])
            with open(csv_path,'w') as out_csv:
                map_out_csv = csv.DictWriter(out_csv, fieldnames=fieldnames)
                map_out_csv.writeheader()
                for row in map_config_csv:
                    label = row['label']
                    if( label in sse_labels ):
                        label_idx = sse_labels.index(label)
                        if(self.debug):
                                self.img = cv.drawContours(self.img, contours=self.conts, contourIdx=label_idx, color=(int(row['b']),int(row['g']),int(row['r'])), thickness=-1)
                        row_keys = [key for key in row.keys()]
                        row_keys.extend(['b_in', 'g_in', 'r_in', 'L_cie_in', 'a_cie_in', 'b_cie_in'])
                        row_vals = [val for val in row.values()]
                        row_vals.extend(self.cont_means[label_idx].astype(np.uint8))
                        #3 sig digits
                        cont_means_lab_red = ["%.3f" % (x) for x in cont_means_lab[label_idx]]
                        row_vals.extend(cont_means_lab_red)
                        row_dict = dict(zip(row_keys,row_vals))
                        map_out_csv.writerow(row_dict)
                if(self.debug):
                        cv.imshow('overlay', self.img)
                        ch = cv.waitKey()
        return;

        
if __name__ == '__main__':
    parsed  = parse_args()
    cc_colormap_gen = CCColormapGenerator(debug=parsed.debug)
    cc_colormap_gen.load_json(parsed.json)
    cc_colormap_gen.load_img(parsed.input)
    cc_colormap_gen.contours()
    cc_colormap_gen.means()
    cc_colormap_gen.write(parsed.config,parsed.output)
