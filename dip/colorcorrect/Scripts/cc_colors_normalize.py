#!/usr/bin/env python
#Author: Andrew Maule
import argparse
import csv
import cv2 as cv
import math
import numpy as np
import pickle
import sys

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Normalize colors of an input image using a set of pre-generated models that map an input color set to an expected output color set (using colorcards).")
    parser.add_argument('-m', '--models', action='store', required=True, help="A python pickle file containing the pre-generated models.")
    parser.add_argument('-i', '--input', action='store', required=True, help="Input image.")
    parser.add_argument('-o', '--output', action='store', required=True, help="Output image.")
    parser.add_argument('-d', '--debug', action='store_true', help="Show debug plots for mapping input to output values.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

class CCColorsNormalize():
    def __init__(self,models_filename):
        with open(models_filename,'rb') as models_fh:
            models = pickle.load(models_fh)
            self.modelL = models['L']
            self.modela = models['a']
            self.modelb = models['b']
            models_fh.close()
            return

    def normalize(self,img_input,img_output,mdebug=False):
        #Now take the image input and map to output using the appropriate interpolation function
        img = cv.imread(img_input)
        img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        L_channel, a_channel, b_channel = cv.split(img_lab)
        #Initialize the new channel values to the original, in case certain models aren't defined
        L_channel_new = L_channel
        a_channel_new = a_channel
        b_channel_new = b_channel
        #Store the shapes of the original channels
        L_channel_shape = L_channel.shape
        a_channel_shape = a_channel.shape
        b_channel_shape = b_channel.shape
        if(self.modelL):
            if( mdebug ):
                print("Predicting L_channel...")
            L_channel_new = self.modelL.predict(np.resize(L_channel,(np.size(L_channel),1)))
            if( mdebug ):
                print("Finished predicting L_channel...")
            L_channel_new = np.resize(L_channel_new,L_channel_shape)
            L_channel_new = np.array(L_channel_new, dtype=np.uint8)
        if(self.modela):
            if( mdebug ):
                print("Predicting a_channel...")
            a_channel_new = self.modela.predict(np.resize(a_channel,(np.size(a_channel),1)))
            if( mdebug ):
                print("Finished predicting a_channel...")
            a_channel_new = np.resize(a_channel_new,a_channel_shape)
            a_channel_new = np.array(a_channel_new, dtype=np.uint8)
        if(self.modela):
            if( mdebug ):
                print("Predicting b_channel...")
            b_channel_new = self.modelb.predict(np.resize(b_channel,(np.size(b_channel),1)))
            if( mdebug ):
                print("Finished predicting b_channel...")
            b_channel_new = np.resize(b_channel_new,b_channel_shape)
            b_channel_new = np.array(b_channel_new, dtype=np.uint8)
        #Generate the newly color normalized image
        img_new = cv.cvtColor(cv.merge((L_channel_new, a_channel_new, b_channel_new)), cv.COLOR_LAB2BGR)
        cv.imwrite(img_output, img_new)
        return

if __name__ == '__main__':
    parsed = parse_args()
    cc_colors_norm = CCColorsNormalize(models_filename=parsed.models)
    cc_colors_norm.normalize(img_input=parsed.input,img_output=parsed.output,mdebug=parsed.debug)
