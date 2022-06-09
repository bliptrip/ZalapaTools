#!/usr/bin/env python3
# Author: Andrew Maule
# Objective: Using binary template images with single foreground 'blob' object, typically with all objects
#            normalized to the same area and centroid centered in middle of image, 
#            generate a chimera binary 'image' as a result of merging a set template images.
#
# Use Case: I have cranberry phenotypic data using categorized 'shapes' based on provided templates.
#           I then generate a chimeric merge of a set of these different categories (using template
#           images in lieu of a categorized berry), and from this 'chimeric' image I can subsequently
#           calculate useful shape parameters.
#
# Output: Binary image of chimera if specified, and a numpy-array of contours for the outline of this
#           chimera object.
#         

# import the necessary packages
import argparse
import cv2
from glob import *
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import sys

stdout_default='-'

# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility for generating a chain code representation of berry shapes.")
    parser.add_argument('-p', '--path', action='store', type=str, required=True, help="Input folder path containing normalized berry template representations of different berry shapes.")
    parser.add_argument('-m', '--map', action='store', default="{'round': '*round_binary.png', 'oblong': '*oblong_binary.png', 'oval': '*oval_binary.png', 'pyriform': '*pyriform_binary.png', 'spindle': '*spindle_binary.png'}", help="Dictionary mapping shape categories to template binary image file (glob-patterns allowed).")
    parser.add_argument('-o', '--output', action='store', default=stdout_default, help="Output file to write chimera contours to. ('{}' for stdout)".format(stdout_default))
    parser.add_argument('--image', action='store', help="Name of output binary image, if desired.  If unspecified, no image is saved (just the contours)")
    parser.add_argument('shapes', metavar='shapes', type=str, nargs='+', help="Sequence of input shape categories to merge into chimera.  Valid values: {}")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

class Chimera:
    def __init__(self):
        self._shapes = None #List of shapes
        self._dtransforms = None #Calculated distance transforms
        self._composite_raw = None #composite of differential distance transforms
        self._composite  = None #Thresholded composite/blended image
        self._contours   = None #Berry Blob Contours

    @property
    def shapes(self):
        return(self._shapes)

    @property
    def dtransforms(self):
        return(self._dtransforms)

    @dtransforms.setter
    def dtransforms(self, val):
        self._dtransforms = val

    @property
    def composite(self):
        return(self._composite)

    @property
    def composite_raw(self):
        return(self._composite_raw)

    @property
    def contours(self):
        return(self._contours)

    def loadMap(self, template_path, template_name_map):
        dtransforms = {}
        smap        = eval(template_name_map)
        for k in smap:
            names           = glob("{}/{}".format(template_path,smap[k]))
            templates    = cv2.imread(names[0], cv2.IMREAD_GRAYSCALE)
            dtransforms[k]  = cv2.distanceTransform(templates, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE) - cv2.distanceTransform(np.bitwise_not(templates), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE)
        self._dtransforms = dtransforms

    def compose(self, *args):
        self._shapes = shapes = list(*args)
        dsums = self.dtransforms[shapes[0]].copy()
        for s in shapes[1:]:
            dsums += self.dtransforms[s]
        self._composite_raw = dsums
        dsums = ((dsums / len(shapes)) > 0).astype(dtype='uint8')
        dsums[dsums > 0] = 255
        self._composite = dsums
        self._contours = measure.find_contours(dsums)[0]

if __name__ == '__main__':
    parsed = parse_args()
    berryc = Chimera()
    berryc.loadMap(parsed.path,parsed.map)
    berryc.compose(parsed.shapes)

    if( parsed.image ):
        cv2.imwrite(parsed.image, berryc.composite)
    if( parsed.output != stdout_default ):
        np.save(parsed.output, berryc.contours)
    else:
        sys.stdout.write(berryc.contours)
