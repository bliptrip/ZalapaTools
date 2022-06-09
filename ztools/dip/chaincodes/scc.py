#!/usr/bin/env python3
# Author: Andrew Maule
# Objective: Calculates the slope chain code on a berry contour, as defined
# in Bribiesca et al.'s 2013 work: https://doi.org/10.1016/j.patcog.2012.09.017
#
# NOTE: This is a slightly different version of SCC that is easier to code
#           in python using the shapely library.
#       Rather than overlaying a circle along the path of the contours, I use
#           the shapely library function 'simplify' to reduce discretize a
#           polygon.  From there, I traverse the points of the simplified
#           contour and calculate slope changes as I traverse.
#   The goal here is to calculate the shape's tortuosity.
#

import argparse
import logging
import math
import numpy as np
from shapely.geometry import LinearRing
import sys

# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility for generating an extended slope chain code (SCC) representation of berry shapes.")
    parser.add_argument('-i', '--input', action='store', type=str, required=True, help="Path to input file containing numpy-formatted array of contours for berry shape.")
    parser.add_argument('-o', '--output', action='store', type=str, required=True, help="Path to output file to store numpy-formatted array of SCC.")
    parser.add_argument('-l', '--level', type=str, default="WARNING", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Set the logging level for the SCC encoder (debugging purposes).")
    parser.add_argument('-t', '--tolerance', type=float, default=0.5, help="Starting tolerance for simplifying berry contours before calculating SCC - this tool will progressively calculate tortuosity at lower and lower "
                            "tolerances until it detects no change -- it will chose the tolerance with the highest tolerance/simplist representation of the berry contour.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

class SCCEncoder(object):
    def __init__(self, logger=None, tolerance=0.1, contours=None, SCC=None):
        if SCC is not None:
            self.SCC  = SCC
        else:
            self.SCC = []
        self.logger = logger
        if contours is not None:
            self.encode(contours, tolerance)

    def encode(self, contours, tolerance):
        #Shift contours to have the max y contour be the first point to begin encoding traversal process
        contours = LinearRing(contours)
        if( contours.is_ccw ):
            contours.coords = list(contours.coords)[::-1] #Make clockwise
        contours_simple = contours.simplify(tolerance)
        #Traverse, calculating slopes
        contours_simple_a = np.asarray(contours_simple)
        if(np.all(contours_simple_a[0] == contours_simple_a[-1])): #If we have a repeat at start and end, remove it
            contours_simple_a = contours_simple_a[0:-1]
        self.SCC = []
        for i in range(0, len(contours_simple_a)): #Start from 1 as we assume start/end are equivalent
            past = contours_simple_a[i-1]
            current = contours_simple_a[i]
            future = contours_simple_a[(i+1)%len(contours_simple_a)]
            diff1 = current - past
            slope_1 = math.atan2(diff1[0],diff1[1])
            diff2 = future - current
            slope_2 = math.atan2(diff2[0],diff2[1])
            alpha = (slope_2 - slope_1)/math.pi #Normalize to [-1,1]
            self.SCC.append(alpha)
        return

    def tortuosity(self, normalize=False):
        tort = np.sum(np.absolute(np.asarray(self.SCC)))
        if normalize == True:
            tort = tort/len(self.SCC)
        return(tort)

    def raw(self):
        return(np.array(self.SCC))

    def __str__(self):
        return(','.join(self.SCC))


if __name__ == '__main__':
    logger      = logging.getLogger()
    parsed      = parse_args()
    decoded_level = eval("logging.{}".format(parsed.level))
    logger.setLevel(decoded_level)
    contours    = np.load(parsed.input)
    highest_tolerance = parsed.tolerance
    tolerance = parsed.tolerance
    scc        = SCCEncoder(logger=logger,contours=contours,tolerance=tolerance)
    highest_tortuosity = scc.tortuosity()
    while( True ):
        tolerance = tolerance/2
        scc.encode(contours=contours, tolerance=tolerance)
        new_tortuosity = scc.tortuosity()
        if( scc.tortuosity() > 1.025 * highest_tortuosity ): #Give ourselves some buffer room on this
            highest_tolerance = tolerance
            highest_tortuosity = new_tortuosity
        else:
            break
    scc.encode(contours=contours, tolerance=highest_tolerance)
    print("File: {},: Tolerance: {}, Tortuosity: {} ({})".format(parsed.input,highest_tolerance,scc.tortuosity(),scc.tortuosity(True)))
