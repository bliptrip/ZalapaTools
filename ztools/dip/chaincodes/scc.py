#!/usr/bin/env python3
# Author: Andrew Maule
# Objective: Calculates the slope chain code on a berry contour, as defined
# in Bribiesca et al.'s 2013 work: https://doi.org/10.1016/j.patcog.2012.09.017
#

import argparse
import logging
import math
import numpy as np
from shapely.geometry import LineString, LinearRing, Point, MultiPoint
import sys

# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility for generating an extended slope chain code (SCC) representation of berry shapes.")
    parser.add_argument('-i', '--input', action='store', type=str, required=True, help="Path to input file containing numpy-formatted array of contours for berry shape.")
    parser.add_argument('-o', '--output', action='store', type=str, required=True, help="Path to output file to store numpy-formatted array of SCC.")
    parser.add_argument('-l', '--level', type=str, default="WARNING", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Set the logging level for the SCC encoder (debugging purposes).")
    parser.add_argument('-n', '--num-segments', dest='segments', default=None, type=int, help="Number of SCC segments to divide contours up into.")
    parser.add_argument('--segment-length', dest='seglength', default=None, type=float, help="Segment lengths")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)


def generateTransectCircle(radius, center=(0,0), num_points=64):
    points = []
    for theta in np.linspace(start=0, stop=2*math.pi, num=num_points, endpoint=False, dtype='float'):
        points.append((center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta)))
    circle = LinearRing(points)
    return(circle)


class SCCEncoder(object):
    def __init__(self, logger=None, segment_length=None, num_segments=None, contours=None, SCC=None, precision=10e-1):
        self.points = [] #Points along original contour that are used for generating SCC
        self.circles = []
        self.precision = precision
        if SCC is not None:
            self.SCC  = SCC
        else:
            self.SCC = []
        self.logger = logger
        if contours is not None:
            self.encode(contours, segment_length, num_segments)

    def encode(self, contours, segment_length=None, num_segments=None):
        #Reset
        self.SCC = []
        self.points = []
        self.circles = []
        #Shift contours to have the max y contour be the first point to begin encoding traversal process
        circles = []
        scc_points = []
        contours = LineString(contours)
        if( (segment_length != None) and (num_segments != None) ):
            assert(False, "Both segment length and number of segments are defined.  Only once should be defined.")
        if segment_length != None:
            radius = segment_length
        elif num_segments != None:
            radius = contours.length/num_segments
        else:
            assert(False, "Neither segment length or number of segments are defined.  One and only one needs to be defined.")
        if contours.length <= radius:
            return
        if(np.all(contours.coords[0] == contours.coords[-1])): #If we have a repeat at start and end, remove it
            contours = LineString(contours.coords[0:-1])
        circle_base = generateTransectCircle(radius)
        n = Point(np.round(contours.coords[0],decimals=2)) #Grab the first contour point as the first point of scc
        distances_total = [0.0]
        while (distances_total[-1] <= (1.10*contours.length)) and (n is not None): #Add 10% to contours length as the SCC line segments and contours
            scc_points.append(n)
            circle = LinearRing(np.array(circle_base.coords) + np.array(n.coords))
            circles.append(circle)
            ints = contours.intersection(circle)
            if type(ints) == Point: 
                nps = [ints]
            elif type(ints) == MultiPoint:
                nps = ints.geoms
            else:
                break
            distances = []
            nps = [(round(p.x,2),round(p.y,2)) for p in nps]
            for e in nps:
                distances.append(contours.project(Point(e)))
            distances = np.round(np.array(distances), decimals=2)
            d_tf = distances > np.max(distances_total)
            nps = np.array(nps)[d_tf]
            distances = distances[d_tf]
            distances_sorted_i = np.argsort(distances)
            n = None
            for i in distances_sorted_i:
                cp = Point(nps[i,:])
                if cp not in scc_points:
                    distances_total.append(distances[i])
                    n = cp
                    break
        if len(scc_points) > 2:
            scc_line = LineString(scc_points)
            diffs = np.array(scc_line.coords[0:] + [scc_line.coords[0]]) - np.array([scc_line.coords[-1]] + scc_line.coords[0:])
            slopes = np.arctan2(diffs[:,1], diffs[:,0])
            slopes_alpha = slopes / math.pi
            alphas = slopes_alpha[1:] - slopes_alpha[0:-1]
            slopes_adjust = np.any((alphas > 1) | (alphas < -1))
            while slopes_adjust == True:
                alphas[alphas > 1] = alphas[alphas > 1] - 2 #2 represents one full rotation/cycle on polar coordinates
                alphas[alphas < -1] = alphas[alphas < -1] + 2 #2 represents one full rotation/cycle on polar coordinates
                slopes_adjust = np.any((alphas > 1) | (alphas < -1)) #Any slopes still need adjusting so that they are within [-1,1] range?
            self.SCC = [a for a in alphas]
            self.points = scc_points
            self.diffs = diffs
            self.slopes = slopes
            self.circles = circles
        return

    def tortuosity(self, normalize=False):
        tort = np.sum(np.absolute(np.asarray(self.SCC)))
        if normalize == True:
            tort = tort/len(self.SCC)
        return(tort)

    def raw(self):
        return(np.array(self.SCC))

    def decode(self, encstring):
        SCC = ((np.array(encstring.encode('utf32'),dtype=float) - 0x400) * 2 * self.precision) + 1
        return(SCC)

    def __str__(self):
        #Discretize the output into a string of characters 
        charsb = (((np.array(self.SCC, dtype='float') - (-1))/2)/self.precision) + 0x400 #0x400 is the beginning of the cyrillic alphabet, which offers 256 contiguous character options
        chars = bytes(charsb.astype('uint32')).decode('utf32')
        return(chars)

if __name__ == '__main__':
    logger      = logging.getLogger()
    parsed      = parse_args()
    decoded_level = eval("logging.{}".format(parsed.level))
    logger.setLevel(decoded_level)
    contours    = np.load(parsed.input)
    highest_tolerance = parsed.tolerance
    num_segments = parsed.segments
    scc        = SCCEncoder(logger=logger,contours=contours,num_segments=num_segments)
    highest_tortuosity = scc.tortuosity()
    while( True ):
        num_segments = num_segments/2
        scc.encode(contours=contours, num_segments=num_segments)
        new_tortuosity = scc.tortuosity()
        if( scc.tortuosity() > 1.025 * highest_tortuosity ): #Give ourselves some buffer room on this
            highest_num_segments = num_segments
            highest_tortuosity = new_tortuosity
        else:
            break
    scc.encode(contours=contours, num_segments=highest_num_segments)
    print("File: {},: Number of SCC Segments: {}, Tortuosity: {} ({})".format(parsed.input,highest_num_segments,scc.tortuosity(),scc.tortuosity(True)))
