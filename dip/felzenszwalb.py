#!/usr/bin/env python3
# Author: Andrew Maule
# Objective: To connect to the semantic segmentation editor (SSE) via its mongodb connection and inject a Felzenszwalb-segmentation of the image
#
# Algorithm overview:
#  - Connect to SSE's mongodb connection to access it's SseSamples collection under the meteor DB.
#  - Recurse through folder specified on command-line looking for image files and generate superpixels using SLIC protocol.
#  - Convert superpixel contours into annotated path objects recognized by SSE and inject into its DB.

# import the necessary packages
import argparse
import datetime
from glob import *
import json
import math
import numpy as np
import os
from pymongo import MongoClient
import re
from shapely.geometry import Point,Polygon
from shapely.ops import nearest_points,snap
from skimage.segmentation import find_boundaries,felzenszwalb
from skimage.util import img_as_float
from skimage import io
from skimage import measure
from skimage import filters
from _sse import *
import sys
import urllib

MAX_DISTANCE = 4000000.0

DEBUG = False
def sdebug(pstr):
    if DEBUG:
        sys.stdout.write(pstr)


def generateSegments(image_path, mdim):
    # load the image and convert it to a floating point data type
    image  = cv2.imread(image_path)
    mindim = np.argmin(image.shape[0:2]);
    newdim = list(image.shape[0:2]) #Initialize to the shape of the original image
    newdim[mindim] = mdim #Reset the smaller dimension to the one desired in the param mdim
    newdim[1-mindim] = int(mdim/image.shape[mindim] * image.shape[1-mindim]) #Set the larger dimension to it's corresponding scaled value based on the minimal dimension
    image_small  = img_as_float(cv2.resize(image, dsize=(newdim[1],newdim[0]), interpolation=cv2.INTER_CUBIC)) #Resize to reduce computational complexity
    # apply felzenszwalb
    segments = felzenszwalb(image_small, scale=100, sigma = 0.25, min_size=50)
    #Resize the 'segment' mask image  back up to original image size 
    segments = cv2.resize(segments, dsize=(image.shape[1],image.shape[0]), interpolation=cv2.INTER_NEAREST);
    return(segments);


def snapPolygons(polygons, dim_x, dim_y, threshold=2.0):
    num_polygons = len(polygons)
    num_polygons_x = round(math.sqrt((num_polygons * dim_x)/dim_y))
    inter_x_distance = dim_x/num_polygons_x
    num_polygons_y = round(num_polygons/num_polygons_x)
    inter_y_distance = dim_y/num_polygons_y
    neighbor_distance_threshold = max(inter_x_distance, inter_y_distance) * 2.5
    sdebug("Neighbor distance threshold: {}\n".format(neighbor_distance_threshold))
    #Initialize distances matrix to MAX_DISTANCE
    distances = np.zeros(shape=[num_polygons, num_polygons], dtype="float")
    distances[:,:] = MAX_DISTANCE
    for i in range(0, num_polygons):
        for j in range(i+1, num_polygons):
            distances[i,j]    = polygons[i].centroid.distance(polygons[j].centroid)
            sdebug("Distance between polygon {} and polygon {}: {}\n".format(i,j,distances[i,j]))
    neighbors = distances < neighbor_distance_threshold
    #Now that neighbors matrix has been calculated, let's do the snap operations
    for i in range(0, num_polygons):
        polygon_neighbors = np.where(neighbors[i])[0]
        for j in polygon_neighbors:
            sdebug("Snapping polygon %d to polygon %d.\n" % (i,j))
            polygons[i] = snap(polygons[i], polygons[j], threshold)
    return(polygons)


def convertPolygonsToContours(polygons, classIndex = 0, layer = 0):
    contours = [{}] * len(polygons)
    num_polygons = len(polygons)
    for i in range(0, num_polygons):
        (x,y) = polygons[i].exterior.coords.xy
        contours[i] = { "classIndex": classIndex, "layer": layer, "polygon": [{}] * len(x) }
        for j in range(0, len(x)):
            contours[i]["polygon"][j] = {"x": float(x[j]), "y": float(y[j])}
    return(contours)


# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility for generating felzenszwalb superpixels in image and injecting into SSE DB.")
    parser.add_argument('-f', '--path', action='store', required=True, help="The path to traverse looking for image files to generate SLIC superpixel data on.")
    parser.add_argument('--hostname', action='store', default='localhost', help="The MongoDB hostname/ip to connect to.")
    parser.add_argument('-p', '--port', action='store', type=int, default=3001, help="The MongoDB port to connect to.")
    parser.add_argument('-e', '--extension', action='store', default="*.JPG", help="The image extension to glob against.")
    parser.add_argument('-d', '--db', action='store', default="meteor", help="The name of the MongoDB database to use.")
    parser.add_argument('-s', '--socName', action='store', default="Drone Berry Development", help="The set-of-class name, or the identifier used to map object class ids to names.")
    parser.add_argument('-c', '--classIndex', action='store', type=int, default=0, help="The default class index (category index) to assign each superpixel object.")
    parser.add_argument('--min_dim', '--mdim', dest='mdim', type=int, default=480, help="Of the two dimensions of the image, resize image along the smaller of the two dimensions to this size, maintaining the dimension ratio for the larger dimension.  This is meant to reduce computational complexity.")
    parser.add_argument('--chain_method', '--chain', dest='chain', action='store', default="CHAIN_APPROX_SIMPLE", choices=["CHAIN_APPROX_NONE", "CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_TC89_L1", "CHAIN_APPROX_TC89_KCOS"], help="Chain approximation method for contours.")
    parser.add_argument('--settings', action='store', default="settings.json", help="The settings.json file used by the active running instance of SSE's meteor webapp.")
    parser.add_argument('-u', '--force-update', dest='update', action='store_true', help="If this flag is set, then forces an overwrite of the an image DB annotation if it already exists.")
    parser.add_argument('--debug', action='store_true', help="Print minimal debug while executing the script.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

  
if __name__ == '__main__':
    parsed        = parse_args()
    if( parsed.debug ):
        DEBUG = True
    client        = MongoClient(parsed.hostname, parsed.port)
    db            = client[parsed.db]
    sse_samples   = db["SseSamples"]
    image_path    = os.path.realpath(parsed.path)
    image_files   = glob(image_path+"/**/"+parsed.extension, recursive=True)
    settings      = json.loads(open(parsed.settings, 'r').read())
    sse_root      = os.path.realpath(settings['configuration']['images-folder'])
    for image_file in image_files[1:]:
        sdebug("Executing Felzenszwalb segmentation on image {}\n".format(image_file))
        segments     = generateSegments(image_file, parsed.mdim);
        #boundaries   = find_boundaries(segments, mode='subpixel')
        levels       = np.unique(segments)
        contours      = [None] * levels.size
        polygons      = [None] * levels.size
        #Generate the contour objects
        for i in levels:
            mask        = (segments == i).astype('float');
            contours_sse = []
            contours[i] = extractContours(mask, contours_sse, parsed.chain)
            polygons[i] = Polygon([[pt[0][0],pt[0,1]] for pt in contours[i][0]])
        #Snap polygons with each other -- This can be incredibly slow
        #dim_y = segments.shape[0]
        #dim_x = segments.shape[1]
        #polygons = snapPolygons(polygons, dim_x, dim_y)
        #Convert polygons back to coordinates
        contours = convertPolygonsToContours(polygons, parsed.classIndex)
        #Initialize a SseSamples document
        url                = generateSSEURL(sse_root, image_file)
        current_datetime   = datetime.datetime.now()
        sse_sample = sse_samples.find_one({ "url": url })
        if( sse_sample ):
            sdebug("{} already found in database: ".format(image_file))
            if( parsed.update ):
                sdebug("Updating ...\n")
                sse_sample["lastEditDate"] = current_datetime
                sse_sample["objects"]      = contours
                if "slic" not in sse_sample["tags"]:
                    sse_sample["tags"].append("felzen")
                sse_samples.update({ "url": url }, sse_sample, upsert=False)
            else:
                sdebug("_NOT_ updating.  Specify '-u' flag to force overwrite of existing entry in the database.\n")
        else:
            sdebug("Inserting {} in database.\n".format(image_file))
            sse_sample = {  "url":             url,
                            "socName":         parsed.socName, 
                            "firstEditDate":   current_datetime,
                            "lastEditDate":    current_datetime,
                            "folder":          generateSSEFolder(sse_root, image_file),
                            "objects":         contours,
                            "tags":            ["felzen"],
                            "file":            os.path.basename(image_file) }
            sse_samples.insert_one(sse_sample);
