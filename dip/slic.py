#!/usr/bin/env python
# Author: Andrew Maule
# Objective: To connect to the semantic segmentation editor (SSE) via its mongodb connection and inject Simple Linear Iterative Clustering (SLIC) superpixels into
#              images in the DB.
#
# Algorithm overview:
#  - Connect to SSE's mongodb connection to access it's SseSamples collection under the meteor DB.
#  - Recurse through folder specified on command-line looking for image files and generate superpixels using SLIC protocol.
#  - Convert superpixel contours into annotated path objects recognized by SSE and inject into its DB.

# import the necessary packages
import argparse
import cv2
import datetime
from glob import *
import numpy as np
import os
from pymongo import MongoClient
import re
from shapely.geometry import Point,Polygon
from shapely.ops import nearest_points,snap
from skimage.segmentation import find_boundaries,slic
from skimage.util import img_as_float
from skimage import io
from skimage import measure
from skimage import filters
import sys
import urllib

MAX_DISTANCE = 4000000.0

def generateSegments(image_path, num_segments):
    # load the image and convert it to a floating point data type
    image = img_as_float(io.imread(image_path))
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    segments = slic(image, n_segments = num_segments, convert2lab=True, enforce_connectivity=True, compactness=20.0, sigma = 2.5)
    return(segments)

def snapPolygons(polygons, threshold=2.0):
    num_polygons = len(polygons)
    distances = np.zeros(shape=[num_polygons, num_polygons], dtype="float")
    distances2 = np.zeros(shape=[num_polygons, num_polygons], dtype="float")
    #Initialize distances matrix to MAX_DISTANCE
    distances[:,:] = MAX_DISTANCE
    distances2[:,:] = MAX_DISTANCE
    neighbors = np.array((num_polygons, num_polygons), dtype="bool")
    for i in range(0, num_polygons):
        for j in range(i+1, num_polygons):
            print("Calculating distance between polygon %d and polygon %d." % (i,j))
            distances[i,j]    = polygons[i].distance(polygons[j])
    neighbors = distances < threshold
    #Now that neighbors matrix has been calculated, let's do the snap operations
    for i in range(0, num_polygons):
        polygon_neighbors = np.where(neighbors[i])[0]
        for j in polygon_neighbors:
            print("Snapping polygon %d to polygon %d." % (i,j))
            polygons[i] = snap(polygons[i], polygons[j], threshold)
    return(polygons)

def convertPolygonsToContours(polygons, contours):
    num_polygons = len(polygons)
    for i in range(0, num_polygons):
        (x,y) = polygons[i].exterior.coords.xy
        contours[i]["polygon"] = [{}] * len(x)
        for j in range(0, len(x)):
            contours[i]["polygon"][j] = {"x": float(x[j]), "y": float(y[j])}

def convertToPolygon(mask, classIndex=0, layer=0):
    #mask3 = np.zeros((mask.shape[0]+24, mask.shape[1]+24), dtype='float')
    #mask3[12:-12,12:-12] = mask
    mask2, contours = cv2.findContours(mask.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ##contours[0]         = contours[0] - [12,12]
    #contours_stacked    = measure.find_contours(mask3, level=0.5, positive_orientation='high', fully_connected='high')[0]
    #contours_stacked[contours_stacked < 0.] = 0.
    #contours_stacked = measure.approximate_polygon(contours_stacked, 0.25)
    contours_stacked    = np.vstack(np.vstack(contours[0]))
    polygon             = Polygon(contours_stacked)
    num_points          = contours_stacked.shape[0]
    contour_object      = { "classIndex": classIndex, "layer": layer, "polygon": None }
    #for i in range(0,num_points):
    #    contour_polygons[i] = {"x": float(contours_stacked[i][1]), "y": float(contours_stacked[i][0])}
    return((polygon,contour_object))

def generateFolder(path, filename):
    #Strip path prefix from filename
    p = re.compile("("+path+")(.+)")
    m = p.match(filename)
    folder = os.path.dirname('/'+m.group(2))
    return(folder)

def generateURL(path, filename):
    #Strip path prefix from filename
    file_path = generateFolder(path, filename)+'/'+os.path.basename(filename)
    url_path  = '/'+urllib.parse.quote(file_path, safe='')
    return(url_path)

# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility for generating SLIC superpixels in image and injecting into SSE DB.")
    parser.add_argument('-f', '--path', action='store', required=True, help="The path to traverse looking for image files to generate SLIC superpixel data on.")
    parser.add_argument('-n', '--num_segments', action='store', type=int, default=900, help="The number of superpixels to segment the image into.")
    parser.add_argument('--hostname', action='store', default='localhost', help="The MongoDB hostname/ip to connect to.")
    parser.add_argument('-p', '--port', action='store', type=int, default=3001, help="The MongoDB port to connect to.")
    parser.add_argument('-e', '--extension', action='store', default="*.JPG", help="The image extension to glob against.")
    parser.add_argument('-d', '--db', action='store', default="meteor", help="The name of the MongoDB database to use.")
    parser.add_argument('-s', '--socName', action='store', default="Drone Berry Development", help="The set-of-class name, or the identifier used to map object class ids to names.")
    parser.add_argument('-c', '--classIndex', action='store', type=int, default=2, help="The default class index (category index) to assign each superpixel object.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

  
if __name__ == '__main__':
    parsed        = parse_args()
    client        = MongoClient(parsed.hostname, parsed.port)
    db            = client[parsed.db]
    sse_samples   = db["SseSamples"]
    image_files   = glob(parsed.path+"/**/"+parsed.extension, recursive=True)
    for image_file in image_files:
        segments     = generateSegments(image_file, parsed.num_segments)
        #boundaries   = find_boundaries(segments, mode='subpixel')
        levels       = np.unique(segments)
        contours      = [[]] * levels.size
        polygons      = [[]] * levels.size
        #Generate the contour objects
        for i in levels:
            mask        = (segments == i).astype('float');
            (polygons[i],contours[i]) = convertToPolygon(mask, classIndex=parsed.classIndex, layer=0)
        #Snap polygons with each other
        polygons = snapPolygons(polygons)
        #Convert polygons back to coordinates
        convertPolygonsToContours(polygons, contours)
        #Initialize a SseSamples document
        url                = generateURL(parsed.path, image_file)
        current_datetime   = datetime.datetime.now()
        sse_sample = sse_samples.find_one({ "url": url })
        if( sse_sample ):
            sse_sample["lastEditDate"] = current_datetime
            sse_sample["objects"]      = contours
            if "slic" not in sse_sample["tags"]:
                sse_sample["tags"].append("slic")
            #r = sse_samples.replace_one({ "url": url }, sse_sample, upsert=True)
            test = None
        else:
            sse_sample = {  "url":             url,
                            "socName":         parsed.socName, 
                            "firstEditDate":   current_datetime,
                            "lastEditDate":    current_datetime,
                            "folder":          generateFolder(parsed.path, image_file),
                            "objects":         contours,
                            "tags":            ["slic"],
                            "file":            os.path.basename(image_file) }
            #r = sse_samples.insert_one(sse_sample);
            test = None
