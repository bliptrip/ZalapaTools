#!/usr/bin/env python3
# Author: Andrew Maule
# Date: 2019-01-14
# Objective: To take multichannel masks and convert to datapoints for the semantic segmentation editor (SSE).  Data is pushed to the mongodb connection used by the SSE server.
#
#
# Parameters:
#   - Path for FPN-predicted *.mask.npz mask files.
#   - MongoDB hostname
#   - MongoDB port
#   
#
# Algorithm overview:
#  - Recursively search for all FPN-predicted *.mask.npz files from a given directory, and for each mask, it's original image file, and a map file, convert these
#    predicted masks to sse contours.

import argparse
import datetime
from glob import glob
import json
import numpy as np
import os
from pymongo import MongoClient
import re
from shapely.geometry import Polygon
+from _sse import *
import sys

# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility for converting FPN-predicted semantic segmentation masks images to Semantic Segmentation Editor polygon objects for display by the software.")
    parser.add_argument('-f', '--path', dest='path', action='store', required=True, help="The path to search for the *.mask.npz files")
    parser.add_argument('--hostname', dest='hostname', action='store', default='localhost', help="The MongoDB hostname/ip to connect to.")
    parser.add_argument('-p', '--port', action='store', type=int, default=3003, help="The MongoDB port to connect to.")
    parser.add_argument('-d', '--db', action='store', default="meteor", help="MongoDB database name to access.")
    parser.add_argument('--mask-map', dest='map', default='sse2masks.map', help='A JSON-encoded map file containing the details on how the original segmentation masks map to masks used in training/prediction (for when deletion/merging is used)')
    parser.add_argument('--chain_method', '--chain', dest='chain', action='store', default="CHAIN_APPROX_SIMPLE", choices=["CHAIN_APPROX_NONE", "CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_TC89_L1", "CHAIN_APPROX_TC89_KCOS"], help="Chain approximation method for contours.")
    parser.add_argument('-s', '--socName', action='store', default="Drone Berry Development", help="The set-of-class name, or the identifier used to map object class ids to names.")
    parser.add_argument('-e', '--exclude', '--exclude-masks', dest='exclude', nargs='+', default=["Aisle"], help="Exclude semantic segmentation mask with categories specified in this parameter.")
    parser.add_argument('--settings', '--exclude', '--exclude-masks', dest='exclude', nargs='+', default=["Aisle"], help="Exclude semantic segmentation mask with categories specified in this parameter.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)


def findClassIndex(drone_class_objects,name):
    index               = -1
    for i,dco in drone_class_objects:
        if name == dco['label']:
            index = i
            break
    return index


def getClassIndex(o):
    return o['classIndex'];


def convertPolygonsToContours(polygons, contours):
    num_polygons = len(polygons)
    for i in range(0, num_polygons):
        (x,y) = polygons[i].exterior.coords.xy
        contours[i]["polygon"] = [{}] * len(x)
        for j in range(0, len(x)):
            contours[i]["polygon"][j] = {"x": float(x[j]), "y": float(y[j])}

if __name__ == '__main__':
    parsed      = parse_args()
    path_search = '{}/**/*.masks.npz'.format(parsed.path)
    mask_files  = glob(path_search, recursive=True)
    meta        = json.loads(open(parsed.map,'r').read())
    client      = MongoClient(parsed.hostname, parsed.port)
    db          = client[parsed.db]
    sse_samps   = db["SseSamples"]
    for mf in mask_files:
        raw_image_filename = os.path.basename(re.sub(r'\.masks\.npz$','',mf))
        print("Loading mask file {}".format(mf))
        masks = np.load(mf)['masks']
        num_masks = masks.shape[-1]
        #Do some validity checking?  See if the number of mask layers matches that in meta
        #Binarize the masks to the majority player
        maxidxs         = np.argmax(masks, axis=-1) #Returns which segmentation class has the majority vote on a pixel
        idm             = np.identity(num_masks,dtype='float')
        bin_masks       = idm[maxidxs]
        contours_sse    = []
        for m in range(num_masks):
            if not (meta['new-objects'][m][1]['label'] in parsed.exclude):
                origClassIndex = meta['new-objects'][m][0] #Back-map the current class index to the original
                extractContours(bin_masks[:,:,m], contours_sse, parsed.chain, classIndex=origClassIndex, layer=0)
        #Initialize a SseSamples document
        folder             = generateSSEFolder(parsed.path, raw_image_path)
        url                = generateSSEURL(folder, raw_image_path)
        current_datetime   = datetime.datetime.now()
        sse_samp = sse_samps.find_one({ "url": url })
        if( sse_samp ):
            sse_samp["lastEditDate"] = current_datetime
            sse_samp["objects"]      = contours_sse
            if "predicted" not in sse_samp["tags"]:
                sse_samp["tags"].append("predicted")
            sse_samps.update({'url': sse_samp['url']}, sse_samp, upsert=False)
        else:
            sse_samp = {  "url":             url,
                          "socName":         parsed.socName, 
                          "firstEditDate":   current_datetime,
                          "lastEditDate":    current_datetime,
                          "folder":          generateFolder(parsed.path, raw_image_path),
                          "objects":         contours_sse,
                          "tags":            ["predicted"],
                          "file":            os.path.basename(raw_image_path) }
            sse_samps.insert_one(sse_samp)

