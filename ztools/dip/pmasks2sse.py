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
from ztools.dip._sse import *
import sys

UNLABELED_CLASS = 1

# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility for converting scikit-learn labeled plots into sse contours.")
    parser.add_argument('-r', '--root-path', dest='root', action='store', required=True, help="The absolute path on filesystem that the sse application considers it's root.")
    parser.add_argument('-f', '--path', dest='path', action='store', required=True, help="The path to search for the *.plabeled.npz files")
    parser.add_argument('--hostname', dest='hostname', action='store', default='localhost', help="The MongoDB hostname/ip to connect to.")
    parser.add_argument('-p', '--port', action='store', type=int, default=27017, help="The MongoDB port to connect to.")
    parser.add_argument('-d', '--db', action='store', default="meteor", help="MongoDB database name to access.")
    parser.add_argument('-g', '--gryg', action='store_true', help="Whether this is a dataset from CNJ0x population or GRYG population.")
    parser.add_argument('--chain_method', '--chain', dest='chain', action='store', default="CHAIN_APPROX_SIMPLE", choices=["CHAIN_APPROX_NONE", "CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_TC89_L1", "CHAIN_APPROX_TC89_KCOS"], help="Chain approximation method for contours.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)


if __name__ == '__main__':
    parsed      = parse_args()
    path_search = '{}/**/*.plabeled.npz'.format(parsed.path)
    plabeled_files  = glob(path_search, recursive=True)
    client      = MongoClient(parsed.hostname, parsed.port)
    db          = client[parsed.db]
    sse_samps   = db["SseSamples"]

    if parsed.gryg:
        socName = "GRYG Population Plots"
    else:
        socName = "CNJ0x Population Plots"

    for pf in plabeled_files:
        raw_image_basename = os.path.basename(re.sub(r'\.plabeled\.npz$','',pf))
        orig_image = raw_image_basename + '.JPG'
        print("Loading plot-labeled file {}".format(pf))
        labels = np.load(pf)['a']
        contours_sse = []
        extractContours(labels, contours_sse, parsed.chain, classIndex=UNLABELED_CLASS, layer=0)
        #Initialize a SseSamples document
        folder             = generateSSEFolder(parsed.root, pf)
        url                = generateSSEURL(folder, orig_image)
        current_datetime   = datetime.datetime.now()
        sse_samp = sse_samps.find_one({ "url": url })
        if( sse_samp ):
            sse_samp["lastEditDate"] = current_datetime
            sse_samp["objects"]      = contours_sse
            if "unmatched" not in sse_samp["tags"]:
                sse_samp["tags"].append("unmatched")
            sse_samps.update_one({'url': sse_samp['url']}, { "$set": sse_samp }, upsert=False)
        else:
            sse_samp = {  "url":             url,
                          "socName":         socName,
                          "firstEditDate":   current_datetime,
                          "lastEditDate":    current_datetime,
                          "folder":          folder,
                          "objects":         contours_sse,
                          "tags":            ["unmatched"],
                          "file":            orig_image }
            sse_samps.insert_one(sse_samp)
