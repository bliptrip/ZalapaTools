#!/usr/bin/env python
# Author: Andrew Maule
# Objective: To connect to the semantic segmentation editor (SSE) via its mongodb connection, access annotated images, pull the annotation state,
#              and generate n-channel tensors of 2D binary masks for each annotation class (for segmentation masks).
#
#
# Parameters:
#   - Local Image path
#   - MongoDB hostname
#   - MongoDB port
#
# Algorithm overview:
#  - Connect to SSE's mongodb connection to access it's SseSamples collection under the meteor DB.
#  - Recurse through folder specified on command-line looking for image files and generate superpixels using SLIC protocol.
#  - Convert superpixel contours into annotated path objects recognized by SSE and inject into its DB.

import argparse
import cv2
from glob import glob
import gzip
import json
import numpy as np
from pymongo import MongoClient
import sys

# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility for converting annotated images on a Semantic Segmentation Editor server to n-channel binary masks, one for each class.  The output is stored as a numpy tensor.")
    parser.add_argument('-f', '--path', action='store', required=True, help="The local image path to store output masks to.")
    parser.add_argument('--hostname', action='store', default='localhost', help="The MongoDB hostname/ip to connect to.")
    parser.add_argument('-p', '--port', action='store', type=int, default=3003, help="The MongoDB port to connect to.")
    parser.add_argument('-d', '--db', action='store', default="meteor", help="MongoDB database name to access.")
    parser.add_argument('-s', '--settings', action='store', default="settings.json", help="The settings.json file.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)


def findClassIndex(drone_class_objects,name):
    index               = -1
    for i,dco in enumerate(drone_class_objects):
        if name == dco['label']:
            index = i
            break
    return index


def generateMultiChannelMask(img, masks, drone_class_objects):
    mchannel_mask       = np.zeros(img.shape, dtype='uint8')
    for i,mask in enumerate(masks):
        h = drone_class_objects[i]['color'].lstrip('#')
        bgr = tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))
        mchannel_mask[mask,:] = bgr
    return(mchannel_mask)
        

def getClassIndex(o):
    return o['classIndex']


if __name__ == '__main__':
    parsed        = parse_args()
    #Load the settings.json file and pull the aisle class and the outer ring/inner ring class index as defined in the mongodb json object annotations.
    #This information is needed as aisles do not form a polygon, which is a limitation of the annotation software used.  Moreover, the plots and partial
    #plots with rings need to be properly segmented, as they were drawn as overlapping polygons.
    with open(parsed.settings, 'r') as settings_fh:
        settings      = json.loads(settings_fh.read())
        sets_of_classes     = settings['sets-of-classes']
        drone_classes       = filter(lambda c: "Drone Berry Development" == c['name'], sets_of_classes)
        for dc in drone_classes:
            dcos = dc['objects']
        aisle_idx     = findClassIndex(dcos,'Aisle')
        plots_idx     = findClassIndex(dcos,'Plot')
        pplots_idx    = findClassIndex(dcos,'Partial Plot')
        harvesto_idx  = findClassIndex(dcos,'Harvest Ring - Outer')
        harvesti_idx  = findClassIndex(dcos,'Harvest Ring - Inner')
        client        = MongoClient(parsed.hostname, parsed.port)
        db            = client[parsed.db]
        sse_samples   = db["SseSamples"]
        for annotation in sse_samples.find():
            #Find the corresponding image on the local drive
            img_path = parsed.path + '/' + annotation['folder'] + '/' + annotation['file']
            #Load the image, mostly to just get size parameters
            img = cv2.imread(img_path)
            #Filter contours by classIndex
            class_mask = []
            union = np.zeros((img.shape[0],img.shape[1]), dtype='bool')
            for i in range(0, len(dcos)):
                contourObjects  = filter(lambda o: i == o['classIndex'],annotation['objects'])
                contours        = [ np.array([[[np.round(p['x']),np.round(p['y'])]] for p in o['polygon']], dtype='int32') for o in contourObjects ]
                mask            = np.zeros((img.shape[0],img.shape[1]), dtype='float')
                cv2.drawContours(mask, contours, -1, color=1.0, thickness=cv2.FILLED)
                mask_bin = mask.astype('bool')
                union    = union | mask_bin
                class_mask.append(mask_bin)
            #Now that we have all contours for all classes, build the aisle mask as the inverse of the union of all other masks
            class_mask[aisle_idx] = ~union #Generate the inverse
            #Subtract out harvest rings from both plots and partial plots
            class_mask[plots_idx] = class_mask[plots_idx] & ~class_mask[harvesto_idx]
            class_mask[pplots_idx] = class_mask[pplots_idx] & ~class_mask[harvesto_idx]
            #Subtract out the inner part of harvest ring from outer
            class_mask[harvesto_idx] = class_mask[harvesto_idx] & ~class_mask[harvesti_idx]
            for i,m in enumerate(class_mask):
                cv2.imwrite(img_path+'.mask.'+str(i)+'.png',m.astype('uint8')*255)
            #Generate a multi-channel mask image that can be viewed for confirmation.
            mchannel_mask = generateMultiChannelMask(img, class_mask, dcos)
            cv2.imwrite(img_path+'.mask.png',mchannel_mask)
