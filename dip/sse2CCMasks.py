#!/usr/bin/env python
# Author: Andrew Maule
# Objective: Connects to an SSE mongodb server, pulls out Color Card-annotated images, and saves selected
#               colorcard tiles as grayscale masks for use by PlantCV's color-correction utility.
#
#
# Parameters:
#   - Local Image path
#   - MongoDB hostname
#   - MongoDB port
#   - And more -- issue --help to see full-list
#
# Algorithm overview:
#  - Connect to SSE's mongodb connection to access it's SseSamples collection under the meteor DB.
#

import argparse
import cv2
import json
import numpy as np
from pathlib import PosixPath
from pymongo import MongoClient
import sys

# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility for converting annotated images on a Semantic Segmentation Editor server to grayscale-masks, one per color-checker card chip.")
    parser.add_argument('--hostname', action='store', default='localhost', help="The MongoDB hostname/ip to connect to.")
    parser.add_argument('-p', '--port', action='store', type=int, default=3001, help="The MongoDB port to connect to.")
    parser.add_argument('-d', '--db', action='store', default="meteor", help="MongoDB database name to access.")
    parser.add_argument('-s', '--settings', action='store', default="settings.json", help="The settings.json file.")
    parser.add_argument('-c', '--class', '--settings-class', dest='settings_class', default='X-Rite Color Checker Card',  help="The name of the 'sets-of-classes' objects used for generating image masks using the semantic segmentation editor settings file.")
    parser.add_argument('--tags', dest='tags', action='append', default=["cc"], help='SseSample tags to filter for when looking for annotated images with color checker.')
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)


if __name__ == '__main__':
    parsed        = parse_args()
    #Load the settings.json file and pull the aisle class and the outer ring/inner ring class index as defined in the mongodb json object annotations.
    #This information is needed as aisles do not form a polygon, which is a limitation of the annotation software used.  Moreover, the plots and partial
    #plots with rings need to be properly segmented, as they were drawn as overlapping polygons.
    with open(parsed.settings, 'r') as settings_fh:
        settings            = json.loads(settings_fh.read())
        root                = settings['configuration']['images-folder']
        sets_of_classes     = settings['sets-of-classes']
        relevant_classes       = filter(lambda c: parsed.settings_class == c['name'], sets_of_classes)
        for rc in relevant_classes:
            all_labels = [r['label'] for r in rc['objects']]
            rcos = list(enumerate(filter(lambda r: (r['label'] != 'VOID') and (r['label'] != 'Square Divider'),rc['objects'])))
            for d in rcos:
                ddict = d[1]
                color_h = ddict['color'].lstrip('#')
                ddict['rgb'] = tuple(int(color_h[i:i+2], 16) for i in (0, 2, 4)) #Deconstruct hex string
        output_indices = [True]*len(rcos) #Initialize to all indices in the initial annotated images
        client          = MongoClient(parsed.hostname, parsed.port)
        db              = client[parsed.db]
        sse_samples     = db.SseSamples
        for annotation in sse_samples.find({"tags": {"$in": parsed.tags}}):
            #Find the corresponding image on the local drive
            img_path = PosixPath(root + annotation['folder'] + '/' + annotation['file'])
            #Load the image, mostly to just get size parameters
            img = cv2.imread(str(img_path))
            mask            = np.zeros((img.shape[0],img.shape[1]), dtype='uint8')
            for i,r in enumerate(rcos):
                class_label = r[1]['label']
                class_index = all_labels.index(class_label)
                #Filter contours by classIndex
                contourObjects  = filter(lambda o: class_index == o['classIndex'], annotation['objects'])
                contours        = [ np.array([[[np.round(p['x']),np.round(p['y'])]] for p in o['polygon']], dtype='int32') for o in contourObjects ]
                cv2.drawContours(mask, contours, -1, color=((i+1)*10), thickness=cv2.FILLED)
            #Save the grayscale mask to a file with same path as the source image file path, with *.masks.png extension.
            mask_dir = img_path.parent
            mask_stem   = img_path.stem
            mask_path = mask_dir / (str(mask_stem) + ".mask.PNG")
            cv2.imwrite(str(mask_path),mask)
