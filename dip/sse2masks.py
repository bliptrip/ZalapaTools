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
import copy
import cv2
from glob import glob
import gzip
import json
import numpy as np
import pyexiv2
from pymongo import MongoClient
import sys

#Some label constants -- can change these in the future, if they are changed in settings.json
LABEL_VOID = 'VOID'
LABEL_AISLE = 'Aisle'
LABEL_PLOT = 'Plot'
LABEL_PARTIAL_PLOT = 'Partial Plot'
LABEL_BEDDING = 'Bedding'
LABEL_HARVEST_RING_OUTER = 'Harvest Ring - Outer' 
LABEL_HARVEST_RING_INNER = 'Harvest Ring - Inner'
LABEL_CRANBERRY_VEGETATION = 'Cranberry Vegetation'


# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility for converting annotated images on a Semantic Segmentation Editor server to n-channel binary masks, one for each class.  The output is stored as a numpy tensor.")
    parser.add_argument('-f', '--path', action='store', required=True, help="The local image path to store output masks to.")
    parser.add_argument('--hostname', action='store', default='localhost', help="The MongoDB hostname/ip to connect to.")
    parser.add_argument('-p', '--port', action='store', type=int, default=3003, help="The MongoDB port to connect to.")
    parser.add_argument('-d', '--db', action='store', default="meteor", help="MongoDB database name to access.")
    parser.add_argument('-s', '--settings', action='store', default="settings.json", help="The settings.json file.")
    parser.add_argument('-c', '--class', '--settings-class', dest='settings_class', default='Drone Berry Development',  help="The name of the 'sets-of-classes' objects used for generating image masks using the semantic segmentation editor settings file.")
    parser.add_argument('-m', '--mode', action='store', choices=['pass-through', 'merge-partial', 'merge-partial-bedding'], default='pass-through', help='The mode, or type of preprocessing that should be done on the sse-designated masks.\r\n' +
            '   pass-through: Converts SSE masks one-for-one to prediction masks (except that outer and inner harvest ring masks are auto-generated, and aisle masks are also auto-generated as anything not currently represented as a mask).\r\n' +
            '   merge-partial: Does everything that pass-through option does, except it merges plots and partial-plots into one mask.\r\n' +
            '   merge-partial-bedding: Does everything that pass-through option does, except it merges plots, partial-plots, and bedding into one mask.\r\n' +
            'NOTE: The logic in merging partial plots/bedding with plots into a single segmentation mask has to do with potentially getting better training/testing/prediction accuracy by having one less class.\n')
    parser.add_argument('--mask-format', '--format', dest='format', action='store', choices=['split-binary', 'split-gray','layer-binary','layer-gray'], default='split-gray', help='The output format to use when generating mask files.  ' +
            'split-binary: This splits the mask layers into separate files and packs them into small, binary format.  ' + 
            'split-gray: Default.  This splits the mask layers into separate files and stores them as grayscale PNG files for easy viewing on any OS.  ' +
            'layer-binary: This stores the mask file as a multi-layer (multi-channel) numpy array in packed binary format.  This reduces folder clutter from having a single file for each segmentation mask for each annotated image.  ' +
            'layer-gray: This stores the mask file as a multi-layer (multi-channel) numpy array in a grayscale (uint8) format.')
    parser.add_argument('--mask-image', dest='mask', action='store_true', help='Generate original image with overlayed masks in PNG format.  NOTE: mask colors based on original ')  
    parser.add_argument('-a', '--alpha', '--mask-image-alpha', dest='alpha', action='store', type=float, default=0.7, help='Generate original image with overlayed masks in PNG format.')  
    parser.add_argument('--strip', '--delete-classes', dest='strip', action='append', help='Whether to remove/strip certain classes from the original settings.json file (either b/c they are irrelevant, or were never annotated in images.)')
    parser.add_argument('--mask-map', dest='map', default='sse2masks.map', help='A JSON-encoded map file containing the details on how the original segmentation masks map to masks used in training/prediction (for when deletion/merging is used)')
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)


def findClassIndex(drone_class_objects,name):
    index               = -1
    for i,dco in drone_class_objects:
        if name == dco['label']:
            index = i
            break
    return index


def generateMaskOverlay(img, masks, drone_class_objects, alpha):
    newimg        = np.zeros(img.shape, dtype='uint8')
    overlay       = np.zeros(img.shape, dtype='uint8')
    for i,mask in enumerate(masks):
        h = drone_class_objects[i][1]['color'].lstrip('#')
        bgr = tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))
        overlay[mask,:] = bgr
    cv2.addWeighted(overlay, alpha, img, 1.0-alpha, 0.0, newimg)
    return((overlay,newimg))
        

def generateXMPMetadata(img_path,argv,json,author='Andrew Maule'):
    meta = pyexiv2.ImageMetadata(img_path)
    meta.read()
    meta['Xmp.dc.creator'] = ['Andrew Maule']
    meta['Xmp.custom.cli'] = ' '.join(sys.argv)
    meta['Xmp.custom.data'] = sse2mask_map_json
    meta.write()

def getClassIndex(o):
    return o['classIndex']


if __name__ == '__main__':
    pyexiv2.register_namespace('/', 'custom')
    parsed        = parse_args()
    sse2mask_map_dict = {'sse_settings_file': None,
                         'class': None,
                         'orig-objects': [],
                         'new-objects': [],
                         'labels': [], #The labels associated with the generated mask files (different from original labels b/c of deletion and/or merging)
                         'map': {} #The map of the new label to the old label
                        }
    #Load the settings.json file and pull the aisle class and the outer ring/inner ring class index as defined in the mongodb json object annotations.
    #This information is needed as aisles do not form a polygon, which is a limitation of the annotation software used.  Moreover, the plots and partial
    #plots with rings need to be properly segmented, as they were drawn as overlapping polygons.
    with open(parsed.settings, 'r') as settings_fh:
        sse2mask_map_dict['sse_settings_file'] = parsed.settings
        sse2mask_map_dict['class'] = parsed.settings_class
        settings            = json.loads(settings_fh.read())
        sets_of_classes     = settings['sets-of-classes']
        drone_classes       = filter(lambda c: "Drone Berry Development" == c['name'], sets_of_classes)
        for dc in drone_classes:
            dcos = list(enumerate(dc['objects']))
            sse2mask_map_dict['orig-objects'] = copy.deepcopy(dcos) #Make a deep copy of the original
        output_indices = [True]*len(dcos) #Initialize to all indices in the initial annotated images
        #Get original indices before any stripping/remapping occurs
        void_idx        = findClassIndex(dcos,LABEL_VOID)
        aisle_idx       = findClassIndex(dcos,LABEL_AISLE)
        plots_idx       = findClassIndex(dcos,LABEL_PLOT)
        pplots_idx      = findClassIndex(dcos,LABEL_PARTIAL_PLOT)
        bedding_idx     = findClassIndex(dcos,LABEL_BEDDING)
        harvesto_idx    = findClassIndex(dcos,LABEL_HARVEST_RING_OUTER)
        harvesti_idx    = findClassIndex(dcos,LABEL_HARVEST_RING_INNER)
        #Initialize map of original mask identifiers to new identifiers
        sse2mask_map_dict['map'] = {e['label']: e['label'] for i,e in dcos}
        #Strip/delete
        for strip_object_name in parsed.strip:
            object_idx = findClassIndex(dcos,strip_object_name)
            if object_idx != -1:
                output_indices[object_idx] = False
        #Preprocess the sse2mask_map_dict labels and new-objects based on the mode
        if parsed.mode == "merge-partial": 
            output_indices[findClassIndex(dcos,LABEL_PARTIAL_PLOT)] = False
            sse2mask_map_dict['map'][LABEL_PARTIAL_PLOT] = findClassIndex(dcos,LABEL_PARTIAL_PLOT)
        elif parsed.mode == "merge-partial-bedding":
            #Rename
            dcos[findClassIndex(dcos,LABEL_PLOT)][1]['label'] = LABEL_CRANBERRY_VEGETATION
            sse2mask_map_dict['map'][LABEL_PLOT] = LABEL_CRANBERRY_VEGETATION
            #Remap
            output_indices[findClassIndex(dcos,LABEL_PARTIAL_PLOT)] = False
            sse2mask_map_dict['map'][LABEL_PARTIAL_PLOT] = LABEL_CRANBERRY_VEGETATION
            #Remap
            output_indices[findClassIndex(dcos,LABEL_BEDDING)] = False
            sse2mask_map_dict['map'][LABEL_BEDDING] = LABEL_CRANBERRY_VEGETATION
        sse2mask_map_dict['new-objects'] = copy.deepcopy([e[1] for e in filter(lambda x: x[0],zip(output_indices,dcos))])
        #Now that we have removed the requested classes, generate a one-to-one map
        sse2mask_map_dict['labels'] = [e[1]['label'] for e in sse2mask_map_dict['new-objects']]
        sse2mask_map_json = json.dumps(sse2mask_map_dict)
        client          = MongoClient(parsed.hostname, parsed.port)
        db              = client[parsed.db]
        sse_samples     = db["SseSamples"]
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
            #Subtract out harvest rings from both plots and partial plots
            for strip_class in parsed.strip:
                strip_class_idx = findClassIndex(dcos,strip_class)
                class_mask[void_idx] |= class_mask[strip_class_idx]
            class_mask[plots_idx] = class_mask[plots_idx] & ~class_mask[harvesto_idx]
            class_mask[pplots_idx] = class_mask[pplots_idx] & ~class_mask[harvesto_idx]
            #Subtract out the inner part of harvest ring from outer
            class_mask[harvesto_idx] = class_mask[harvesto_idx] & ~class_mask[harvesti_idx]
            #Now that we have all contours for all classes, build the aisle mask as the inverse of the union of all other masks
            class_mask[aisle_idx] = ~union #Generate the inverse
            #Check the mode to see if we need to merge anything
            if parsed.mode == "merge-partial": 
                class_mask[plots_idx] |= class_mask[pplots_idx] #Merge the partial plots into the plots
            elif parsed.mode == "merge-partial-bedding":
                class_mask[plots_idx] |= (class_mask[pplots_idx] | class_mask[bedding_idx]) #Merge the partial plots and bedding into the plots, and rename the 'plots' destination mask to LABEL_CRANBERRY_VEGETATION
            #Based on the mask file format option, determine how to generate the output mask files
            class_mask_subset = [e[1] for e in filter(lambda x: x[0], zip(output_indices,class_mask))]
            if parsed.format == 'split-binary':
                for i,m in enumerate(class_mask_subset):
                    packed = np.packbits(m, axis=-1)
                    np.savez_compressed(img_path+'.mask.packed.npz',packed)
            elif parsed.format == 'split-gray':
                for i,m in enumerate(class_mask_subset):
                    cv2.imwrite(img_path+'.mask.'+str(i)+'.png',m.astype('uint8')*255)
            else:
                #Convert class_mask to an array 
                m = np.array(class_mask_subset)
                if parsed.format == 'layer-binary':
                    packed = np.packbits(m, axis=-1)
                    np.savez_compressed(img_path+'.mask.packed.npz',packed)
                elif parsed.format == 'layer-gray': 
                    m = m.astype('uint8')*255 #Convert to grayscale
                    m.tofile(img_path+'.mask.unpacked.bin')
            #Generate a multi-channel mask image that can be viewed for confirmation.
            overlay,mchannel_mask = generateMaskOverlay(img, class_mask_subset, sse2mask_map_dict['new-objects'], parsed.alpha)
            cv2.imwrite(img_path+'.mask.png',overlay)
            cv2.imwrite(img_path+'.overlay.mask.png',mchannel_mask)
            #Write mapping parameters into the png overlay image.
            generateXMPMetadata(img_path+'.mask.png',sys.argv,sse2mask_map_json)
            generateXMPMetadata(img_path+'.overlay.mask.png',sys.argv,sse2mask_map_json)
        with open(parsed.map, 'w') as map_fh:
            map_fh.write(json.dumps(sse2mask_map_dict))
            map_fh.close()
