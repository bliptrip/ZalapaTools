#!/usr/bin/env python
#Author: Andrew Maule
#Purpose: To wrap/streamline the color correction process using the class helpers that handle individual steps.

import argparse
import csv
import cv2 as cv
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import basename,dirname
from pathlib import Path
import re
import sys

sys.path.append(dirname(__file__))
from cc_colormap_generator import CCColormapGenerator
from cc_colors_normalize import CCColorsNormalize
from cc_model_generator import CCModelGenerator

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Correct colors for a group of images, using an input configuration file.")
    parser.add_argument('-p', '--path', action='store', required=True, help="The full path to the config file and also the location of the auxiliary metadata files and images.")
    parser.add_argument('-c', '--config', action='store', required=True, help="Input configuration file containing expected RGB/CIE Lab color values.")
    parser.add_argument('-t', '--cc_template_path', action='store', required=True, help="The full path to the color card template map files.")
    parser.add_argument('-e', '--extensions', action='store', default='jpg,JPG', help="Comma-separated list of extensions to search for images when generating an include file list.")
    parser.add_argument('-o', '--output', action='store', required=True, help="The output path to create to put the color converted images into.")
    parser.add_argument('-d', '--debug', action='store_true', help="Show debug output.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

if __name__ == '__main__':
    cc_image_dict = {}
    parsed = parse_args()
    #Preparse the config file and generate a structure more amenable to building our color map file, the model, the include paths for all images, and the trained models
    with open(parsed.config, 'r') as config_fh:
        configCSV = csv.DictReader(config_fh)
        for row in configCSV:
            session           = row['session']
            cc_image          = row['cc_image']
            cc_card_template  = row['cc_card_template']
            cc_mask_json      = row['cc_mask_json']
            cc_map_file       = row['cc_map_file']
            cc_include_images = row['cc_include_images']
            trained_model     = row['trained_model']
            model_flags       = row['model_flags']
            if not cc_image in cc_image_dict.keys():
                cc_image_dict[cc_image] = {'session':[], 'cc_card_template':"", 'cc_mask_json':"", 'cc_map_file':"", 'cc_include_images':"", 'trained_model':"", 'model_flags':""}
            cc_image_dict[cc_image]['session'].append(session)
            if cc_image_dict[cc_image]['cc_card_template'] == "":
                cc_image_dict[cc_image]['cc_card_template'] = cc_card_template
            if cc_image_dict[cc_image]['cc_mask_json'] == "":
                cc_image_dict[cc_image]['cc_mask_json'] = cc_mask_json
            if cc_image_dict[cc_image]['cc_map_file'] == "":
                cc_image_dict[cc_image]['cc_map_file'] = cc_map_file
            if cc_image_dict[cc_image]['cc_include_images'] == "":
                cc_image_dict[cc_image]['cc_include_images'] = cc_include_images
            if cc_image_dict[cc_image]['trained_model'] == "":
                cc_image_dict[cc_image]['trained_model'] = trained_model
            if cc_image_dict[cc_image]['model_flags'] == "":
                cc_image_dict[cc_image]['model_flags'] = model_flags
        config_fh.close()
    #Build the maps if they do not already exist
    for cc_image_k in cc_image_dict.keys():
        cc_image        = cc_image_dict[cc_image_k]
        cc_map_file     = cc_image['cc_map_file']
        cc_map_path     = Path(parsed.path+'/'+cc_map_file)
        #Check that the json file exists (with the ROI masks)
        cc_json         = cc_image['cc_mask_json']
        cc_json_path    = Path(parsed.path+'/'+cc_json)
        assert cc_json_path.is_file(), "ERR: JSON file %s does not exist." % (str(cc_json_path))
        #Check that the colorcard template file exists
        cc_template             = cc_image['cc_card_template']
        cc_card_template_path   = Path(parsed.cc_template_path+'/'+cc_template)
        assert cc_card_template_path.is_file(), "ERR: CC Template File %s does not exist." % (str(cc_card_template_path))
        #Check that the image with the colorcard exists
        cc_image_k_path = Path(parsed.path+'/'+cc_image_k)
        assert cc_image_k_path.is_file(), "ERR: CC Image File %s does not exist." % (str(cc_image_k_path))
        #Now that we've validated that we have what we need, let's continue
        
        #If the map path does not exist, then generate it.
        if not cc_map_path.is_file():
            cc_map_gen = CCColormapGenerator(debug=parsed.debug)
            cc_map_gen.load_json(str(cc_json_path))
            cc_map_gen.load_img(str(cc_image_k_path))
            cc_map_gen.contours()
            cc_map_gen.means()
            cc_map_gen.write(str(cc_card_template_path),str(cc_map_path))
        #If an includes file is specified and it does not exist, generate it.
        cc_includes_path = Path(parsed.path+'/'+cc_image['cc_include_images'])
        if not cc_includes_path.is_file():
            include_files = [str(cc_image_k_path)]
            for session in cc_image['session']:
                include_file_path = parsed.path+'/'+session+'/'
                for ext in parsed.extensions.split(','):
                    include_files.extend(glob.glob(include_file_path+'*.'+ext))
            cc_includes_fh = open(str(cc_includes_path),'w')
            assert cc_includes_fh, "ERR: Failed to open %s" % (str(cc_includes_path))
            for include_file in include_files:
               cc_includes_fh.write(include_file+'\n')
            cc_includes_fh.close()
        #If the model does not exist, generate it.  If there is an includes file listed but it does not exis
        cc_trained_model_path = Path(parsed.path+'/'+cc_image['trained_model'])
        if not cc_trained_model_path.is_file():
            cc_model_gen = CCModelGenerator()
            model_gen_string = "cc_model_gen.generate_models(mmap=str(cc_map_path),output=str(cc_trained_model_path),mdebug=parsed.debug,include_images=str(cc_includes_path)"
            model_flags = cc_image['model_flags']
            if(model_flags):
                #Replace semicolons with commas (semicolons used to not mix up csv format)
                model_flags = re.sub(";",",",model_flags)
                model_gen_string += ','+model_flags+')'
            else:
                model_gen_string += ')'
            eval(model_gen_string) 
        #Now that we've generated everything needed, we can do the color correction on all files in session path
        for session in cc_image['session']:
            session_files     = []
            session_files_path = parsed.path+'/'+session+'/'
            session_files_out_path = Path(parsed.path+'/'+session+'/'+parsed.output+'/')
            if not session_files_out_path.is_dir():
                try:
                    os.mkdir(str(session_files_out_path))
                except OSError:  
                    print ("Creation of the directory %s failed" % str(session_files_out_path))
                    sys.exit()
            for ext in parsed.extensions.split(','):
                session_files.extend(glob.glob(session_files_path+'*.'+ext))
            for session_file in session_files:
                cc_colors_norm = CCColorsNormalize(models_filename=str(cc_trained_model_path))
                img_output = str(session_files_out_path)+'/'+basename(session_file)
                cc_colors_norm.normalize(img_input=session_file,img_output=img_output,mdebug=parsed.debug)
