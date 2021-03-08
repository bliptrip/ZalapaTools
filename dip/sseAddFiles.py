#!/usr/bin/env python3
# Author: Andrew Maule
# Date: 2020-05-08
# Objective: Adds all *.jpg files in a subdirectory to the mongodb SseSamples document, adding tags as specified to each document entry.
#
#
# Parameters:
#   - Image path to search recursively for *.jpg files.
#   - Path to SSE settings file (JSON-formatted) to convert paths in image path relative to settings['configuration']['images-folder']
#   - Tags to add to each document.
#   - MongoDB hostname
#   - MongoDB port
#   - MongoDB database name
#   
#
# Algorithm overview:
# Converts the folder and the url for the SseSample collection items based on what was the 'old' root folder and the 'new' root folder.  A mongodb filter string can be
# supplied to only selectively apply changes. 
#

import argparse
from datetime import datetime
from glob import glob
import json
import os
from pathlib import PurePath
from pymongo import MongoClient
import re
import sys
import urllib

# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility to batch-add (through glob pattern) a set of images to SSE collection 'SseSample' meteor MongoDB, for the purpose of allowing users to annotate.  The script allows the caller to specify 'tags' that can provide filtered access to images.")
    parser.add_argument('-i', '--image_path', dest='image_path', action='store', required=True, help="Full path to location of image files for adding to the Semantic Segmentation Editor.")
    parser.add_argument('-s', '--settings', dest='settings', action='store', required=True, help="Full path to the Semantic Segmentation Editor settings.json file.")
    parser.add_argument('-g', '--glob', dest='glob', action='store', default='*.JPG', help="Glob pattern used to search for image files with image_path.")
    parser.add_argument('--hostname', dest='hostname', action='store', default='localhost', help="The MongoDB hostname/ip to connect to.")
    parser.add_argument('-p', '--port', action='store', type=int, default=3001, help="The MongoDB port to connect to.")
    parser.add_argument('-d', '--db', action='store', default="meteor", help="MongoDB database name to access.")
    parser.add_argument('--soc_name', action='store', dest='soc', default="Drone Berry Development", help="The name of the class set to use in settings.json.")
    parser.add_argument('-t', '--tags', action='append', default=['toannotate'], help="Add tag to the tags field in the SseSamples document entry for each added image.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)


if __name__ == '__main__':
    parsed      = parse_args()
    client      = MongoClient(parsed.hostname, parsed.port)
    db          = client[parsed.db]
    image_path  = parsed.image_path
    extensions  = parsed.glob
    settings    = json.loads(open(parsed.settings, 'r').read())
    root_path   = settings['configuration']['images-folder']
    sse_samps   = db.SseSamples
    soc         = parsed.soc
    tags        = parsed.tags
    image_files   = glob(image_path+"/**/"+extensions, recursive=True)
    for image_file in image_files:
        image_file_path     = PurePath(image_file)
        image_relative_path = PurePath('/' + str(image_file_path.relative_to(root_path)))
        url_path  = '/' + urllib.parse.quote(str(image_relative_path), safe='')
        result = sse_samps.insert_one({ 'url': url_path,
                                        'socName': soc,
                                        'firstEditDate': datetime.utcnow(),
                                        'lastEditDate': datetime.utcnow(),
                                        'folder': str(image_relative_path.parent),
                                        'objects': [],
                                        'tags': tags,
                                        'file': str(image_relative_path.name) })
