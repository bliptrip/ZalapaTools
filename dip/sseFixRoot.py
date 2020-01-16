#!/usr/bin/env python3
# Author: Andrew Maule
# Date: 2019-01-16
# Objective: Fixes the folder/URL designation for a set of files
#
#
# Parameters:
#   - Old root folder on local drive
#   - New root folder on local drive
#   - MongoDB hostname
#   - MongoDB port
#   - MongoDB Filter String
#   
#
# Algorithm overview:
# Converts the folder and the url for the SseSample collection items based on what was the 'old' root folder and the 'new' root folder.  A mongodb filter string can be
# supplied to only selectively apply changes. 
#

import argparse
import json
import os
from pathlib import Path
from pymongo import MongoClient
import re
import sys
import urllib

# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility for fixing SseSample paths in a meteor MongoDB.")
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument('--prepend', action='store', help="Prepend string to front of all matching items folder and url fields.")
    operation_group.add_argument('--strip', action='store', help="Strip leading string from front of all matching items folder and url fields.")
    parser.add_argument('--hostname', dest='hostname', action='store', default='localhost', help="The MongoDB hostname/ip to connect to.")
    parser.add_argument('-p', '--port', action='store', type=int, default=3001, help="The MongoDB port to connect to.")
    parser.add_argument('-d', '--db', action='store', default="meteor", help="MongoDB database name to access.")
    parser.add_argument('-s', '--search', '--filter', dest='filter', action='store', default="{}", help="MongoDB search string.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)


def generateURL(folder, filename):
    url_path  = '/'+urllib.parse.quote(str(Path(folder + '/' + filename)), safe='')
    return(url_path)


if __name__ == '__main__':
    parsed      = parse_args()
    client      = MongoClient(parsed.hostname, parsed.port)
    db          = client[parsed.db]
    sfilter     = json.loads(parsed.filter)
    sse_samps   = db.SseSamples.find(sfilter)
    for s in sse_samps:
        if parsed.prepend:
            s['folder'] = str(Path('/' + parsed.prepend + '/' + s['folder']))
        elif parsed.strip:
            s['folder'] = str(Path('/' + re.sub(parsed.strip, "", s['folder'])))
        s['url'] = generateURL(s['folder'], s['file'])
        db.SseSamples.replace_one({'_id': s['_id']}, s, upsert=False)
