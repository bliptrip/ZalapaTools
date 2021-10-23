#!/usr/bin/env python3
# Author: Andrew Maule
# Date: 2019-01-21
# Objective: Helper functions for formatting paths/urls for the semantic-segmentation-editor and other shared functionality around this.
#

import cv2
import imutils
import os
from pathlib import Path
import re
import urllib

def generateSSEFolder(sse_root_path, file_full_path):
    #Strip path prefix from filename
    p = re.compile("("+sse_root_path+")(.+)")
    m = p.match(file_full_path)
    if m:
        folder = os.path.dirname('/'+m.group(2))
    else:
        folder = sse_root_path
    return(folder)

def generateSSEURL(sse_root_path, file_full_path):
    folder = generateSSEFolder(sse_root_path, file_full_path)
    filename = os.path.basename(file_full_path)
    url_path  = '/'+urllib.parse.quote(str(Path(folder + '/' + filename)), safe='')
    return(url_path)

def extractContours(mask, contours_sse, chain_method, classIndex=0, layer=0):
    contours  = cv2.findContours(mask.astype('uint8'), cv2.RETR_LIST, eval('cv2.'+chain_method))
    contours  = imutils.grab_contours(contours)
    contours_sse.extend([{"classIndex": classIndex, "layer": layer, "polygon": [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in polygon]} for polygon in contours])
    return(contours)

