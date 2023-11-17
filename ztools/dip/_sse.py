#!/usr/bin/env python3
# Author: Andrew Maule
# Date: 2019-01-21
# Objective: Helper functions for formatting paths/urls for the semantic-segmentation-editor and other shared functionality around this.
#

import copy
import cv2
import imutils
import numpy as np
import os
from pathlib import Path
import re
from skimage import img_as_float
import sys
import urllib
from pymongo import MongoClient

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

def findClassIndex(drone_class_objects,name):
    index               = -1
    for i,dco in drone_class_objects:
        if name == dco['label']:
            index = i
            break
    return index

def findClassName(drone_class_objects,index):
    name = None
    for i,dco in drone_class_objects:
        if index == i:
            name = dco['label']
            break
    return name

def generateMaskOverlay(img, masks, drone_class_objects, alpha=0.4):
    newimg        = np.zeros(img.shape, dtype='uint8')
    overlay       = np.zeros(img.shape, dtype='uint8')
    for i,mask in enumerate(masks):
        h = drone_class_objects[i][1]['color'].lstrip('#')
        bgr = tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))
        overlay[mask,:] = bgr
    cv2.addWeighted(overlay, alpha, img, 1.0-alpha, 0.0, newimg)
    return((overlay,newimg))
        

def getClassIndex(o):
    return o['classIndex']

class SSEObject():
    def __init__(self,parent,object_dict,label,color):
        self._parent = parent #Parent annotation
        self._o = object_dict
        self._label = label #Taken from settings file
        self._color = color #Taken from settings file

    @property
    def label(self):
        return(self._label)

    @property
    def color(self):
        return(self._color)

    @property 
    def classIndex(self):
        return(self._o['classIndex'])
    @classIndex.setter
    def classIndex(self,ci):
        self._o['classIndex'] = ci

    @property 
    def layer(self):
        return(self._o['layer'])
    @layer.setter
    def layer(self,layerIndex):
        self._o['layer'] = layerIndex

    @property
    def contours(self):
        _contours = np.array([[np.round(p['x']),np.round(p['y'])] for p in self._o['polygon']], dtype='int32')
        return(_contours)
    @contours.setter
    def contours(self,newContours):
        self._o['polygon'] = [{'x':c[0], 'y':c[1]} for c in newContours]
    
    @property
    def contoursRelative(self):
        _relative = np.apply_along_axis(lambda a: a - np.min(a), 0, self.contours)
        return(_relative)

    @property
    def shape(self):
        r = self.contoursRelative
        return(np.max(r, axis=0).tolist())

    @property
    def bbox(self):
        r = self.contours
        return(np.min(r, axis=0).tolist() + np.max(r, axis=0).tolist())

    @property
    def image(self):
        i = np.zeros(self.shape, dtype='uint8')
        cv2.drawContours(i, [self.contoursRelative], 0, color=255, thickness=2)
        return(img_as_float(i))



class SSEAnnotation():
    def __init__(self,collection,annotation):
        self._a = annotation
        self.collection = collection
        self._objects = [SSEObject(self,_o,collection.classLabel(_o['classIndex']),collection.classIndex2Color(_o['classIndex'])) for _o in annotation['objects']]

    @property
    def objects(self):
        return(self._objects)

    def query(self, classLabel):
        res = self._objects.filter(lambda o: o.label == classLabel)
        return(res)

    def labels(self):
        res = [o.label for o in self._objects]
        return(res)

    @property
    def id(self):
        return(self._a['_id'])

    @property
    def socName(self):
        return(self._a['socName'])
    @socName.setter
    def socName(self,newSocName):
        self._a['socName'] = newSocName

    @property
    def tags(self):
        return(self._a['tags'])
    @tags.setter
    def tags(self,newTags):
        self._a['tags'] = newTags
    def addTag(self, aTag):
        self._a.append(aTag)
    def rmTag(self, rTag):
        del self._a[rTag]

    @property
    def folder(self):
        return(self._a['folder'])
    @folder.setter
    def folder(self, f):
        f = '/' + f.strip('/')
        self._a['folder'] = f
        self.url = f + '/' + self.file

    @property
    def file(self):
        return(self._a['file'])
    @file.setter
    def file(self,f):
        f = f.strip('/')
        self._a['file'] = f
        self.url = self.folder + '/' + f

    @property
    def rawfilepath(self):
        return(self.collection.root.rstrip('/') + self.folder + '/' + self.file)

    @rawfilepath.setter
    def rawfilepath(self,filepath):
        p = re.compile("("+self.collection.root+")(.+)")
        m = p.match(filepath)
        if m:
            self.folder = os.path.dirname('/'+m.group(2))
            self.file = os.path.basename('/'+m.group(2))
        else:
            self.folder = os.path.dirname(filepath)
            self.file = os.path.basename(filepath)
        
    @file.setter
    def file(self,f):
        f = f.strip('/')
        self._a['file'] = f
        self.url = self._a['folder'] + '/' + f

    @property
    def url(self):
        return(self._a['url'])
    @url.setter
    def url(self,relpath):
        self._a['url'] = '/'+urllib.parse.quote(relpath, safe='')

    def update(self):
        self.collection.update({'url': sse_samp['url']}, self._a, upsert=False)

    def insert(self):
        self.collection.insert_one(self._a)


class SSECollection():
    def __init__(self,classSettings={},host="localhost",port=27017,db="meteor",rootpath="/mnt/external"):
        client = MongoClient(host, port)
        assert(client is not None, "Failed to connect to mongodb://{}:{}.".format(host,port))
        cdb = client[db]
        self.collection = cdb.SseSamples
        self.classSettings = classSettings
        self.classLabels2Colormap = dict([(o['label'],o['color']) for o in classSettings['objects']])
        self.classLabels2Index = dict([(o['label'],i) for i,o in enumerate(classSettings['objects'])])
        self.annotations = []
        self.root = rootpath

    def query(self, tags):
        self.annotations = [SSEAnnotation(self,_a) for _a in self.collection.find({"tags": {"$in": tags}})]
        return(self.annotations)

    def find(self, search):
        self.annotations = [SSEAnnotation(self,_a) for _a in self.collection.find(search)]
        return(self.annotations)

    def classIndex(self,label):
        return(self.classLabels2Index[label])

    def classColor(self,label):
        return(self.classLabels2Colormap[label])

    def classIndex2Color(self,index):
        return(self.classSettings['objects'][index]['color'])

    def classLabel(self,index):
        return(self.classSettings['objects'][index]['label'])

