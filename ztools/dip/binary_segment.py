#!/usr/bin/env python3
#
# Author: Andrew Maule
# Objective: Simple binary-threshold method for segmenting foreground from background.
#
import cv2
from .segment import Segment

class BinaryThresholdSegment(Segment):
    def __init__(self, channel=2, threshold=100, **kwargs):
        super().__init__(**kwargs)
        self.channel    = channel #Channel to threshold on
        self.threshold  = threshold #Threshold value - values less than this are considered 'foreground' pixels
        return

    def predict(self,image):
        image = super().preprocess(image) #Handle global segment preprocessing on image before doing binary classification
        image = (image[:,:,self.channel] < self.threshold)
        return(super().postprocess(image))
