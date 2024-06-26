#!/usr/bin/env python
# Author: Andrew Maule
# Objective: To match a given image's histogram to another set of images.  Also contains support functions for generating an average histogram from a list of image files,
# and more.
#
# Algorithm overview:
#  - Generate cdf's of both the reference image (or from list of image's average histogram) and an image to match against the reference (source image).  
#  - Match source image cdf to reference cdf.  To do this, generate a map of source pixel value -> source cdf percentile -> reference cdf percentile -> target pixel value.


import argparse
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot as subplt
import numpy as np
import pickle
from scipy.interpolate import UnivariateSpline
import sys

class MatchHists:
    def __init__(self, num_bins=256, nchannels=1, verbose=False, draw=False):
        '''Constructor for matching histograms.'''
        self.draw                               = draw
        self.verbose                            = verbose
        self.num_bins                           = num_bins
        self.nchannels                          = nchannels #Number of channels to calculate histogram over
        self.source_cdf                         = [None] * nchannels
        self.reference_cdf                      = np.zeros((num_bins,nchannels), dtype=np.float64)
        self.reference_icdf                     = [None] * nchannels
        self.reference_hist                     = np.zeros((num_bins,nchannels), dtype=np.float64)
        self.reference_image_count              = 0 #An internal counter of total reference images added thus-far, for later calculating the average histogram
        self.reference_image_hist_accumulator   = np.zeros((num_bins,nchannels), dtype=np.uint64) #An internal accumulator of histogram values as references images are pushed/added
        return

    def set_reference_pdf(self, pdf):
        '''Set the reference probability and cumulative distribution functions manually.  The pdf parameter is a numpy array of size (num_bins,nchannels)'''
        self.reference_hist[:,:] = pdf
        self.reference_cdf[:,:]  = np.cumsum(pdf, axis=0)
        #Also generate the inverse map anytime a new cdf is set.
        self.generate_inverse_cdf()
        return


    def push_reference(self, img, ma=None):
        '''Adds an image to the reference image histogram accumulator.'''
        if type(ma) != type(None):
            img = np.ma.masked_array(img,mask=~ma)
        for i in range(0,self.nchannels):
            if( self.nchannels > 1 ):
                cimg = img[:,:,i]
            else:
                cimg = img[:,:]
            if type(ma) != type(None):
                cimg = cimg.compressed() #This only works on masked arrays
            (img_hist,img_bins)                         = np.histogram(cimg, bins=self.num_bins, range=(0,self.num_bins))
            img_hist                                    = img_hist.astype(np.uint64)
            self.reference_image_hist_accumulator[:,i]  += img_hist
        self.reference_image_count += 1
        return


    def clear_reference(self):
        '''Clears out the reference image histogram accumulator.'''
        self.reference_image_count                  = 0
        self.reference_image_hist_accumulator[:,:]  = 0
        self.reference_cdf[:,:]                     = 0
        self.reference_hist[:,:]                    = 0


    def calc_reference_pdf(self):
        '''Using the reference histogram accumulator, calculate the average histogram and the new reference cdf.'''
        if( self.reference_image_count > 0 ):
            reference_hist       = self.reference_image_hist_accumulator/self.reference_image_count
            reference_hist       = reference_hist/np.sum(reference_hist,axis=0) #Scale to a density distribution
            self.set_reference_pdf(reference_hist)
        return

    def gen_reference_cdf(self, imgs):
        '''Calculates a reference cumulative distribution function from a list of images.'''
        for img in imgs:
            self.push_reference(img)
        self.calc_reference_pdf()
        return

    def set_source_cdf(self, cdf):
        '''Set the source cumulative distribution function.'''
        self.source_cdf = cdf
        return

    def map(self, img, ma=None):
        '''Map a source image to target image by matching against the reference histogram.  NOTE: This requires that set_source_cdf() was previously called, and is a low-level routine that is wrapped by match()'''
        if type(ma) != type(None):
           img = np.ma.masked_array(img,mask=~ma)
        new_img = [None] * self.nchannels
        #First convert input image to a set of cdf percentiles
        for i in range(0, self.nchannels):
            if( self.nchannels > 1 ):
                cimg = img[:,:,i]
            else:
                cimg = img
            if type(ma) != type(None):
                cimg_flat = cimg.compressed()
            else:
                cimg_flat = cimg.flatten()
            img_percentiles = self.source_cdf[i][cimg_flat]

            new_img[i]  = self.reference_icdf[i](img_percentiles)
            #Now deal with clipping correctly
            new_img[i][new_img[i] >= self.num_bins] = self.num_bins-1
            new_img[i][new_img[i] < 0] = 0
            new_img[i]         = new_img[i].astype(img.dtype) #Convert to the same as input img type
            if type(ma) != type(None):
                np.place(img[:,:,i], ~img.mask[:,:,i], new_img[i])
                new_img[i]         = img[:,:,i]
            else:
                new_img[i]         = new_img[i].reshape(img.shape[0:2]) #Reshape to the same as input img shape
        #Now combine all three channels
        return(np.stack(new_img, axis=-1))

    def match(self, img, ma=None):
        '''Map a source image to target image by matching against the reference histogram.  This wraps the map() function and internally calculates the input image's histogram and cumulative distribution function first.'''
        source_cdf = [None] * self.nchannels
        img_hist = [None] * self.nchannels
        if type(ma) != type(None):
            img = np.ma.masked_array(img, mask=~ma)
        for i in range(0, self.nchannels):
            if( self.nchannels > 1 ):
                cimg = img[:,:,i]
            else:
                cimg = img
            if type(ma) != type(None):
                cimg = cimg.compressed() #This only works on masked arrays
            (img_hist[i],img_bins) = np.histogram(cimg, bins=self.num_bins, range=(0,self.num_bins), density=True)
            source_cdf[i] = np.cumsum(img_hist[i])
        self.set_source_cdf(source_cdf)
        new_img = self.map(img,ma)
        if( self.draw ):
            #Show the reference histogram, the original histogram, and the new histogram
            for i in range(0,self.nchannels):
                subplt(3, self.nchannels, i + 1)
                plt.plot(np.arange(0,(self.reference_hist[:,i]).size,1),self.reference_hist[:,i],'b-')
                plt.title("Reference Histogram")
                plt.xlim(0, (self.reference_hist[:,i]).size)
                subplt(3, self.nchannels, self.nchannels + i + 1) 
                plt.plot(np.arange(0,img_hist[i].size,1),img_hist[i],'b-')
                plt.title("Source Image Histogram")
                plt.xlim(0, img_hist[i].size)
                (new_img_hist, new_img_bins) = np.histogram(new_img[:,:,i], bins=self.num_bins, range=(0,self.num_bins), density=True)
                subplt(3,self.nchannels, (2*self.nchannels) + i + 1)
                plt.plot(np.arange(0,new_img_hist.size,1),new_img_hist,'g-')
                plt.title("Target Image Histogram")
                plt.xlim(0, new_img_hist.size)
                plt.show()
        if( self.nchannels == 1 ):
            new_img = new_img[:,:,0] #Remove last dimension, as it's superfluous
        return (new_img)

    def generate_inverse_cdf(self, spline_smooth=1):
        '''Internally used to find an optimal regression estimator for generating the inverse cdf.'''
        #Cap the cdf to ensure that the spline is correctly extrapolating in case cdf doesn't start at zero
        for i in range(0,self.nchannels):
            ina    = np.zeros(self.reference_cdf[:,i].size+1)
            ina[0] = 0.0
            ina[1:self.reference_cdf[:,i].size+1] = self.reference_cdf[:,i]
            ina    = self.make_strictly_increasing(ina) #Input values are the reference cdf percentiles
            outa   = np.arange(-1,self.reference_cdf[:,i].size,1,dtype=np.int32) #Output values are the pixel values
            self.reference_icdf[i] = UnivariateSpline(ina, outa, k=3, s=3)
            if( self.draw ):
                plt.scatter(ina, outa, c='r')
                input_range = np.linspace(0.0,1.0,2000) #Move half a percent at a time
                output_range = self.reference_icdf[i](input_range)
                plt.plot(input_range, output_range, 'b-')
                plt.title("CDF and spline interpolation overlay.")
                plt.ylim(0.0,255.0)
                plt.xlim(0.0,1.0)
                plt.show()
        return(self.reference_icdf)

    def make_strictly_increasing(self, x, adjust=1e-10):
        '''Scipy's UnivariateSpline has a strictly incrasing requirement for its elements.  Because some of the entries in the cdf are equal to previous entries,'''
        ''' this rule is violated.  Add some small factor (1e-10) to those entries that are equivalent to get around this annoying problem.'''
        for i in range(1, x.size):
            if( x[i] <= x[i-1] ):
                x[i] = x[i-1] + adjust
        return(x)

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line wrapper to functionality in the matchHist package.  This utility allows one to test matching the histograms of two images.")
    parser.add_argument('-i', '--input', action='store', required=True, help="The source image to adjust image histogram to match the reference(s).")
    parser.add_argument('-o', '--output', action='store', help="The output target image with adjusted histogram matching the reference(s).")
    parser.add_argument('-d', '--draw', action='store_true', help="Whether to draw histograms of reference image, source and target images, and to show inverse CDF interpolation function.")
    parser.add_argument('-s', '--save', action='store', help="Save the calculated reference pdf/cdfs to a file for retrieval")
    parser.add_argument('-l', '--load', action='store_true', help="If flag set, the reference file should be the result of a previous save operation.")
    parser.add_argument('reference', nargs="+", type=str, help="The input reference image(s) used to generate a reference histogram.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

if __name__ == '__main__':
    parsed = parse_args()
    img = cv2.imread(parsed.input)
    if( parsed.draw ):
        cv2.imshow('Original image.', img)
        cv2.waitKey(0)
    if len(img.shape) == 2:
        nchannels = 1
    else:
        nchannels = img.shape[-1]
    if parsed.load:
        histFileObj = open(parsed.reference[0],'rb')
        matchHist   = pickle.load(histFileObj)
        histFileObj.close()
    else:
        matchHist = MatchHists(draw=parsed.draw, nchannels=nchannels)
        for ref in parsed.reference:
            ref_img = cv2.imread(ref)
            if( parsed.draw ):
                cv2.imshow('Ref Image %s' % (ref), ref_img)
                cv2.waitKey(0)
            matchHist.push_reference(ref_img)
        matchHist.calc_reference_pdf()
    new_img = matchHist.match(img)
    if( parsed.draw ):
        cv2.imshow('Recalibrated image.', new_img)
        cv2.waitKey(0)
    if( parsed.output ):
        cv2.imwrite(parsed.output, new_img)
    if( parsed.save ):
        histFileObj = open(parsed.save, 'wb')
        pickle.dump(matchHist, histFileObj)
        histFileObj.close()
