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
from scipy.interpolate import UnivariateSpline
import sys

class MatchHists:
    def __init__(self, num_bins=256, verbose=False, draw=False):
        '''Constructor for matching histograms.'''
        self.draw                               = draw
        self.verbose                            = verbose
        self.reference_cdf                      = None
        self.reference_hist                     = None
        self.source_cdf                         = None
        self.num_bins                           = num_bins
        self.reference_image_count              = 0 #An internal counter of total reference images added thus-far, for later calculating the average histogram
        self.reference_image_hist_accumulator   = np.zeros((self.num_bins), dtype=np.uint64) #An internal accumulator of histogram values as references images are pushed/added
        return

    def set_reference_pdf(self, pdf):
        '''Set the reference probability and cumulative distribution functions manually.  The cdf parameter is a numpy array (matrix).'''
        self.reference_hist   = pdf
        self.reference_cdf    = np.cumsum(pdf)
        #Also generate the inverse map anytime a new cdf is set.
        self.generate_inverse_cdf()
        return


    def push_reference(self, img):
        '''Adds an image to the reference image histogram accumulator.  NOTE: img must be a single-channel, gray-scale/intensity image for this to work.'''
        (img_hist,img_bins)                     = np.histogram(img, bins=self.num_bins, range=(0,self.num_bins))
        img_hist                                = img_hist.astype(np.uint64)
        self.reference_image_hist_accumulator   += img_hist
        self.reference_image_count              += 1
        return


    def clear_reference(self):
        '''Clears out the reference image histogram accumulator.'''
        self.reference_image_count              = 0
        self.reference_image_hist_accumulator   = 0
        self.reference_cdf                      = None
        self.reference_hist                     = None


    def calc_reference_pdf(self):
        '''Using the reference histogram accumulator, calculate the average histogram and the new reference cdf.'''
        if( self.reference_image_count > 0 ):
            reference_hist       = self.reference_image_hist_accumulator/self.reference_image_count
            reference_hist       = reference_hist/sum(reference_hist) #Scale to a density distribution
            self.set_reference_pdf(reference_hist)
        return

    def gen_reference_cdf(self, imgs):
        '''Calculates a reference cumulative distribution function from a list of images.'''
        '''NOTE: The input image must be a single-channel of an image.'''
        for img in imgs:
            self.push_reference(img)
        self.calc_reference_pdf()
        return

    def set_source_cdf(self, cdf):
        '''Set the source cumulative distribution function.'''
        self.source_cdf = cdf
        return

    def map(self, img):
        '''Map a source image to target image by matching against the reference histogram.  NOTE: This requires that set_source_cdf() was previously called, and is a low-level routine that is wrapped by match()'''
        '''NOTE: The input image must be a single-channel of an image.'''
        #First convert input image to a set of cdf percentiles
        img_percentiles = self.source_cdf[img.flatten()]
        new_img         = self.reference_icdf(img_percentiles)
        #Now deal with clipping correctly
        new_img[new_img >= self.num_bins] = self.num_bins-1
        new_img[new_img < 0] = 0
        new_img         = new_img.astype(img.dtype) #Convert to the same as input img type
        new_img         = new_img.reshape(img.shape) #Reshape to the same as input img shape
        return(new_img)

    def match(self, img):
        '''Map a source image to target image by matching against the reference histogram.  This wraps the map() function and internally calculates the input image's histogram and cumulative distribution function first.'''
        '''NOTE: The input image must be a single-channel of an image.'''
        (img_hist,img_bins)   = np.histogram(img, bins=self.num_bins, range=(0,self.num_bins), density=True)
        self.set_source_cdf(np.cumsum(img_hist))
        new_img = self.map(img)
        if( self.draw ):
            #Show the reference histogram, the original histogram, and the new histogram
            subplt(3,1,1)
            plt.plot(np.arange(0,self.reference_hist.size,1),self.reference_hist,'b-')
            plt.title("Reference Histogram")
            plt.xlim(0, self.reference_hist.size)
            subplt(3,1,2)
            plt.plot(np.arange(0,img_hist.size,1),img_hist,'b-')
            plt.title("Source Image Histogram")
            plt.xlim(0, img_hist.size)
            (new_img_hist, new_img_bins) = np.histogram(new_img, bins=self.num_bins, range=(0,self.num_bins), density=True)
            subplt(3,1,3)
            plt.plot(np.arange(0,new_img_hist.size,1),new_img_hist,'g-')
            plt.title("Target Image Histogram")
            plt.xlim(0, new_img_hist.size)
            plt.show()
        return (new_img)

    def generate_inverse_cdf(self, spline_smooth=1):
        '''Internally used to find an optimal regression estimator for generating the inverse cdf.'''
        #Cap the cdf to ensure that the spline is correctly extrapolating in case cdf doesn't start at zero
        ina    = np.zeros(self.reference_cdf.size+1)
        ina[0] = 0.0
        ina[1:self.reference_cdf.size+1] = self.reference_cdf
        ina    = self.make_strictly_increasing(ina) #Input values are the reference cdf percentiles
        outa   = np.arange(-1,self.reference_cdf.size,1,dtype=np.int32) #Output values are the pixel values
        self.reference_icdf = UnivariateSpline(ina, outa, k=3, s=3)
        if( self.draw ):
            plt.scatter(ina, outa, c='r')
            input_range = np.linspace(0.0,1.0,2000) #Move half a percent at a time
            output_range = self.reference_icdf(input_range)
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
    parser.add_argument('-o', '--output', action='store', required=True, help="The output target image with adjusted histogram matching the reference(s).")
    parser.add_argument('-r', '--reference', action='append', nargs=1, type=str, required=True, help="The input reference image(s) used to generate a reference histogram.")
    parser.add_argument('-d', '--draw', action='store_true', help="Whether to draw histograms of reference image, source and target images, and to show inverse CDF interpolation function.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

if __name__ == '__main__':
    parsed = parse_args()
    matchHist = MatchHists(draw=parsed.draw)
    img = cv2.imread(parsed.input, cv2.IMREAD_GRAYSCALE)
    if( parsed.draw ):
        cv2.imshow('Original image.', img)
        cv2.waitKey(0)
    references = [r[0] for r in parsed.reference]
    for ref in references:
        ref_img = cv2.imread(ref, cv2.IMREAD_GRAYSCALE)
        if( parsed.draw ):
            cv2.imshow('Ref Image %s' % (ref), ref_img)
            cv2.waitKey(0)
        matchHist.push_reference(ref_img)
    matchHist.calc_reference_pdf()
    new_img = matchHist.match(img)
    if( parsed.draw ):
        cv2.imshow('Recalibrated image.', new_img)
        cv2.waitKey(0)
    cv2.imwrite(parsed.output, new_img)
