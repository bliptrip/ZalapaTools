#!/usr/bin/env python3
#
# Author: Andrew Maule
# Objective: Abstract base classes for image segmentation.
#
from abc import ABC, abstractmethod
import cv2
#import imutils
from math import pi
import numpy as np
from numpy import arcsin as asin, sqrt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelextrema
from skimage.color import rgb2gray
from skimage.filters.rank import mean
from skimage.segmentation import clear_border, find_boundaries, mark_boundaries
from skimage.measure import label, regionprops_table, regionprops
from skimage.morphology import closing, disk, flood_fill
from skimage.transform import radon #Radon projection transform, for determining the number of columns, and thus the number of column bins
from sklearn.cluster import AgglomerativeClustering

columns = [ 'LvsW',
            'blobLength',
            'blobWidth',
            'projectedArea',
            'projectedPerimeter',
            'skinSurface',
            'blobVolume',
            'blobEccentricity',
            'blobSolidity',
            'bwColor',
            'vbwColor',
            'LvsW_r',
            'blobLength_r',
            'blobWidth_r',
            'projectedArea_r',
            'projectedPerimeter_r',
            'skinSurface_r',
            'blobVolume_r',
            'blobEccentricity_r',
            'blobSolidity_r',
            'bwColor_r',
            'vbwColor_r',
            'blobLength_r2',
            'blobWidth_r2',
            'projectedArea_r2',
            'projectedPerimeter_r2',
            'skinSurface_r2',
            'blobVolume_r2',
            'accuracy',
            'locationX',
            'locationY',
            'R_med',
            'G_med',
            'B_med',
            'R_var',
            'G_var',
            'B_var']

dtypes  = pd.Series(['float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64',
                     'float64'],index=columns)


class Segment():
    def __init__(self, resize=0.5, minArea=1500, maxArea=3000, numRefs=12, refSize=2.54, grid=False, grid_num_columns=0):
        super().__init__()
        self.resize  = resize
        self.minArea = minArea
        self.maxArea = maxArea
        self.numRefs = numRefs
        self.refSize = refSize
        self.grid    = grid
        self.num_columns = grid_num_columns
        return

    def preprocess(self, image):
        image = cv2.resize(image, (0,0), fx=self.resize, fy=self.resize) 
        avg = np.zeros(image.shape, dtype='uint8')
        for c in range(image.shape[2]):
            avg[:,:,c] = mean(image[:,:,c], disk(4))
        return(avg)

    def postprocess(self, binimage):
        numOnes = np.where(binimage)[0].size
        numZeros = binimage.size - numOnes
        if( numOnes > numZeros ):
            binimage = ~binimage #Invert -- Why, I'm not sure, but this was in original GiNA code -- maybe had to do with putting green berries on black background
        binimage = closing(binimage, disk(3)) #Luis did this image postprocessing to remove noise
        #bwareaopen doesn't have a direct implementation.  Could just fill in 'holes', and then remove anything that
        #is below a given size in the segmention stage
        holes = binimage.copy().astype('uint8')*255
        holes = flood_fill(holes, (0,0), 255)
        # invert holes mask, img fill in holes
        holes = ~holes
        binimage = binimage | holes.astype('bool')
        return(clear_border(binimage))

    def segment(self, binimage, image=None):
        '''
        Returns a set of region properties on labeled image using skimage

        @param binimage: Binary image -- White pixels are foreground objects (berries, reference standards, etc.), black pixels are background
        @param image: Original image -- if not specified, then color parameters cannot be calculated
        '''
        image_boundaries = None
        if( image != None ):
            image = self.preprocess(image)
        if( (image == None) or (binimage.shape != image.shape[0:2]) ): #Need to check the following, b/c if I input Matlab-derived binary files to compare to the resized image, Matlab scales slightly differently (rounds instead of floor) than OpenCV2
            binimage = binimage.astype('uint8')*255
            if( self.resize != 1.0 ):
                if( image != None ):
                    binimage = cv2.resize(binimage, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) #Have to flip shape, as cv2 expects image convention (cols,rows), while numpy treats shape as (rows,cols)
                else:
                    binimage = cv2.resize(binimage, (0,0), fx=self.resize, fy=self.resize, interpolation=cv2.INTER_NEAREST)
            binimage = binimage.astype('bool')
        label_image = label(binimage)
        #Extract regions/blobs
        regions_raw = [r for r in filter(lambda r: (r.area >= self.minArea ) & (r.area <= self.maxArea), regionprops(label_image, cache=True))]
        regions_raw.sort(key = lambda r: r.centroid[1])
        regions = pd.DataFrame(regionprops_table(label_image, properties=['major_axis_length',
                                                                'minor_axis_length',
                                                                'area',
                                                                'perimeter',
                                                                'eccentricity',
                                                                'solidity',
                                                                'label',
                                                                'bbox',
                                                                'centroid']))
        regions.columns = ['major_axis_length',
                           'minor_axis_length',
                           'area',
                           'perimeter',
                           'eccentricity',
                           'solidity',
                           'label',
                           'bbox_minrow',
                           'bbox_mincol',
                           'bbox_maxrow',
                           'bbox_maxcol',
                           'centroid_row',
                           'centroid_col']
        #Filter out regions based on area (in square pixels)
        regions = regions[(regions.area >= self.minArea) & (regions.area <= self.maxArea)]
        regions.sort_values(by='centroid_col', inplace=True) #Sort blobs left to right
        regions.index = range(0, len(regions.index))
        #Also set the index of the regions_raw list, so as to sort it equivalently with regions later.
        for i,r in enumerate(regions_raw):
            r.index = i

        #If user specifies that berries are in a grid pattern, then use radon transform to determine 'peaks' in the y-direction
        if self.grid:
            binxdim = binimage.shape[0]
            binydim = binimage.shape[1]
            maxdim = max(binxdim, binydim)
            newbinimage = np.zeros((maxdim, maxdim), dtype='uint8') #Necessary to convert to square, as radon transform apparently shrinks to smallest dimension
            newbinimage[:binxdim,:binydim] = binimage * 255 #Copy the binimage into the square version of it
            regions['grid_bins']    = pd.Series(np.zeros(len(regions.index)), index=regions.index) #Create a grid_bins column for storing clustered grid columns
            regions_centroid_col    = np.asarray(regions.centroid_col).reshape(-1,1) #Needed to get AgglomerativeClustering working
            if( self.num_columns == 0 ): #Try to infer the number of columns of berries in grid layout
                grid_peaks              = np.squeeze(radon(newbinimage, theta=[0.0], circle=True, preserve_range=True)) #Project only at 0 degrees (down columns)
                grid_peaks_blur         = gaussian_filter(grid_peaks, sigma=12.0) #Gaussian blur the radon transform output in order to avoid spurious peaks from off-center berries in column
                grid_peak_extrema       = argrelextrema(grid_peaks_blur, np.greater)[0]
                grid_col_clustering     = AgglomerativeClustering(n_clusters=len(grid_peak_extrema)).fit(regions_centroid_col) #Sort blob centroids based on column offset
            else:
                grid_col_clustering     = AgglomerativeClustering(n_clusters=self.num_columns).fit(regions_centroid_col) #Sort blob centroids based on column offset
            #Reorder the labels_ from 0 to len(np.unique(labels_), as we want to keep sort
            new_label = -1 
            last_label = -1
            new_labels = np.array(grid_col_clustering.labels_, copy=True)
            for i, l in enumerate(grid_col_clustering.labels_):
                if( last_label != l ): #Update the new label
                    new_label += 1
                    last_label = l
                new_labels[i] = new_label
            regions.grid_bins = new_labels
            regions.sort_values(by=['grid_bins','centroid_row'], inplace=True)
            regions_index = tuple(regions.index)
            regions_raw.sort(key = lambda r: regions_index.index(r.index))
            #Now, make sure we resort the regions_raw to match the 
        regions.index = range(0,len(regions.index)) #Renumber the index from 0 to len(regions.index), since the dataframe will hold the original ordering b/f sorting

        datadf = pd.DataFrame(np.zeros((regions.shape[0],len(columns)),dtype='float64'), columns=columns) #Pre-allocate
        #Extract contours/boundaries of each blob
        datadf.LvsW               = regions.major_axis_length / regions.minor_axis_length
        datadf.blobLength         = regions.major_axis_length
        datadf.blobWidth          = regions.minor_axis_length
        datadf.projectedArea      = regions.area
        datadf.projectedPerimeter = regions.perimeter
        datadf.blobEccentricity   = regions.eccentricity
        datadf.blobSolidity       = regions.solidity
        datadf.locationX          = regions.centroid_col
        datadf.locationY          = regions.centroid_row
        A = 0.5 * regions.minor_axis_length #Assume that we are a prolate spheroid (elongated like an egg, not flattened, or oblate (like a rotating planet))
        B = A
        C = 0.5 * regions.major_axis_length 
        datadf.blobVolume         = ((4*pi)/3)*(A*B*C)
        #Orig: Failed to work b/c was getting the sqrt of negative numbers - datadf.skinSurface = (2*pi*A**2)+(pi*A)*((((B**2)/sqrt((B**2)-(A**2)))*acos(A/B))+(((C**2)/sqrt((C**2)-(A**2)))*acos(A/C)))
        E                           = sqrt(1 - ((A**2)/(C**2)))
        datadf.skinSurface        = (2*pi*(A**2))*(1 + (C/(A*E))+asin(E)) #Again, this is a formula assuming a prolate spheroid, which better defines most cranberry shapes
        #Calculate Color Stats for Each Blob if an original image is provided
        if( image != None ):
            for i,r in regions.iterrows():
                blobPixels                  = image[np.where(label_image == r.label)]
                B_med,G_med,R_med           = np.median(blobPixels, axis=0)
                datadf.loc[i, 'R_med']      = R_med
                datadf.loc[i, 'G_med']      = G_med
                datadf.loc[i, 'B_med']      = B_med
                B_var,G_var,R_var           = np.var(blobPixels, axis=0)
                datadf.loc[i, 'R_var']      = R_var
                datadf.loc[i, 'G_var']      = G_var
                datadf.loc[i, 'B_var']      = B_var
                blobPixelsGray              = 1.0-rgb2gray(blobPixels)/255.0
                datadf.loc[i, 'bwColor']    = np.sum(blobPixelsGray)/(r.area)
                datadf.loc[i, 'vbwColor']   = np.var(blobPixelsGray)*100.0
        num_blobs = len(regions.index)
        half_blobs = int(self.numRefs/2)
        fruit_index = np.array(range(half_blobs, num_blobs - half_blobs), dtype='int')
        #Create a boundaries overlay for displaying fruit blobs for debugging
        not_fruit_labels = np.ones(label_image.shape, dtype='bool')
        for i in fruit_index:
            r = regions_raw[i]
            not_fruit_labels[r.coords[:,0], r.coords[:,1]] = False
        fruit_label_image = label_image.copy()
        fruit_label_image[not_fruit_labels] = 0 #Mark as 0 to clear out other label entries
        if( image != None ):
            image_boundaries = (mark_boundaries(image, fruit_label_image, color=(0,0,1))*255).astype('uint8')
        fruit_boundaries = find_boundaries(fruit_label_image)

        reference_index = np.setdiff1d(np.array(range(0, num_blobs)), fruit_index)
        not_reference_labels = np.ones(label_image.shape, dtype='bool')
        for i in reference_index:
            r = regions_raw[i]
            not_reference_labels[r.coords[:,0], r.coords[:,1]] = False
        reference_label_image = label_image.copy()
        reference_label_image[not_reference_labels] = 0 #Mark as 0 to clear out other label entries
        if( image != None ):
            image_boundaries = (mark_boundaries(image_boundaries, reference_label_image, color=(1,0,0))*255).astype('uint8')
        reference_boundaries = find_boundaries(reference_label_image)

        #Normalize the features relative to the size standard, and to the 'real' size, if needed
        if( self.numRefs > 0 ):
            datadf.loc[fruit_index, 'LvsW_r'] = datadf.loc[fruit_index, 'LvsW']/np.median(datadf.loc[reference_index, 'LvsW'])
            datadf.loc[fruit_index, 'blobLength_r'] = datadf.loc[fruit_index, 'blobLength']/np.median(datadf.loc[reference_index, 'blobLength'])
            datadf.loc[fruit_index, 'blobWidth_r'] = datadf.loc[fruit_index, 'blobWidth']/np.median(datadf.loc[reference_index, 'blobWidth'])
            datadf.loc[fruit_index, 'projectedArea_r'] = datadf.loc[fruit_index, 'projectedArea']/np.median(datadf.loc[reference_index, 'projectedArea'])
            datadf.loc[fruit_index, 'projectedPerimeter_r'] = datadf.loc[fruit_index, 'projectedPerimeter']/np.median(datadf.loc[reference_index, 'projectedPerimeter'])
            datadf.loc[fruit_index, 'skinSurface_r'] = datadf.loc[fruit_index, 'skinSurface']/np.median(datadf.loc[reference_index, 'skinSurface'])
            datadf.loc[fruit_index, 'blobVolume_r'] = datadf.loc[fruit_index, 'blobVolume']/np.median(datadf.loc[reference_index, 'blobVolume'])
            datadf.loc[fruit_index, 'blobEccentricity_r'] = datadf.loc[fruit_index, 'blobEccentricity']/np.median(datadf.loc[reference_index, 'blobEccentricity'])
            datadf.loc[fruit_index, 'blobSolidity_r'] = datadf.loc[fruit_index, 'blobSolidity']/np.median(datadf.loc[reference_index, 'blobSolidity'])
            datadf.loc[fruit_index, 'bwColor_r'] = datadf.loc[fruit_index, 'bwColor']/np.median(datadf.loc[reference_index, 'bwColor'])
            datadf.loc[fruit_index, 'vbwColor_r'] = datadf.loc[fruit_index, 'vbwColor']/np.median(datadf.loc[reference_index, 'vbwColor'])
            
            digitalLength = np.median(datadf.blobLength[reference_index]);
            ratioConv=self.refSize/digitalLength;
            datadf.loc[fruit_index, 'blobLength_r2'] = datadf.loc[fruit_index, 'blobLength'] * ratioConv
            datadf.loc[fruit_index, 'blobWidth_r2'] = datadf.loc[fruit_index, 'blobWidth'] * ratioConv
            datadf.loc[fruit_index, 'projectedArea_r2'] = datadf.loc[fruit_index, 'projectedArea'] * (ratioConv**2)
            datadf.loc[fruit_index, 'projectedPerimeter_r2'] = datadf.loc[fruit_index, 'projectedPerimeter'] * ratioConv
            datadf.loc[fruit_index, 'skinSurface_r2'] = datadf.loc[fruit_index, 'skinSurface'] * (ratioConv**2)
            datadf.loc[fruit_index, 'blobVolume_r2'] = datadf.loc[fruit_index, 'blobVolume'] * (ratioConv**3)
            datadf.loc[reference_index, 'accuracy'] = datadf.loc[reference_index, 'blobLength'] * (ratioConv)

        return((datadf, fruit_index, reference_index, image_boundaries, fruit_boundaries, reference_boundaries))
