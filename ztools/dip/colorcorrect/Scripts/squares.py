#!/usr/bin/env python

'''
Simple "Square Detector" program.

Loads several images sequentially and tries to find squares in each image.
'''

#General Strategy
# 1. Gradually decrease threshold from 255 for all color channels to find the single white 'square'.  Tag it as such to distinguish from other squares and to help orient.
# 2. Continue to look for other squares by thresholding over range at lower intensity values.  Generate a union polygon of all of the thresholded edges over their range.
# 3. Reduce complexity of the thresholded edges in binary image by using approxPolyDP
# 4. Find all pairs of points along the 4-sided polygon approximations, and calculate the slope and intercepts
# 5. Use clustering to group pairs along lines.
# 6. Regress all clustered pairs to generate 'average' perspective lines
# 7. Find centroids that fall b/w any pairs of lines that frame/box squares.  Cluster these.
# 8. Draw line that connects corresponding clustered centroids, with intersections representing the 

# Python 2/3 compatibility
import sys
PY3 = sys.version_info[0] == 3
infinite_slope=999999999.0

if PY3:
    xrange = range

import argparse
import cv2 as cv
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering,KMeans

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Detect rectangles in a ColorChecker passport.  If some rectangles not detected, infer their locations based on location and size of the others.")
    parser.add_argument('-i', "--image", action='store', required=True, help="Path to input image.")
    parser.add_argument('-r', "--dim1", action='store', type=int, default=4, help="Number of dim1 in passport color checker.")
    parser.add_argument('-c', "--dim2", action='store', type=int, default=6, help="Number of columns in passport color checker.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

def rotate(x,y,xo,yo,theta): #rotate x,y around xo,yo by theta (rad)
    xr=math.cos(theta)*(x-xo)-math.sin(theta)*(y-yo)   + xo
    yr=math.sin(theta)*(x-xo)+math.cos(theta)*(y-yo)  + yo
    return [xr,yr]

def inertia_between(original_data, labels):
    means = np.zeros((len(labels),2), dtype=np.float)
    #Calc the mean of each variable feature for each class
    for i,label in enumerate(labels):
        means[label] += original_data[i,:]
    #for label in np.unique(labels):
        
    
class SquareDetector:
    def __init__(self,img):
        self.img_shape = img.shape

    def angle_cos(self, p0, p1, p2):
        d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
        return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

    def find_squares_thr(self, img, thr):
        squares = []
        if thr == 0:
            bin = cv.Canny(img, 0, 50, apertureSize=5)
            bin = cv.dilate(bin, None)
        else:
            _retval, bin = cv.threshold(img, thr, 255, cv.THRESH_BINARY)
        bin, contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt_len = cv.arcLength(cnt, True)
            cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
            if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
               cnt = cnt.reshape(-1, 2)
               max_cos = np.max([self.angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
               if max_cos < 0.1:
                   #Okay, it's a rectangle, but is it approximately a square?
                   (x, y, w, h) = cv.boundingRect(cnt)
                   ar = w / float(h)
                   # a square will have an aspect ratio that is approximately
                   # equal to one, otherwise, the shape is a rectangle
                   if ar >= 0.95 and ar <= 1.05:
                       squares.append(cnt)
        return(squares)

    def find_squares(self, img, thr_range):
        img = cv.GaussianBlur(img, (5, 5), 0)
        squares = []
        for gray in cv.split(img):
            for thr in thr_range:
                squares.extend(self.find_squares_thr(gray, thr))
        self.squares = squares
        return(squares)

    def get_squares():
        return(self.squares)

class ColorCheckerDetector(SquareDetector):
    def __init__(self, img, dim1, dim2):
        super(ColorCheckerDetector,self).__init__(img)
        self.dim1 = dim1
        self.dim2 = dim2
        self.merged_cnts = []


    #Merge overlapping squares.
    def merge_squares(self):
        "Merge overlapping squares to simplify the output."
        mask = np.zeros((self.img_shape[0],self.img_shape[1],1), dtype=np.uint8)
        #Draw all the contours on a binary masked image, but filled in
        #Now backtrack on the mask image and recalculate the new contours
        for square in self.squares:
            mask = cv.fillPoly(mask, [square], color=255)
        mask, new_cnt, _hierarchy = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        angles = []
        for cnt in new_cnt:
            (pt1, pt2, angle) = cv.minAreaRect(cnt)
            angles.append(angle)
        self.angles = angles
        self.angle_mean = np.mean(angles)
        self.merged_cnts = []
        self.merged_centroids = []
        for cnt in new_cnt:
            cnt_len = cv.arcLength(cnt, True)
            cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
            self.merged_cnts.append(cnt)
            M = cv.moments(cnt)
            cX = int((M["m10"] / M["m00"]))
            cY = int((M["m01"] / M["m00"]))
            self.merged_centroids.append(np.array([cX,cY]))
        #cv.imshow('merged', mask)
        #self.merged_cnts = new_cnt
        return(self.merged_cnts)

    #Get merged contours
    def get_contours(self):
        "Return the merged contours"
        return(self.merged_cnts)

    #Get angles of two perpendicular sides of each contour, and calculate the average angle
    def get_angles(self):
        #Take the bottom-most point of each 'square' contour and the connecting point immediately to the right, and calculate the angle of the edge connecting the two relative
        #to a horizontal line.
        #for cnt in self.merged_cnts:
        return(self.angles)

    def get_linear_coefficients(self):
        'Calculates linear slope and intercept params of each pair of adjacent lines in the found squares.'
        linear_coefficients_flat=[]
        linear_coefficients=[]
        for square in self.merged_cnts:
            max_i = len(square)
            square_coefficients = []
            for i in range(0,len(square)):
                pt1 = square[i][0]
                pt2 = square[(i+1)%max_i][0]
                if( pt2[0] == pt1[0] ):
                    theta = math.pi/2 #Indicates vertical line
                    b = pt1[0] #Use the x-intercept
                else:
                    m = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
                    theta  = math.atan(m)
                    if( (theta > math.pi/4) or (theta < -math.pi/4) ):
                        #Use the x-intercept to avoid extreme values in the y-intercept at very high angles
                        m_i = 1/m
                        b = pt2[0] - m_i * pt2[1]
                    else:
                        #Use the y-intercept to avoid extreme values in the x-intercept at very low angles
                        if( m == 0 ):
                            b = pt2[1]
                        else:
                            b = pt2[1] - m*pt2[0]
                square_coefficients.append([pt1,pt2,theta,b])
                linear_coefficients_flat.append([pt1,pt2,theta,b])
            linear_coefficients.append(square_coefficients)
        def normalize_coefficients(coefficients):
            cf = np.array(coefficients,dtype=np.object)
            ba = np.array([], dtype=np.float) #collection of all intercepts
            for (p1, p2, theta, b) in cf:
                if( (theta > math.pi/4) or (theta < -math.pi/4) ):
                    ba = np.append(ba,np.array([b,0.0]))
                else:
                    ba = np.append(ba,np.array([b,1.0]))
            ba = np.resize(ba,(int(ba.size/2),2))
            #Center the b's to have mean at 0
            x_i     = ba[:,1] == 0.0
            y_i     = ba[:,1] == 1.0
            bx      = ba[x_i,0]
            by      = ba[y_i,0]
            bmax_x  = np.max(bx)
            bmin_x  = np.min(bx)
            bmean_x = (bmax_x + bmin_x)/2
            #Center x-intercepts
            ba[x_i,0] = ba[x_i,0] - bmean_x
            #Scale x-intercepts
            cf[x_i,3] = ba[x_i,0] * math.pi/(bmax_x - bmin_x)
            bmax_y  = np.max(by)
            bmin_y  = np.min(by)
            bmean_y = (bmax_y + bmin_y)/2
            #Center y-intercepts
            ba[y_i,0] = ba[y_i,0] - bmean_y
            #Scale y-intercepts
            cf[y_i,3] = ba[y_i,0] * math.pi/(bmax_y - bmin_y)
            return(cf)
        self.linear_coefficients = linear_coefficients
        self.linear_coefficients_flat = normalize_coefficients(linear_coefficients_flat)
        return(self.linear_coefficients)

    def cluster_lines(self):
        'Clusters the pt pairs according to similar slope and intercepts.'
        #First cluster along theta, the angle of each edge of a square.  This should cluster into 2 groups.  Then, within each group,
        #cluster by their corresponding x or y-intercepts. 
        c1 = AgglomerativeClustering(n_clusters=(self.dim1 * 2) + (self.dim2 * 2), affinity='euclidean', linkage='ward')  
        linear_coefficients_minimal = self.linear_coefficients_flat[:,(2,3)]
        cout = c1.fit_predict(linear_coefficients_minimal)
        #Calculate the centroids and use as seed to KMeans clustering in order to rebuild the partitions
        #centroids = 
        #cout_unique = np.unique(cout)
                

    #Derive the optimal lines representing the rows and columns of the grid and nearly intersecting the centroids.  The goal is to derive the location
    #of the missing squares.
    def calc_rowcol_lines(self):
        "Calculate the optimal row and column lines based on centroids of squares and angles of lines"
        is_dim1 = False
        #Rotate/project all centroids into new positions based on the average angle.  This should be indifferent to the pivot point chosen
        angle_rad = (self.angle_mean*np.pi)/180
        rotated_centroids  = []
        for pt in self.merged_centroids:
            rpt = rotate(pt[0],pt[1],0,0,angle_rad)
            rotated_centroids.append(rpt)
        #After rotating, collapse all points onto x-axis (y=0) and y-axis (x=0), and use agglomerative clustering to determine how points line up.  The number of clusters
        #can be assumed based on the dimensional hints
        rpts_x = np.array([],dtype=np.float)
        #Collapse onto x-axis 
        for rpt in rotated_centroids:
            rpt_x = np.array([rpt[0],0])
            rpts_x = np.append(rpts_x,rpt_x)
        rpts_x = rpts_x.reshape((-1,2))
        #Compute agglomerative clustering using both dimensions specified, and then see which one has higher b/w inertia/total inertia (this is the correct dimension)
        c1 = AgglomerativeClustering(n_clusters=self.dim2, affinity='euclidean', linkage='ward')  
        cout = c1.fit_predict(rpts_x)  
        #one way to see if clustering works is to see if the number of points in each cluster equals the other dimension -- if not, flip the dimension we are clustering on
        cout2 = cout
                

    #Get centroids of contours
    def get_centroids(self):
        return(self.merged_centroids)

    #Find the missing squares based on the hint for number of dim1 and columns.
    def fill_missing_squares(self):
        return
        
    #Find all squares, inferring missed one's locations based on the location and size of the others.
    def find_all_squares(self):
        return

def main(image, dim1, dim2):
    img = cv.imread(image)
    h,w,d = img.shape
    newimg = cv.resize(img, (int(w/8), int(h/8)))
    ccdetector = ColorCheckerDetector(newimg,dim1,dim2)
    ccdetector.find_squares(newimg,xrange(100,255,2))
    squares = ccdetector.merge_squares()
    cv.drawContours( newimg, squares, -1, (0, 255, 0), 3 )
    ccdetector.get_linear_coefficients()
    ccdetector.cluster_lines()
    centroids = ccdetector.get_centroids()
    for centroid in centroids:
        newimg = cv.circle(newimg, (centroid[0],centroid[1]), 5, (255,0,0), -1)
    #ccdetector.calc_rowcol_lines()
    cv.imshow('squares', newimg)
    ch = cv.waitKey()
    cv.destroyAllWindows()

if __name__ == '__main__':
    parsed = parse_args()
    main(parsed.image, parsed.dim1, parsed.dim2)
