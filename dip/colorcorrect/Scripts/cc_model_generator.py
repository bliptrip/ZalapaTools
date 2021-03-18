#!/usr/bin/env python
#Author: Andrew Maule
import argparse
import csv
import cv2
import math
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplot as subplt
import numpy as np
import pickle
from scipy import interpolate
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import sys

#What Regression estimators to iterate over, and the polynomial range
degree_range = range(1,15)
estimators = [('OLS', 'LinearRegression()'),
              ('Ridge', 'Ridge()')]

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Generate a color correction model to map input colors to  'normalized' output colors.")
    parser.add_argument('-m', '--map', action='store', required=True, help="Color mapping file in csv file format, with label, index, L_cie_in, a_cie_in, b_cie_in, L_cie, a_cie, b_cie.")
    parser.add_argument('-o', '--output', action='store', required=True, help="An output binary file that contains the information for the trained model, in python pickle format.")
    parser.add_argument('-d', '--debug', action='store_true', help="Show debug plots for mapping input to output values.")
    parser.add_argument('-i', '--include_images', action='store', help="A newline-separated file of all image files that are part of the training data.  NOTE: This is optional, but if specified, it changes the range of possible mapping values.")
    parser.add_argument('--include_image_path', action='store', default="", help="If --include_images specified, this is the prefix to attach to all files.")
    parser.add_argument('--no_L', action='store_true', help="Exclude the CIE L* channel from being remapped.")
    parser.add_argument('--no_a', action='store_true', help="Exclude the CIE a* channel from being remapped.")
    parser.add_argument('--no_b', action='store_true', help="Exclude the CIE b* channel from being remapped.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)


class CCModelGenerator():
    def __init__(self):
        return

    def find_session_extremes(self,include_path,includes):
        'Scan all images in includes file, attaching include_path to each file.  This is intended to derive max/min bounds to specify on model, with the goal of better regression accuracy.'
        L_max = 0.0
        a_max = 0.0
        b_max = 0.0
        L_min = 255.0
        a_min = 255.0
        b_min = 255.0
        with open(includes,'r') as includes_fh:
            for image_file in includes_fh.readlines():
                image_file = include_path + image_file.rstrip()
                img = cv2.imread(image_file)
                img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                L_channel, a_channel, b_channel = cv2.split(img_lab)
                img_L_max = np.max(L_channel)
                img_L_min = np.min(L_channel)
                L_max = max(L_max,img_L_max)
                L_min = min(L_min,img_L_min)
                img_a_max = np.max(a_channel)
                img_a_min = np.min(a_channel)
                a_max = max(a_max,img_a_max)
                a_min = min(a_min,img_a_min)
                img_b_max = np.max(b_channel)
                img_b_min = np.min(b_channel)
                b_max = max(b_max,img_b_max)
                b_min = min(b_min,img_b_min)
            includes_fh.close()

        return(L_min,L_max,a_min,a_max,b_min,b_max)

    def cleanup_map(self,ina,outa,minv,maxv,canonical_minv=0,canonical_maxv=255):
            'Cleanup the existing input and output arrays to deal with missing data and to cap/floor results to minv or maxv'
            #To avoid saturation effects under conditions where training set input has smaller range than training
            #set output, but the image data to be converted has values out of the range of the training input, insert some
            #train set input -> output maps that fill in the gaps.
            #Find the minimum input value, and set a range between 0 and that value with step sizes of 10
            ina_min = np.min(ina)
            ina_min_idx = np.min(np.nonzero(ina == ina_min))
            outa_min = outa[ina_min_idx]
            #Now derive a linear function that maps input to output as a simple 1 degree line from minv,minv -> ina_min,outa_min
            min_m = (outa_min-canonical_minv)/(ina_min-canonical_minv)
            min_b = outa_min - (min_m * ina_min)
            f_min = lambda x: x * min_m + min_b
            ina_holes = [x for x in range(int(minv),int(ina_min),10)]
            outa_holes = map(f_min, ina_holes)
            #Map these intermediate steps to output based on line that connects minv,minv to minIn,minOut
            for x,y in zip(ina_holes, outa_holes):
                    ina = np.append(ina,x)
                    outa = np.append(outa,y)
            #Now fill in the maximum input values from maxIn,maxOut to 255,255
            ina_max = np.max(ina)
            ina_max_idx = np.max(np.nonzero(ina == ina_max))
            outa_max = outa[ina_max_idx]
            max_m = (canonical_maxv-outa_min)/(canonical_maxv-ina_min)
            max_b = outa_max - (max_m * ina_max)
            f_max = lambda x: x * max_m + max_b
            ina_holes = [x for x in range(math.ceil(ina_max+10.0),int(maxv),10)]
            outa_holes = map(f_max, ina_holes)
            for x,y in zip(ina_holes, outa_holes):
                    ina = np.append(ina,x)
                    outa = np.append(outa,y)
            #Append the mapping at the extreme
            ina = np.append(ina, maxv)
            outa = np.append(outa, maxv)
            #Now cap or floor anything that falls above or below the desired range
            ina[ina < minv] = minv
            ina[ina > maxv] = maxv
            outa[outa < minv] = minv
            outa[outa > maxv] = maxv
            return([ina,outa])

    def find_optimal_estimator(self, estimators, degree_range, ina, outa, channel, mdebug):
            models  = []
            for name, estimator in estimators:
                    for degree in degree_range:
                            if(mdebug):
                                print("Lab CIE Channel: %s -- Running pipeline %s with degree %d." % (channel, name, degree))
                            estimator_object = eval(estimator)
                            lmodel = make_pipeline(PolynomialFeatures(degree), estimator_object)
                            lmodel.fit(ina, outa)
                            mse = mean_squared_error(lmodel.predict(ina), outa)
                            models.append({"name":name, "degree":str(degree), "model":lmodel, "mse":mse})
            #Choose the model that has the lowest mse
            mses   = [x["mse"] for x in models]
            idx    = mses.index(min(mses))
            return(models[idx])

    def generate_models(self,mmap,output,mdebug=False,include_image_path="",include_images="",nL=False,na=False,nb=False):
        #Initialize the theoretical maximum & minimum values for the CIE Lab channel values
        L_max = 255.0
        a_max = 254.0
        b_max = 254.0
        L_min = 0.0
        a_min = 0.0
        b_min = 0.0
        #Initialize the input and output CIE Lab map lists
        L_cie_in_map = []
        L_cie_map = []
        a_cie_in_map = []
        a_cie_map = []
        b_cie_in_map = []
        b_cie_map = []
        #Initialize the models
        modelL = None
        modela = None
        modelb = None
        #Read in the maps from the colorchecker map file.
        with open(mmap, 'r') as csvfile:
            mapCSV = csv.DictReader(csvfile)
            for row in mapCSV:
                if not nL:
                    L_cie_in_map.append(row['L_cie_in'])
                    L_cie_map.append(row['L_cie'])
                if not na:
                    a_cie_in_map.append(row['a_cie_in'])
                    a_cie_map.append(row['a_cie'])
                if not nb:
                    b_cie_in_map.append(row['b_cie_in'])
                    b_cie_map.append(row['b_cie'])
            #If the user wants a prescan of a set of corresponding images to find empirical minimum/maximums values for each channel, then do this here.
            if( include_images ):
                    L_min, L_max, a_min, a_max, b_min, b_max = self.find_session_extremes(include_image_path, include_images)
            #For each CIE Lab channel, find the optimal model that maps inputs to output
            if not nL:
                L_cie_in_map = np.array(L_cie_in_map, dtype=np.float)*(255/100)
                L_cie_map = np.array(L_cie_map, dtype=np.float)*(255/100)
                [L_cie_in_map,L_cie_map] = self.cleanup_map(L_cie_in_map, L_cie_map, L_min, L_max)
                L_cie_in_map = L_cie_in_map[:,np.newaxis]
                L_cie_map = L_cie_map[:,np.newaxis]
                modelLd  = self.find_optimal_estimator(estimators, degree_range, L_cie_in_map, L_cie_map, 'L*', mdebug)
                modelL   = modelLd['model']
            if not na:
                a_cie_in_map = np.array(a_cie_in_map, dtype=np.float)+128
                a_cie_map = np.array(a_cie_map, dtype=np.float)+128
                [a_cie_in_map,a_cie_map] = self.cleanup_map(a_cie_in_map, a_cie_map, a_min, a_max)
                a_cie_in_map = a_cie_in_map[:,np.newaxis]
                a_cie_map = a_cie_map[:,np.newaxis]
                modelad  = self.find_optimal_estimator(estimators, degree_range, a_cie_in_map, a_cie_map, 'a*', mdebug)
                modela   = modelad['model']
            if not nb:
                b_cie_in_map = np.array(b_cie_in_map, dtype=np.float)+128
                b_cie_map = np.array(b_cie_map, dtype=np.float)+128
                [b_cie_in_map,b_cie_map] = self.cleanup_map(b_cie_in_map, b_cie_map, b_min, b_max)
                b_cie_in_map = b_cie_in_map[:,np.newaxis]
                b_cie_map = b_cie_map[:,np.newaxis]
                modelbd  = self.find_optimal_estimator(estimators, degree_range, b_cie_in_map, b_cie_map, 'b*', mdebug)
                modelb   = modelbd['model']
            #If the user specifies debug, then display a graph of the input->output model mapping.
            if( mdebug ):
                    #3 figures for each Lab CIE channel, 2 plots per picture to account for actual data values and predicted data values
                    if not nL:
                        subplt(2,2,1)
                        L_debug_in  = np.array(range(int(L_min),int(L_max)+1),dtype=np.float)
                        L_debug_in  = L_debug_in.reshape(-1,1)
                        L_debug_out = modelL.predict(L_debug_in)
                        plt.scatter(L_cie_in_map, L_cie_map, c='r')
                        plt.plot(L_debug_in, L_debug_out, 'b-')
                        plt.title("Channel: L*: Model: %s, Poly Degrees: %s, MSE: %.2f" % (modelLd['name'],modelLd['degree'],modelLd['mse']))
                        plt.xlim(L_min-5,L_max+5)
                        plt.ylim(L_min-5,L_max+5)
                    if not na:
                        subplt(2,2,2)
                        a_debug_in  = np.array(range(int(a_min),int(a_max)+1),dtype=np.float)
                        a_debug_in  = a_debug_in.reshape(-1,1)
                        a_debug_out = modela.predict(a_debug_in)
                        plt.scatter(a_cie_in_map, a_cie_map, c='r')
                        plt.plot(a_debug_in, a_debug_out, 'b-')
                        plt.title("Channel: a*: Model: %s, Poly Degrees: %s, MSE: %.2f" % (modelad['name'],modelad['degree'],modelad['mse']))
                        plt.xlim(a_min-5,a_max+5)
                        plt.ylim(a_min-5,a_max+5)
                    if not nb:
                        subplt(2,2,3)
                        b_debug_in  = np.array(range(int(b_min),int(b_max)+1),dtype=np.float)
                        b_debug_in  = b_debug_in.reshape(-1,1)
                        b_debug_out = modelb.predict(b_debug_in)
                        plt.scatter(b_cie_in_map, b_cie_map, c='r')
                        plt.plot(b_debug_in, b_debug_out, 'b-')
                        plt.title("Channel: b*: Model: %s, Poly Degrees: %s, MSE: %.2f" % (modelbd['name'],modelbd['degree'],modelbd['mse']))
                        plt.xlim(b_min-5,b_max+5)
                        plt.ylim(b_min-5,b_max+5)
                    plt.show()
                    ch = cv2.waitKey(10000)
            models_dict = {"L": modelL, "a": modela, "b": modelb}
            with open(output, 'wb') as output_fh:
                pickle.dump(models_dict, output_fh)
                output_fh.close()


if __name__ == '__main__':
    parsed     = parse_args()
    ccmodgen   = CCModelGenerator()
    ccmodgen.generate_models(mmap=parsed.map,output=parsed.output,mdebug=parsed.debug,include_image_path=parsed.include_image_path,include_images=parsed.include_images,nL=parsed.no_L,na=parsed.no_a,nb=parsed.no_b)
