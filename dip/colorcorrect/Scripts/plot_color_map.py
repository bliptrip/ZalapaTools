#!/usr/bin/env python
#Author: Andrew Maule
import argparse
import csv
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from sklearn.linear_model import (LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import sys

estimators = [('OLS3', LinearRegression(), 3),
              ('OLS4', LinearRegression(), 4),
              ('OLS5', LinearRegression(), 5),
              ('OLS6', LinearRegression(), 6),
              ('OLS7', LinearRegression(), 7)]
colors = {'OLS3': 'turquoise', 'OLS4': 'gold', 'OLS5': 'lightgreen', 'OLS6': 'black', 'OLS7': 'red'}
linestyle = {'OLS3': '-', 'OLS4': '-.', 'OLS5': '--', 'OLS6': '--', 'OLS7': '-'}
lw = 3

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Plot the mappings of the L, a, b channels from input to output and see how interpolation looks.")
    parser.add_argument('-m', '--map', action='store', required=True, help="Mapping file in csv file format, with index, L_in, a_in, b_in, L_out, a_out, b_out.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

L_in_map = []
L_out_map = []
a_in_map = []
a_out_map = []
b_in_map = []
b_out_map = []

if __name__ == '__main__':
    parsed = parse_args()
    with open(parsed.map, 'r') as csvfile:
        mapCSV = csv.DictReader(csvfile)
        for row in mapCSV:
            L_in_map.append(row['L_in'])
            L_out_map.append(row['L_out'])
            a_in_map.append(row['a_in'])
            a_out_map.append(row['a_out'])
            b_in_map.append(row['b_in'])
            b_out_map.append(row['b_out'])
        L_in_map = np.array(L_in_map, dtype=np.float)*(255/100)
        L_out_map = np.array(L_out_map, dtype=np.float)*(255/100)
        i = L_out_map.argsort()
        L_in_map = L_in_map[i]
        L_out_map = L_out_map[i]
        L_in_map = L_in_map[:,np.newaxis]
        L_out_map = L_out_map[:,np.newaxis]
        #Lmap = interpolate.splrep(L_in_map, L_out_map, s=2)
        plt.scatter(L_in_map, L_out_map)
        #L_in_points = np.linspace(L_in_map[0],L_in_map[-1],100)
        #L_out_points = interpolate.splev(L_in_points, Lmap, der=0)
        #plt.plot(L_in_points, L_out_points)
        #plt.title('Interpolation behavior of map file.')
        #plt.xlabel('Input image L values')
        #plt.ylabel('Reference L value')
        #plt.show()

        xi = np.linspace(0.0,255.0,512)
        xi = xi[:,np.newaxis]

        for name, estimator, degree in estimators:
            model = make_pipeline(PolynomialFeatures(degree), estimator)
            model.fit(L_in_map, L_out_map)
            mse = mean_squared_error(model.predict(L_in_map), L_out_map)
            yi = model.predict(xi)
            plt.plot(xi, yi, color=colors[name], linestyle=linestyle[name],
                    linewidth=lw, label='%s mse=%.3f' % (name,mse))
        legend_title = 'Model'
        legend = plt.legend(loc='upper left', frameon=False, title=legend_title,
                prop=dict(size='x-small'))
        plt.xlim(0.0, 255.0)
        plt.ylim(0.0, 255.0)
        plt.title('Model prediction line')
        plt.show()
        c = cv2.waitKey(0)
