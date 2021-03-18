# Uses greyscale mask in image with XRite Colorcard to generate an output png with mean rgb values for encompassed by mask overlayed on color image.
#

import argparse
import cv2
import numpy as np
import os
from plantcv.plantcv import fatal_error
import sys

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Overlays a greyscale mask on a color image with an XRite ColorCard to generate a mean RGB png image for each card chip.") 
    parser.add_argument('-i', '--image', dest='image', action='store', required=True, help="Input color (RGB) image with XRite ColorCard.")
    parser.add_argument('-m', '--mask', action='store', required=True, help="Greyscale image file with different colorcard chips specified as unique greyscale values.  Values should be increasing and in order of the chip numbering on the card used.")
    parser.add_argument('-o', '--output', action='store', help="The output filename.  If not specified, is the name of the input color image with suffix *CCchip.means.png extension."
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)


def get_color_matrix(rgb_img, mask):
    """ Calculate the average value of pixels in each color chip for each color channel.

    Inputs:
    rgb_img         = RGB image with color chips visualized
    mask        = a gray-scale img with unique values for each segmented space, representing unique, discrete
                    color chips.

    Outputs:
    color_matrix        = a 22x4 matrix containing the average red value, average green value, and average blue value
                            for each color chip.
    headers             = a list of 4 headers corresponding to the 4 columns of color_matrix respectively

    :param rgb_img: numpy.ndarray
    :param mask: numpy.ndarray
    :return headers: string array
    :return color_matrix: numpy.ndarray
    """
    # Check for RGB input
    if len(np.shape(rgb_img)) != 3:
        fatal_error("Input rgb_img is not an RGB image.")
    # Check mask for gray-scale
    if len(np.shape(mask)) != 2:
        fatal_error("Input mask is not an gray-scale image.")

    # create empty color_matrix
    color_matrix = np.zeros((len(np.unique(mask))-1, 4))

    # create headers
    headers = ["chip_number", "r_avg", "g_avg", "b_avg"]

    # declare row_counter variable and initialize to 0
    row_counter = 0

    # for each unique color chip calculate each average RGB value
    for i in np.unique(mask):
        if i != 0:
            chip = rgb_img[np.where(mask == i)]
            color_matrix[row_counter][0] = i
            color_matrix[row_counter][1] = np.mean(chip[:, 2])
            color_matrix[row_counter][2] = np.mean(chip[:, 1])
            color_matrix[row_counter][3] = np.mean(chip[:, 0])
            row_counter += 1

    return headers, color_matrix


