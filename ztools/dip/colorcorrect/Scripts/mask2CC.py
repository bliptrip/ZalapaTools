#!/usr/bin/env python
# Uses greyscale mask in image with XRite Colorcard to generate an output png with mean rgb values for encompassed by mask overlayed on color image.
#

import argparse
import cv2
import json
import numpy as np
import os
from pathlib import PosixPath
from pymongo import MongoClient
import sys

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Overlays a greyscale mask on a color image with an XRite ColorCard to generate a mean RGB png image for each card chip.") 
    parser.add_argument('-i', '--image', action='store', help="Input color (RGB) image filepath with XRite ColorCard.  If unspecified, searches through SSE MongoDB for annotated images tagged with value specified in input flag --tags")
    parser.add_argument('-m', '--mask', action='store', help="Greyscale image file with different colorcard chips specified as unique greyscale values.  Values should be increasing and in order of the chip numbering on the card used.")
    parser.add_argument('-o', '--output', action='store', help="The output filename.  If not specified, is the name of the input color image with suffix *CCchip.means.PNG extension.")
    parser.add_argument('--hostname', action='store', default='localhost', help="The MongoDB SSE hostname/ip to connect to.")
    parser.add_argument('-s', '--settings', action='store', default="settings.json", help="The settings.json file.")
    parser.add_argument('-p', '--port', action='store', type=int, default=3001, help="The MongoDB SSE database port to connect to.")
    parser.add_argument('-d', '--db', action='store', default="meteor", help="MongoDB database name to access.")
    parser.add_argument('--tags', dest='tags', action='append', default=["cc"], help='SseSample tags to filter for when looking for annotated images with color checker.')
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)


def get_cc_chip_means(rgb_img, mask):
    """ Calculate the average value of pixels in each color chip for each color channel.

    Inputs:
    rgb_img         = RGB image with color chips visualized
    mask        = a gray-scale img with unique values for each segmented space, representing unique, discrete
                    color chips.

    Outputs:
    color_matrix        = a 22x3 matrix containing the average red value, average green value, and average blue value
                            for each color chip.

    :param rgb_img: numpy.ndarray
    :param mask: numpy.ndarray
    :return color_matrix: numpy.ndarray
    """
    # Check for RGB input
    if len(rgb_img.shape) != 3:
        print("Input rgb_img is not an RGB image.")
        return
    # Check mask for gray-scale
    if len(mask.shape) != 2:
        print("Input mask is not an gray-scale image.")
        return

    # create empty color_matrix
    color_matrix = np.zeros((len(np.unique(mask))-1, 3)) #Minus 1 b/c black is not a mask

    # declare row_counter variable and initialize to 0
    row_counter = 0

    # for each unique color chip calculate each average RGB value
    for m in np.sort(np.unique(mask)):
        if m != 0: #Black is not a mask
            chip = rgb_img[np.where(mask == m)]
            color_matrix[row_counter][2] = np.mean(chip[:, 2])
            color_matrix[row_counter][1] = np.mean(chip[:, 1])
            color_matrix[row_counter][0] = np.mean(chip[:, 0])
            row_counter += 1

    return color_matrix.reshape((-1,1,3)).astype('uint8')


if __name__ == '__main__':
    parsed        = parse_args()
    with open(parsed.settings, 'r') as settings_fh:
        settings            = json.loads(settings_fh.read())
        root                = settings['configuration']['images-folder']
        if( not(parsed.image or parsed.mask or parsed.output) ): #If none of these are specified, then access the mongodb to derive all annotated images.  NOTE: The sse2CCMask.py script should be run before this script, so that the corresponding mask images have been generated."
            client          = MongoClient(parsed.hostname, parsed.port)
            db              = client[parsed.db]
            sse_samples     = db.SseSamples
            for annotation in sse_samples.find({"tags": {"$in": parsed.tags}}):
                #Find the corresponding image on the local drive
                img_path  = PosixPath(root + annotation['folder'] + '/' + annotation['file'])
                #Load the image, mostly to just get size parameters
                img         = cv2.imread(str(img_path))
                mask_dir    = img_path.parent
                mask_stem   = img_path.stem
                mask_path   = mask_dir / (str(mask_stem) + ".mask.PNG")
                mask        = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                means       = get_cc_chip_means(img, mask)
                means_path  = mask_dir / (str(mask_stem) + ".CCchip.means.PNG")
                cv2.imwrite(str(means_path),means)
        else:
            assert parsed.image, "Must specify input color image path to --image flag."
            assert parsed.mask, "Must specify mask image path to --mask flag."
            img     = cv2.imread(str(parsed.image))
            mask    = cv2.imread(str(parsed.mask), cv2.IMREAD_GRAYSCALE)
            means   = get_cc_chip_means(img,mask)
            if parsed.output:
                cv2.imwrite(parsed.output,means)
            else:
                img_path     = PosixPath(parsed.image)
                means_dir    = img_path.parent
                means_stem   = img_path.stem
                means_path   = means_dir / (str(means_stem) + ".CCchip.means.PNG")
                cv2.imwrite(str(means_path),means)
