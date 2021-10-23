#!/usr/bin/env python
# Uses greyscale mask in image with XRite Colorcard to generate an output png with mean rgb values for encompassed by mask overlayed on color image.
#

import argparse
import cv2
import numpy as np
import pandas as pd
import sys

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Generate a reference color 'swatch'  as specified for the color-card used.  Use a CSV file, which should define the RGB values for each chip of the swatch.")
    parser.add_argument('-c', '--config', action='store', required=True, help="The configuration file to use, in CSV format, that defines the reference colors.")
    parser.add_argument('-o', '--output', action='store', required=True, help="The output filepath.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)


if __name__ == '__main__':
    parsed        = parse_args()
    ref_df        = pd.read_csv(parsed.config)
    selected_cols = ["b","g","r"]
    refimg        = np.array(ref_df.loc[:,selected_cols],dtype="uint8").reshape((-1,1,3))
    cv2.imwrite(parsed.output,refimg)
