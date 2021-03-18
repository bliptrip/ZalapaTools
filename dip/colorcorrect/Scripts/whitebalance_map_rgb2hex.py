#!/usr/bin/env python
#Author: Andrew Maule
import argparse
import csv
import sys

def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="Output RGB values as hex.")
    parser.add_argument('-m', '--map', action='store', required=True, help="Mapping file in csv file format, label, r,g,b values.")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

def clamp(x): 
  return max(0, min(x, 255))

if __name__ == '__main__':
    parsed = parse_args()
    with open(parsed.map, 'r') as csvfile:
        mapCSV = csv.DictReader(csvfile)
        for row in mapCSV:
                id      = row['id']
                label   = row['label']
                r       = int(row['r'])
                g       = int(row['g'])
                b       = int(row['b'])
                hexstr  = "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))
                print("%s: %s: %s"%(id,label,hexstr))
