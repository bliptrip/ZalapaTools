#!/usr/bin/env python3
# Author: Andrew Maule
# Objective: Calculates the normalized unsigned manhattan chain code per Li Mao et al., 2016's work: 10.1016/j.jvcir.2016.03.001
#
#         
# import the necessary packages
import argparse
import logging
import math
import numpy as np
import sys

# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description="A command-line utility for generating a unsigned manhattan chain code (UMCC) representation of berry shapes.")
    parser.add_argument('-i', '--input', action='store', type=str, required=True, help="Path to input file containing numpy-formatted array of contours for berry shape.")
    parser.add_argument('-o', '--output', action='store', type=str, required=True, help="Path to output file to store numpy-formatted array of unsigned manhattan chain code.")
    parser.add_argument('-n', '--nnormalize', action='store_true', help="Flag to let script know that it should _not_ normalize the UMCC string.  By default the UMCC is normalized.")
    parser.add_argument('-l', '--level', type=str, default="WARNING", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Set the logging level for the UMCC encoder (debugging purposes).")
    parsed = parser.parse_args(sys.argv[1:])
    return(parsed)

def process_rule_1(Ux, Uy, dx, dy):
    Ux = Ux.append(str(abs(dx)))
    Uy = Uy.append(str(abs(dy)))
    return

def process_rule_2(Ux, Uy, dx, dy):
    Ux = Ux.extend(['0','1'])
    Uy = Uy.extend(['0','1'])
    return

def process_rule_3(Ux, Uy, dx, dy):
    Ux = Ux.extend(['0','1',None])
    Uy = Uy.extend(['0','0',str(abs(dy))])
    return

def process_rule_4(Ux, Uy, dx, dy):
    Ux = Ux.extend(['0','0',str(abs(dx))])
    Uy = Uy.extend(['0','1',None])
    return

class UMCCEncoder(object):
    UMCC_DICT = {-90:[0,1], -45:[1,1], 0:[1,0], 45:[1,-1], 90:[0,-1], 135:[-1,-1], 180:[-1,0], -135:[-1,1]}
    ALLOWED_DIRECTIONS = np.array([0, 45, 90, 135, 180, -45, -90, -135])

    umcc_rules = {
        (0,0): process_rule_1,
        (1,1): process_rule_2,
        (1,0): process_rule_3,
        (0,1): process_rule_4
    }

    def __init__(self, logger, normalize=True, contours=None, U=None):
        if U is not None:
            self.Ux = U[0]
            self.Uy = U[1]
            self.x  = self.asnumeric(U[0])
            self.y  = self.asnumeric(U[1])
        else:
            self.Ux = np.array([])
            self.Uy = np.array([])
            self.x  = 0
            self.y  = 0
        self.normalize = normalize
        self.logger = logger
        if contours is not None:
            self.encode(contours)

    def find_nearest(self,array,value):
        '''
        Find the nearest element of array to the given value
        '''
        idx = (np.abs(array-value)).argmin()
        return array[idx]

    def circShiftL(self, U, number):
        shifted     = number
        if( len(U) > 1 ):
            if( U[0] == None ):
                shift = 2 #We need to shift '2' for the list-form, as we ignore the 'None'
            else:
                shift = 1
            U   = U[shift:] + U[0:shift]
        if( (number).bit_length() > 1 ):
            msb_shift   = (number).bit_length() - 1
            msb         = number >> msb_shift
            shifted     = ((number & ((1 << msb_shift)-1)) << 1) + msb
        return((U,shifted))


    def circShiftR(self, U, number):
        shifted     = number
        if( len(U) > 1 ):
            if( U[-1] == None ):
                shift = 2 #We need to shift '2' for the list-form, as we ignore the 'None'
            else:
                shift = 1
            U   = U[-shift:] + U[0:-shift]
        if( (shifted).bit_length() > 1 ):
            lsb_shift   = (number).bit_length() - 1
            lsb         = number & 1
            shifted     = (lsb << lsb_shift) + (number >> 1)
        return((U,shifted))


    def asnumeric(self, U):
        npU         = np.array(U)
        npUs        = npU[npU != None] #'s' stands for 'stripped' - of 'None'
        return(int('0b'+''.join(npUs),2))
        

    def doNormalize(self):
        Ux      = self.Ux
        Uy      = self.Uy
        Ux_numeric = self.asnumeric(Ux)
        kxshifts = 0
        xshifted = Ux_numeric_min = Ux_numeric
        Ux_min   = Ux
        for i in range(1,(Ux_numeric).bit_length()):
            Ux,xshifted = self.circShiftL(Ux,xshifted)
            if( xshifted < Ux_numeric_min ):
                Ux_min         = Ux
                Ux_numeric_min = xshifted
                kxshifts = i
        Uy_numeric = self.asnumeric(Uy)
        kyshifts = 0
        yshifted = Uy_numeric_min = Uy_numeric
        for i in range(1,(Uy_numeric).bit_length()):
            Uy, yshifted = self.circShiftL(Uy, yshifted)
            if( yshifted < Uy_numeric_min ):
                Uy_min         = Uy
                Uy_numeric_min = yshifted
                kyshifts = i
        Kshifts = min(kxshifts,kyshifts)
        self.logger.debug("kx: {}, ky: {}, K: {}".format(kxshifts,kyshifts,Kshifts))
        #Now shift right
        Ux = Ux_min
        xshifted = Ux_numeric_min
        for i in range(0, kxshifts - Kshifts):
            Ux,xshifted = self.circShiftR(Ux,xshifted)
        Uy = Uy_min
        yshifted = Uy_numeric_min
        for i in range(0, kyshifts - Kshifts):
            Uy,yshifted = self.circShiftR(Uy,yshifted)
        self.Ux = Ux
        self.x  = xshifted
        self.Uy = Uy
        self.y  = yshifted
        return


    def encode(self, contours):
        contours = np.insert(contours,0,contours[0],axis=0) #Repeat first element to make it loop back on itself
        contours = np.insert(contours[-1:0:-1],0,contours[0],axis=0) #Make clockwise
        fx = 0 #Initialize sign-change flag to 0 in x-direction
        fy = 0 #Initialize sign-change flag to 0 in y-direction
        sx = 1 #Assume a positive direction to begin with for x-direction
        sy = 1 #Assume a positive direction to begin with for y-direction
        Ux = [] #List of encoded unsigned manhattan codes in x-direction
        Uy = [] #List of encoded unsigned manhattan codes in y-direction
        for i in range(1,len(contours)):
            cprev = contours[i-1]
            self.logger.debug("cprev: {}".format(cprev))
            ccurr = contours[i]
            self.logger.debug("ccurr: {}".format(ccurr))
            cdiff = (ccurr - cprev)
            self.logger.debug("cdiff: {}".format(cdiff))
            angle = math.degrees(math.atan2(-1*cdiff[1],cdiff[0]))
            angle = self.find_nearest(UMCCEncoder.ALLOWED_DIRECTIONS, angle)
            self.logger.debug("angle: {}".format(angle))
            step = UMCCEncoder.UMCC_DICT[angle]
            self.logger.debug("step: {}".format(step))
            fx = 0
            if( (step[0] != 0) and (step[0] != sx) ):
                sx = -1*sx
                fx = 1 #Monotonicity changes in x-direction
            fy = 0
            if( (step[1] != 0) and (step[1] != sy) ):
                sy = -1*sy
                fy = 1 #Monotonicity changes in y-direction
            self.logger.debug("sx: {}, fx: {}, sy: {}, fy: {}".format(sx,fx,sy,fy))
            UMCCEncoder.umcc_rules[(fx,fy)](Ux,Uy,step[0],step[1])
        self.Ux = Ux
        self.Uy = Uy
        self.x  = self.asnumeric(Ux)
        self.y  = self.asnumeric(Uy)
        if(self.normalize):
            self.doNormalize() #Normalize the chain code for equivalent comparison b/w different chaincodes
        return


    def __str__(self):
        npUx = np.array(self.Ux)
        npUx[npUx == None] = ' ' #Replace 'None' with space for useful visualization
        npUy = np.array(self.Uy)
        npUy[npUy == None] = ' ' #Replace 'None' with space for useful visualization
        return("'0b{}'\n'0b{}'".format(''.join(npUx.tolist()),''.join(npUy.tolist())))


    def numeric(self):
        return({'Ux': self.x, 'Uy': self.y})


    def raw(self):
        return({'Ux': self.Ux, 'Uy': self.Uy})
        
if __name__ == '__main__':
    logger      = logging.getLogger()
    parsed      = parse_args()
    decoded_level = eval("logging.{}".format(parsed.level))
    logger.setLevel(decoded_level)
    contours    = np.load(parsed.input)
    umcc        = UMCCEncoder(logger=logger,normalize=(not parsed.nnormalize), contours=contours)
    umccR       = umcc.raw()
    umcc_np     = np.array([umccR['Ux'],umccR['Uy']])
    np.save(parsed.output,umcc_np,allow_pickle=True)
