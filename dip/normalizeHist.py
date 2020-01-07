#!/usr/bin/env python
# Author: Andrew Maule
# Objective: To do automatic contrast/brightening of an 'image'
#
# Taken from code on stack overflow: http://answers.opencv.org/question/75510/how-to-make-auto-adjustmentsbrightness-and-contrast-for-image-android-opencv-image-correction/
#
#/**
# *  \brief Automatic brightness and contrast optimization with optional histogram clipping
# *  \param [in]src Input image GRAY or BGR or BGRA
# *  \param [out]dst Destination image 
# *  \param clipHistPercent cut wings of histogram at given percent tipical=>1, 0=>Disabled
# *  \note In case of BGRA image, we won't touch the transparency
# */
#void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent=0)
#{
#
#    CV_Assert(clipHistPercent >= 0);
#    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));
#
#    int histSize = 256;
#    float alpha, beta;
#    double minGray = 0, maxGray = 0;
#
#    //to calculate grayscale histogram
#    cv::Mat gray;
#    if (src.type() == CV_8UC1) gray = src;
#    else if (src.type() == CV_8UC3) cvtColor(src, gray, CV_BGR2GRAY);
#    else if (src.type() == CV_8UC4) cvtColor(src, gray, CV_BGRA2GRAY);
#    if (clipHistPercent == 0)
#    {
#        // keep full available range
#        cv::minMaxLoc(gray, &minGray, &maxGray);
#    }
#    else
#    {
#        cv::Mat hist; //the grayscale histogram
#
#        float range[] = { 0, 256 };
#        const float* histRange = { range };
#        bool uniform = true;
#        bool accumulate = false;
#        calcHist(&gray, 1, 0, cv::Mat (), hist, 1, &histSize, &histRange, uniform, accumulate);
#
#        // calculate cumulative distribution from the histogram
#        std::vector<float> accumulator(histSize);
#        accumulator[0] = hist.at<float>(0);
#        for (int i = 1; i < histSize; i++)
#        {
#            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
#        }
#
#        // locate points that cuts at required value
#        float max = accumulator.back();
#        clipHistPercent *= (max / 100.0); //make percent as absolute
#        clipHistPercent /= 2.0; // left and right wings
#        // locate left cut
#        minGray = 0;
#        while (accumulator[minGray] < clipHistPercent)
#            minGray++;
#
#        // locate right cut
#        maxGray = histSize - 1;
#        while (accumulator[maxGray] >= (max - clipHistPercent))
#            maxGray--;
#    }
#
#    // current range
#    float inputRange = maxGray - minGray;
#
#    alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
#    beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0
#
#    // Apply brightness and contrast normalization
#    // convertTo operates with saurate_cast
#    src.convertTo(dst, -1, alpha, beta);
#
#    // restore alpha channel from source 
#    if (dst.type() == CV_8UC4)
#    {
#        int from_to[] = { 3, 3};
#        cv::mixChannels(&src, 4, &dst,1, from_to, 1);
#    }
#    return;
#}

import cv2
import matplotlib.pyplot as plt
import numpy as np

class NormalizeHist:
    def __init__(self, out_range=None, clip_percent=0):
        '''Constructor for histogram.'''
        if( out_range == None):
            self.out_range = (0,256)
        else:
            self.out_range = out_range
        #Percentage to clip on each side of cumulative probability distribution
        self.clip_percent = clip_percent
        return

    def normalize(self, src, draw=False):
        '''Normalize automatically for brightness and contrast'''
        ''' src should already be on an 8-bit, uint8 scale (pass in the colorscale channel you care about normalizing)'''
        hist = cv2.calcHist(src, [0], None, [256], (0,256), accumulate=False)
        hist = hist / sum(hist)
        if (self.clip_percent == 0):
            minV,maxV,minL,maxL = cv2.minMaxLoc(src)
        else:
            cdf  = np.cumsum(hist)
            clip_tail_left  = (self.clip_percent/2.0)/100
            clip_tail_right = 1-clip_tail_left
            #Find the clip_start
            for i,e in enumerate(cdf):
                if e >= clip_tail_left:
                    minV = i
                    break
            #Find the clip_end
            for i in range(1,cdf.size+1):
                j = cdf.size - i
                if cdf[j] <= clip_tail_right:
                    maxV = j
                    break
        in_range = (maxV - minV) + 1
        out_range = self.out_range[1] - self.out_range[0]
        alpha = out_range/in_range
        beta = -minV*alpha
        #dst = src * alpha + beta
        dst = cv2.convertScaleAbs(src, alpha=alpha, beta=beta)
        dst = dst.astype(np.uint8)
        out_hist = cv2.calcHist(dst, [0], None, [out_range], (0,out_range), accumulate=False)
        out_hist = out_hist/sum(out_hist)
        if(draw):
            fig, (ax1, ax2) = plt.subplots(1,2, num=10, sharey=True, clear=True)
            ax1.plot(np.array(range(0, hist.size)), hist)
            ax1.set_title('Original Image Histogram')
            plt.xlim((0,hist.size))
            ax2.plot(np.array(range(0, out_range)), out_hist)
            plt.xlim((0,out_hist.size))
            ax2.set_title('New Image Histogram')
            plt.show()
        return dst
