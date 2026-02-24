#!/usr/bin/env python
# coding: utf-8


import sys
import random
import numpy as np
from scipy import stats

def distr_bin(data, n_bin=30, logbin=True, namefile=''):
    ###This is a very old function copied from my c++ library. It's ugly but works :)
    """ Logarithmic binning of raw positive data;
        Input:
            data = np array,
            bins= number if bins,
            logbin = if true log bin
        Output (array: bins, array: hist) 
        bins: centred bins 
        hist: histogram value / (bin_length*num_el_data) [nonzero]
    """
    if len(data)==0:
        print( "Error empty data\n")
    min_d = float(min(data))
    if logbin and min_d<=0:
        print ("Error nonpositive data\n")
    n_bin = float(n_bin)            #ensure float values
    bins = np.arange(n_bin+1)
    if logbin:
        data = np.array(data)/min_d
        base= np.power(float(max(data)) , 1.0/n_bin)
        bins = np.power(base,bins)
        bins = np.ceil(bins)                   #to avoid problem for small ints
    else:
        data = np.array(data) + min_d          #to include negative data
        delta = (float(max(data)) - float(min(data)))/n_bin
        bins = bins*delta + float(min(data))
    n_bin = int(n_bin)
    #print ('first bin: ', bins[0], 'first data:', min(data), 'max bin:', bins[n_bin], 'max data', float(max(data)))
    hist = np.histogram(data, bins)[0]
    ii = np.nonzero(hist)[0]            #take non zero values of histogram
    bins = bins[ii]
    hist = hist[ii]
    bins=np.append(bins,float(max(data)))          #append the last bin
    bin_len = np.diff(bins)
    bins =  bins[:-1] + bin_len/2.0     #don't return last bin, centred boxes
    if logbin:
        hist = hist/bin_len                 #normalize values
        bins = bins*min_d                   #restore original bin values
    else:
        bins = bins - min_d 
    res = list(zip(bins, hist/float(sum(hist))))    #restore original bin values, norm hist
    if len(namefile)>0:
        np.savetxt(namefile,res)
    return list(zip(*res))
#end function


def powlaw(x, a, b) :
    return a * np.power(x, b)
def linlaw(x, a, b) :
    return a + x * b

def curve_fit_log(xdata, ydata) :
    from scipy.optimize import curve_fit
    """Fit data to a power law with weights according to a log scale"""
    # Weights according to a log scale
    # Apply fscalex
    xdata_log = np.log10(xdata)
    # Apply fscaley
    ydata_log = np.log10(ydata)
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
    #print(popt_log, pcov_log)
    # Apply fscaley^-1 to fitted data
    ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
    # There is no need to apply fscalex^-1 as original data is already available
    return (popt_log, pcov_log, ydatafit_log)


def get_distribution(data_sequence, number_of_bins = 30):
    # Modified from: https://github.com/ivanvoitalov/tail-estimation/
    # define the support of the distribution
    zeros = 0 in data_sequence
    lower_bound = min(data_sequence) if not zeros else min([k for k in data_sequence if k>0])
    upper_bound = max(data_sequence)
    # define bin edges
    log = np.log10
    upper_bound = log(upper_bound)
    lower_bound = log(lower_bound)
    bins = np.logspace(lower_bound, upper_bound, number_of_bins+1)
    if zeros:
        # construct a temporary bin for the zeros
        bins = np.insert(np.logspace(lower_bound, upper_bound, number_of_bins+1), 0, 0, axis=0)
    # compute the histogram using numpy
    y, _ = np.histogram(data_sequence, bins = bins, density = True)
    if zeros:
        # remove the temporary bin for the zeros, and the corresponding density
        bins = np.delete(bins, 0)
        y = np.delete(y, 0)
    # for each bin, compute its average
    x, _, _ = stats.binned_statistic(data_sequence, data_sequence, statistic='mean', bins = bins)
    # if bin is empty, drop it from the resulting list
    #drop_indices = [i for i,k in enumerate(y) if k == 0.0]
    #x = [k for i,k in enumerate(x) if i not in drop_indices]
    #y = [k for i,k in enumerate(y) if i not in drop_indices]
    return x, y

