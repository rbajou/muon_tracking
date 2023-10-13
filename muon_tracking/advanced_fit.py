#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 2021

@author: Raphaël Bajou
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os 
import sys
import argparse
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnchoredText
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal.waveforms import square
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import scipy.special as sc #binomial coef
import pylandau
import iminuit



def gaus(x,amp,x0,sigma):
    return amp*np.exp(-(x-x0)**2/(2*sigma**2))


def erf_func(x:np.ndarray, b:float, p:float,t:float,w:float):
    """
    b:lowest efficiency
    p:plateau value at high energy
    t:best estimate threshold position
    w:threshold width
    """
    func = b + p/2*(1+sc.erf( (x-t)/(np.sqrt(2)*w) ))
    return func

def fit_landau_migrad(x, y, p0, limit_mpv, limit_eta, limit_sigma, limit_A):
    def minimizeMe(mpv, eta, sigma, A):
        chi2 = np.sum(np.square(y - pylandau.langau(x, mpv, eta, sigma, A).astype(float)) / np.square(yerr.astype(float)))
        return chi2 / (x.shape[0] - 5)  # devide by NDF

    # Prefit to get correct errors
    yerr = np.sqrt(y)  # Assume error from measured data
    yerr[y < 1] = 1
    m = iminuit.Minuit(minimizeMe,
                       mpv=p0[0],
                       eta=p0[1],
                       sigma=p0[2],
                       A=p0[3],
    )
    m.limits['mpv'] = limit_mpv
    m.errors['mpv'] = 1
    m.limits['eta'] = limit_eta
    m.errors['eta'] = 0.1
    m.limits['sigma'] = limit_sigma
    m.errors['sigma'] = 0.1
    m.limits['A'] = limit_A
    m.errors['A'] = 1
    m.errordef = 1
    m.print_level = 2
    m.migrad()

    if not m._fmin.is_valid:
    #if not m.get_fmin().is_valid:
        raise RuntimeError('Fit did not converge')

    # Main fit with model errors
    yerr = np.sqrt(pylandau.langau(x,
                          mpv=m.values['mpv'],
                          eta=m.values['eta'],
                          sigma=m.values['sigma'],
                          A=m.values['A']))  # Assume error from measured data
    #yerr[y < 1] = 1

    m = iminuit.Minuit(minimizeMe,
                       mpv=m.values['mpv'],
                       eta=m.values['eta'],
                       sigma=m.values['sigma'],
                       A=m.values['A'],
    )
    m.limits['mpv'] = limit_mpv
    m.errors['mpv'] = 1
    m.limits['eta'] = limit_eta
    m.errors['eta'] = 0.1
    m.limits['sigma'] = limit_sigma
    m.errors['sigma'] = 0.1
    m.limits['A'] = limit_A
    m.errors['A'] = 1
    m.errordef = 1
    m.print_level = 2
    m.migrad()

    fit_values = m.values

    values = np.array([fit_values['mpv'],
                       fit_values['eta'],
                       fit_values['sigma'],
                       fit_values['A']])

    m.hesse()

    m.minos()
    minos_errors = m._merrors #m.get_merrors()

    if not minos_errors['mpv'].is_valid:
        print('Warning: MPV error determination with Minos failed! You can still use Hesse errors.')

    errors = np.array([(minos_errors['mpv'].lower, minos_errors['mpv'].upper),
                       (minos_errors['eta'].lower, minos_errors['eta'].upper),
                       (minos_errors['sigma'].lower, minos_errors['sigma'].upper),
                       (minos_errors['A'].lower, minos_errors['A'].upper)])

    return values, errors, m


def adc_panels(ADC, out_dir, name, l_id_panel) : 
    fig = plt.figure(0, figsize= (16,9))
    
    gs1 = GridSpec(1, len(l_id_panel), left=0.05, right=0.95, wspace=0.1)
    print(name)
    for i, id_panel in enumerate(l_id_panel): 
       
        axg  = fig.add_subplot(gs1[0, i])  
       
        xmax=0.75* max(ADC[i])
       
        nbins = 100
        
        arr_ADC = np.asarray(ADC[i])
    
        entries, bins, patches = axg.hist(arr_ADC,  range = (0,  xmax), bins =  nbins, color = "orange", label='inliers')
        bin_centers = np.array([ (bins[i+1]+bins[i])/2 for i in range(len(bins)-1) ])
        #bin_widths = np.array([ (bins_in[i+1]-bins_in[i]) for i in range(len(bins_in)-1) ])
        N = len(bin_centers)     
        mean = sum( bin_centers * entries) / sum(entries)#(n*max(nentries))#sum(np.multiply(bin_centers, nentries)) / n
        sigma = np.sqrt( sum( entries*(bin_centers - mean)**2  )  /  ((N-1)/N*sum(entries))  )
        rough_max = np.max( bin_centers[bin_centers>10][entries.argmax()] )#bin_centers[np.where(entries==max(entries))] )
        prom = 100
        print(rough_max)
        # peaks, _ = find_peaks( entries, height=(0., max(entries)),  prominence=prom)
        # rough_max = bin_centers[peaks][0]
        # fitrange =  ( ( rough_max*0.2 < bin_centers ) & (bin_centers< 4.0*rough_max ) )
        # yerr = np.asarray([np.sqrt(n) for n in entries[fitrange] ]) 
        #yerr[entries<1] = 1
        # xfit = bin_centers[fitrange]
        # yfit = entries[fitrange]
        # mpv, eta, amp = int(rough_max), 0.2*sigma, int(entries[bin_centers == rough_max][0])
        # print(mpv, eta, int(sigma), amp)
        # coeff, pcov = curve_fit(pylandau.langau, xfit, yfit,
        #                     sigma=yerr,
        #                     absolute_sigma=False,
        #                     p0=(mpv, eta, sigma, amp),
        #                     bounds=(1, 10000) )
        # axg.plot(xfit, pylandau.langau(xfit, *coeff), "-", color="red", label='MPV={:.0f}ADC\n$\\eta$={:.0f}ADC\n$\\sigma$={:.0f}ADC\nA={:.0f}'.format(*coeff)) #
        
         # Fit the data with numerical exact error estimation
    # Taking into account correlations
        # x=xfit
        # y=yfit
        # values, errors, m = fit_landau_migrad(
        #                                 x,
        #                                 y,
        #                                 p0=[mpv, eta, sigma, amp],#
        #                                 limit_mpv=(rough_max*0.2,rough_max*3), #(10., 100.)
        #                                 limit_eta=(0.1*sigma,0.5*sigma), #(2., 20.)
        #                                 limit_sigma=(0.5*sigma,1.5*sigma), #(2., 20.)
        #                                 limit_A=(0.5*amp,1.5*amp) #(500., 1500.)
        #                                 ) 

        # sym_err = [np.max(np.abs(e)) for e in errors] 
        # # Plot fit result
        # yerr = np.sqrt(pylandau.langau(x, *values))
        # axg.errorbar(x, y, yerr, fmt='.', color='r')
        # #print(values[:-1], errors)
        # param = ['MPV', '$\\eta$', '$\\sigma$', 'A']
        # label = ' '.join('{}={:0.1f}$\\pm${:0.1f} ADC\n'.format(p, value, error) for p, value, error in zip(param[:-1], values[:-1], sym_err[:-1]  )   )
        
        # axg.plot(x, pylandau.langau(x, *values), '-', label=label+'{}={:0.1f}$\\pm${:0.1f}'.format(param[-1], values[-1],sym_err[-1]), color='r' )



        # Show Chi2 as a function of the mpv parameter
        # The cut of the parabola at dy = 1 defines the error
        # https://github.com/iminuit/iminuit/blob/master/tutorial/tutorial.ipynb
        #plt.clf()
        #m.draw_mnprofile('mpv', subtract_min=False)

        # Show contour plot, is somewhat correlation / fit confidence
        # of these two parameter. The off regions are a hint for
        # instabilities of the langau function
        #plt.clf()
        #m.draw_mncontour('mpv', 'sigma', nsigma=3)
            
            
            
        
        
        axg.set_xlabel('ADC', fontsize=14) #(X) + n$_{adc}$(Y)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axg.legend(loc='center right', fontsize=10)
        
        axg.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        if i==0 : 
            axg.set_ylabel('entries', fontsize=14)
    
    plt.figtext(.5,.95, "Charge distributions (ADC)", fontsize=12, ha='center')
    plt.savefig(
        os.path.join(out_dir, "",  name+".png")
    )
    plt.close()
        


# if __name__ == '__main__':
    
#     _start_time = time.time()
#     print("Start: ", time.strftime("%H:%M:%S", time.localtime()))
#     home_path = os.environ['HOME']

#     parser=argparse.ArgumentParser( description='''This script takes as input arguments:

#         \n the output .txt file of RANSAC fit inliers from preprocessing output '--in_file',
#         \n the label of data file names '--label',
#         \n the output directory for flux maps plots '--plot_dir' 
#         ''')

#     parser.add_argument('--telescope', help='example:"SNJ1"', type=str, required=True)
#     parser.add_argument('--in_file', default="", help='', type=str)
#     parser.add_argument('--label', default='all_reco', help='', type=str)
#     parser.add_argument('--plot_dir', default= '', help='', type=str)
#     parser.add_argument('--out_dir', default= '', help='', type=str)
#     args=parser.parse_args()
#     plot_dir = args.plot_dir
#     label = args.label
   
#     Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
#     tel = Telescope(args.telescope)
#     panels = tel.panels
#     dict_panel_config = { id_panel : panel.dict_config for id_panel, panel in panels.items()}

#     l_id_panel = list(dict_panel_config.keys()) 

#     #by panel, all scintillators
   
#     l_sumadcX_gold, l_sumadcY_gold = list(list() for i in range(len(l_id_panel))), list(list() for i in range(len(l_id_panel)))
#     l_sumadcXY_gold = list(list() for i in range(len(l_id_panel)))
    
#     l_nadcX_panels_gold, l_nadcY_panels_gold = list(list() for i in range(len(l_id_panel))), list(list() for i in range(len(l_id_panel)))
#     l_maxadcX_gold, l_maxadcY_gold = list(list() for i in range(len(l_id_panel))), list(list() for i in range(len(l_id_panel)))
   


   
    
#     if args.in_file: 
#         file= os.path.join(args.in_file)
#         with open("{}".format(file), 'rt') as fg : 
            
#             for i,line_g in enumerate(fg):
                
#                 lg = line_g.split()
#                 # lxg = list(int(adc) for adc,id_p in zip(lg[4::5], lg[7::5]) if int(id_p) in id_req_panels) #in lg[4::5] if int(lg[lg.index(adc)+3]) in [id_front, id_rear])
#                 # lyg = list(int(adc) for adc,id_p in zip(lg[6::5], lg[7::5]) if int(id_p) in id_req_panels) #in lg[6::5] if int(lg[lg.index(adc)+1]) in [id_front, id_rear])
#                 # lsumx = sum(lxg)
#                 # lsumy = sum(lyg)
#                 # l_sumadctot_gold.extend( [lsumx, lsumy] )
#                 for j, id_panel in enumerate(l_id_panel):
#                     lx_p_g = list(int(adcx) for adcx,id_p in zip(lg[4::5], lg[7::5]) if int(id_p) == id_panel)
#                     ly_p_g = list(int(adcy) for adcy,id_p in zip(lg[6::5], lg[7::5]) if int(id_p) == id_panel)
#                     sum_lx_p_g, sum_ly_p_g= sum(lx_p_g), sum(ly_p_g)
#                     l_sumadcX_gold[j].append( sum_lx_p_g )  
#                     l_sumadcY_gold[j].append( sum_ly_p_g )  
#                     sumtot_g = sum_lx_p_g+sum_ly_p_g
#                     l_sumadcXY_gold[j].append(sumtot_g)
                    
#                     l_nadcX_panels_gold[j].extend( lx_p_g )  
#                     l_nadcY_panels_gold[j].extend( ly_p_g )
#         fg.close()
   
   
#     sumX_n_sumY_g = list(list() for i in range(len(l_id_panel)))
#     for i in range(len(l_id_panel)):
         
#         sumX_n_sumY_g[i]   = l_sumadcX_gold[i]+l_sumadcY_gold[i]

#     adc_panels(ADC=sumX_n_sumY_g,out_dir=plot_dir, name="sumXnsumY_"+label, l_id_panel=l_id_panel)
    
#     adc_panels(ADC=l_sumadcXY_gold,out_dir=plot_dir, name="sumXY_"+label, l_id_panel=l_id_panel)
        
      