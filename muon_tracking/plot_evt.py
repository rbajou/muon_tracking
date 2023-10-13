#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 2021
@author: Raphaël Bajou
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines  #use for legend settings
from typing import List, Union
import os
from pathlib import Path
import inspect
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(filename))
import pandas as pd
import glob
#personal modules
from muon_tracking.configuration import Telescope, dict_tel, str2telescope
from muon_tracking.processing import InputType
from muon_tracking.analysis import RecoData
import argparse
from random import sample

class PlotEvt:
    def __init__(self, telescope:Telescope, recodir:str, input_type=InputType.DATA, outdir:str=os.environ['HOME']):
        self.telescope = telescope
        self.outdir = outdir
        print(glob.glob(os.path.join(recodir,  '', '*_reco*') )[0])
        try:
            f_reco = glob.glob(os.path.join(recodir,  '', '*_reco*') )[0]
            print(f_reco)
            f_inlier = glob.glob(os.path.join(recodir,  '', '*_inlier*') )[0]
            f_outlier = glob.glob(os.path.join(recodir,  '', '*_outlier*') )[0]
        except: raise ValueError
        reco_trk  = RecoData(file=f_reco, 
                                                               telescope=self.telescope, 
                                                               input_type=input_type)
        inlier_data = RecoData(file=f_inlier, 
                                                               telescope=self.telescope, 
                                                               input_type=input_type)
        outlier_data = RecoData(file=f_outlier, 
                                                               telescope=self.telescope, 
                                                               input_type=input_type)
        self.df_reco = reco_trk.df
        self.evtID = list(self.df_reco.index)
        self.df_inlier = inlier_data.df
        self.evtID_in = list(set(self.df_inlier.index))
        #print(len(self.evtID_in))
        sumADC_XY_in = self.df_inlier['ADC_X'] + self.df_inlier['ADC_Y'] 
        self.df_inlier = self.df_inlier.assign(sumADC_XY=pd.Series(sumADC_XY_in).values)
        
        self.df_outlier = outlier_data.df
        self.evtID_out = list(set(self.df_outlier.index))
        sumADC_XY_out = self.df_outlier['ADC_X'] + self.df_outlier['ADC_Y'] 
        self.df_outlier = self.df_outlier.assign(sumADC_XY=pd.Series(sumADC_XY_out).values)

    def get_points(self, evtID:Union[List, int]):
        l_evtID = []
        if type(evtID)==list: l_evtID.extend(evtID)
        else : l_evtID.append(evtID)
        
        Z = np.sort(list(set(self.df_inlier['Z'])))
        #print(self.df_inlier.head)
        self.xyz_reco = { i : np.array([ np.concatenate( (self.df_reco.loc[i][[f'X_{p.position.loc}', f'Y_{p.position.loc}']].to_numpy(), p.position.z) , axis=None )   for p in self.telescope.panels ]) for i in l_evtID}
        #print(self.xyz_reco)
        self.xyz_in = { i : self.df_inlier.loc[i][['X', 'Y', 'Z']].to_numpy() if i in self.evtID_in else np.zeros(3) for i in l_evtID}
        #print({ i : self.df_inlier.loc[i]['sumADC_XY'] if i in self.evtID_in else 0 for i in evtID })
        #print({ i : type(self.df_inlier.loc[i]['sumADC_XY']) if i in self.evtID_in else 0 for i in evtID })
        self.adc_in = { i : self.df_inlier.loc[i]['sumADC_XY'].tolist() if i in self.evtID_in else [0] for i in l_evtID }
        self.xyz_out = { i : self.df_outlier.loc[i][['X', 'Y', 'Z']].to_numpy() if i in self.evtID_out else np.zeros(3) for i in l_evtID}
        self.adc_out = { i : self.df_outlier.loc[i]['sumADC_XY'].tolist() if i in self.evtID_out else [0] for i in l_evtID}
        
        # print(f'self.xyz_reco= {self.xyz_reco}')
        # print(f'self.xyz_in= {self.xyz_in}')
        # print(f'self.adc_in= {self.adc_in}')
        # print(f'self.adc_out= {self.adc_out}')
        #self.ADC_XY_inlier = [ np.array(self.df_inlier.loc[self.df_inlier['Z'] ==z]["sumADC_XY"].values) for z in Z]
        #self.XYZ_inlier = [ np.array(self.df_inlier.loc[self.df_inlier['Z'] ==z]["sumADC_XY"].values) for z in Z]
        #self.ADC_XY_outlier = [ np.array(self.df_outlier.loc[self.df_outlier['Z'] ==z]["sumADC_XY"].values) for z in Z]


    def plot3D(self, evtID:Union[List, int], isReco:bool=True, isInlier:bool=True, isOutlier:bool=True  ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d' )
        ax.set_facecolor('white')
        self.telescope.plot3D(ax, position=np.zeros(3))
        vis_factor = 100 #for size XY pints
        l_evtID = []
        if type(evtID)==list: l_evtID.extend(evtID)
        else : l_evtID.append(evtID)
        for i in l_evtID : 
            if isReco==True:
                ####Trajectory
                xyz_r = self.xyz_reco[i]
                xyz_r = xyz_r[~np.all( (xyz_r[:, :-1] == 0.), axis=1)]
                xs, ys, zs = xyz_r[0] #1st intersection pt
                xe, ye, ze = xyz_r[-1] #last intersection pt
                ax.plot([xs, xe],
                        [ys, ye],
                        [zs, ze], 
                        c="blue", linewidth=0.75)
                ax.scatter(
                    xyz_r[:, 0],
                    xyz_r[:, 1],
                    xyz_r[:, 2],
                    c='blue',
                    s=10,
                    marker='+',
                    linewidths=1
                )

            ###Inliers & Outliers (RANSAC)
            xyz_i, xyz_o = self.xyz_in[i], self.xyz_out[i]
            if xyz_i.ndim == 1: xyz_i = xyz_i[np.newaxis, :]
            if xyz_o.ndim == 1: xyz_o = xyz_o[np.newaxis, :]
           
            adc_i, adc_o = list(), list()
            if type(self.adc_in[i]) == float :  adc_o.append(self.adc_in[i])
            else: adc_i = self.adc_in[i]
            if type(self.adc_out[i]) == float :  adc_o.append(self.adc_out[i])
            else : adc_o = self.adc_out[i]
            adc_all = adc_i+adc_o
            
            if isInlier==True and xyz_i.any() != False:
                
                arr_norm_sum_adc_in = adc_i/np.max(adc_all)
                
                ax.scatter(
                    xyz_i[:, 0],
                    xyz_i[:, 1],
                    xyz_i[:, 2],
                    s= arr_norm_sum_adc_in*vis_factor,
                    c= 'limegreen',
                    marker='o',
                    edgecolor='green',
                    label= 'inlier' 
                )
                ax.scatter(
                    xyz_i[:, 0],
                    xyz_i[:, 1],
                    xyz_i[:, 2],
                    c='green',
                    s=10,
                    marker='.',
                    linewidths=1
                )
            
            
            if isOutlier==True and xyz_o.any() != False :
                arr_norm_sum_adc_out = adc_o/np.max(adc_all)
                ax.scatter(
                    xyz_o[:, 0],
                    xyz_o[:, 1],
                    xyz_o[:, 2],
                    c='tomato',
                    s= arr_norm_sum_adc_out*vis_factor,#np.exp(arr_norm_sum_adc_out*vis_factor),
                    marker='o',
                    edgecolor='red',
                    label='outlier'
                )
            
                ax.scatter(
                    xyz_o[:, 0],
                    xyz_o[:, 1],
                    xyz_o[:, 2],
                    c='r',
                    s=10,
                    marker='.',
                    linewidths=1
                )
    
        
        
            
            #for adc in np.round(np.linspace(round(min(adc),-1), round(max(adc), -1), ntouched_panels), -2):
            # for adc in adc[id_nearest]:
            #     ax.scatter( [], [], c= 'lightblue', marker='o',edgecolor='blue', alpha=0.3, s=exp(adc/max(adc)*vis_factor), label=str(int(adc)) )

            #leg1 = ax.legend(frameon=True, title='n$_{adc}$(X)+n$_{adc}$(Y)', loc='best')
            #ax.add_artist(leg1)
            
            #Set the legend
            fit_mark = mlines.Line2D([], [], color='blue',  marker='+', linestyle='None',
                            markersize=10, label='reco')
            inlier_mark = mlines.Line2D([], [], color='limegreen',  marker='o', linestyle='None',
                            markersize=10, label='inlier')
            outlier_mark = mlines.Line2D([], [], color='tomato',  marker='o', linestyle='None',
                            markersize=10, label='outlier')
          
            ax.legend(handles=[inlier_mark, outlier_mark], fontsize=16, loc='upper right')
    

        
        plt.show()
        

if __name__ == '__main__':
    
    
    parser=argparse.ArgumentParser(
    description='''Visualize reconstructed event''', epilog="""All is well that ends well.""")
    parser.add_argument('--telescope', '-tel', default=dict_tel["SNJ"], help='Input telescope name (e.g "tel_SNJ"). It provides the associated configuration.',  type=str2telescope)
    parser.add_argument('--reco_dir', '-i', default=None, help='/path/to/ransac/out/file/  One can input a data directory, a single datfile, or a list of data files e.g "--reco_data <file1.dat> <file2.dat>"', type=str)
    parser.add_argument('--out_dir', '-o', default='/path/to/output/directory', help='Path to analysis output', type=str) 
    args=parser.parse_args()    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f'outdir={args.out_dir}')
    print(args.reco_dir)
    pl = PlotEvt(telescope=args.telescope, recodir=args.reco_dir, outdir=args.out_dir)
    evtID_sample = sample(pl.evtID, 1) ##randomly draw event sample
    pl.get_points(evtID=evtID_sample)
    pl.plot3D(evtID=evtID_sample, isReco=True, isInlier=True, isOutlier=True) 
    plt.show()
    