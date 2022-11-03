#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 2021
@author: Raphaël Bajou
"""
from argparse import ArgumentError
from dataclasses import dataclass, field
from typing import List, Union, Dict
from enum import Enum, auto
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
import inspect
import scipy.io as sio
from scipy import interpolate
from scipy.optimize import curve_fit
import pandas as pd
import pylandau
import os
from pathlib import Path
from datetime import datetime, date, timezone
import sys
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(filename))
#personal modules
from configuration import Telescope, Panel
from processing import InputType
main_dir = os.environ["HOME"]
from tools.tools import create_subtitle
from tools.advanced_fit import fit_landau_migrad, gaus


@dataclass
class Observable:
    name : str
    value: Union[dict, np.ndarray]
    error: Union[dict, np.ndarray] = None


@dataclass
class RecoData: 
    file : str
    input_type : InputType
    telescope : Telescope
    df : pd.DataFrame = field(init=False) 
    kwargs : Dict = field(default_factory=lambda : {"index_col":0, "delimiter":'\t'})
    index : List = field(default_factory=lambda: []) #index specific evts
    def __post_init__(self):
        if self.file.endswith(".csv") or self.file.endswith(".csv.gz"): 
            self.df= pd.read_csv(self.file, **self.kwargs) 
        else : raise ValueError("Input file should be a .csv file.") 
        
        if len(self.index) != 0 :      
            self.df = self.df[self.df.index.isin(self.index)]

class EventType(Enum):
    GOLD = auto()
    MAIN = auto()
    PRIMARY = auto()

class DataType(Enum):
    CALIB = auto()
    TOMO  = auto()

param_dir=os.path.join(script_path,'','AcquisitionParams')


class AcqVars : 
    def __init__(self, telescope:Telescope, param_dir:str=param_dir, mat_files:dict=None, tomo:bool=False):
        self.tel = telescope
        self.param_dir = Path(param_dir) / self.tel.name
        if not self.param_dir.exists(): raise ValueError("Parameter directory not found.")
        barwidths = { p.ID :  float(p.matrix.scintillator.width) for p  in self.tel.panels }
        self.sconfig= telescope.configurations
        if tomo: 
            acqVars_3p_file = Path(self.param_dir) / 'acqVars_3p.mat'
            acqVars_4p_file = Path(self.param_dir) / 'acqVars_4p.mat'
            if mat_files is not None: 
                if "3p" in mat_files: 
                    acqVars_3p_file = str(self.param_dir / mat_files["3p"]) 
                if "4p" in mat_files:  
                    acqVars_4p_file = str(self.param_dir / mat_files["4p"]) 
            
        self.az_os,  self.ze_os,  self.AZ_OS_MESH, self.ZE_OS_MESH={},{},{},{}
        self.az_tomo,  self.ze_tomo,  self.AZ_TOMO_MESH, self.ZE_TOMO_MESH={},{},{},{}
        for conf, l_panel in self.sconfig.items():
            first_panel = l_panel[0]
            last_panel = l_panel[-1]
            nbars = int(first_panel.matrix.nbarsX)
            panel_side = barwidths[first_panel.ID] * nbars
            #length= abs(first_panel.position[1] - last_panel.position[1] )
            length= abs(first_panel.position.z - last_panel.position.z )
            xlos = 2*first_panel.matrix.nbarsX-1
            ylos = 2*first_panel.matrix.nbarsY-1
            #OPEN-SKY angles values
            self.az_os[conf] = np.linspace(-180, 180, xlos)
            ze_max = np.around(np.arctan(panel_side/length)*180/np.pi, 1)
            self.ze_os[conf] = np.linspace(-ze_max, ze_max, ylos)
            self.AZ_OS_MESH[conf], self.ZE_OS_MESH[conf] = np.meshgrid(self.az_os[conf], self.ze_os[conf])
        
            if not tomo: continue 
            #TOMO angles values   
            if conf.startswith("3p"): 
                acqVars_3p_mat = sio.loadmat(acqVars_3p_file) 
                az_tomo, ze_tomo = acqVars_3p_mat['azimutAngleMatrix']*180/np.pi, acqVars_3p_mat['zenithAngleMatrix']*180/np.pi
            elif conf.startswith("4p"): 
                acqVars_4p_mat = sio.loadmat(acqVars_4p_file) 
                az_tomo, ze_tomo = acqVars_4p_mat['azimutAngleMatrix']*180/np.pi, acqVars_4p_mat['zenithAngleMatrix']*180/np.pi
            self.az_tomo[conf] = az_tomo
            self.ze_tomo[conf] = ze_tomo
            self.AZ_TOMO_MESH[conf], self.ZE_TOMO_MESH[conf] = np.meshgrid(self.az_tomo[conf],  self.ze_tomo[conf])
       
        
class Cut:
    def __init__(self, column:str, vmin:float=None, vmax:float=None, label:str=""):
        self.column = column            
        self.vmin = vmin
        self.vmax = vmax
        self.label = label
        self.evtID = None
    def __call__(self, df:pd.DataFrame):
        if self.column not in df.columns: raise ValueError(f"'{self.column}' not in '{df.columns}'")
        self.cut = None
        ix = df.index
        if self.vmin is not None and self.vmax is not None:
            self.cut = (self.vmin<df[self.column])  & (df[self.column]<self.vmax)
        elif self.vmin is not None:
            cut_vmin = (self.vmin<df[self.column])
            self.cut =  cut_vmin
        elif self.vmax is not None:
            self.cut (df[self.column]<self.vmax)
        else : self.cut = np.ones(shape=len(ix), dtype=bool)
        self.loss = len(self.cut[self.cut== False])/ len(ix)
        df_new = df[self.cut]
        self.evtID  = df_new.index
        return df_new
                
class AnaBase: 
    def __init__(self, recofile:RecoData, label:str, evtIDs:list=None, tlim:tuple=None, cuts:List=None):
        self.recofile = recofile
        self.df_reco= self.recofile.df
        self.label  = label
        #if tlim is None: self.tlim = (0, int(datetime(2032, 12, 31, hour=23,minute=59,second=59).replace(tzinfo=timezone.utc).timestamp()))
        #else : self.tlim = tlim
        #self.df_reco= self.df_reco.loc[ ( (self.tlim[0]<self.df_reco['timestamp_s']) & (self.df_reco['timestamp_s']<self.tlim[1]) )]

        self.df_reco_gold= self.df_reco.loc[self.df_reco['gold']==1.]
        
        if cuts is not None:
            for cut in cuts: self.df_reco = cut(self.df_reco)
            
        if evtIDs:
            df_tmp = self.df_reco.loc[evtIDs, : ] #Reconstructed primaries
            self.df_reco= df_tmp[~df_tmp.index.duplicated(keep='first')]
            self.evtIDs=evtIDs
        else: 
            self.evtIDs = list(self.df_reco.index)


def GoF(ax, df, column, color:str="blue", is_gold:bool=False, *args, **kwargs):
    if column not in df.columns: raise ValueError(f"'{column}' not in '{df.columns}'")
    entries, edges = np.histogram(df[column], *args, **kwargs)
    norm = np.sum(entries)
    fmain = entries/norm
    centers    = 0.5*(edges[1:]+edges[:-1])
    widths     =   edges[1:]-edges[:-1] 
    handles=[]
    hdl = ax.bar(centers, fmain, widths, color=color)
    handles.append(hdl)
    if is_gold: 
        entries_gold, edges = np.histogram(df[df["gold"]==1][column],  *args, **kwargs)
        fgold = entries_gold/norm
        centers    = 0.5*(edges[1:]+edges[:-1])
        widths     =   edges[1:]-edges[:-1] 
        hdl = ax.bar(centers, fgold, widths, color="orange")
        handles.append(hdl)
    ax.legend(handles=handles)
    ax.tick_params(axis='both', which='both', bottom=True)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.set_xlabel('goodness-of-fit', fontsize=14)
    ax.set_ylabel('probability density', fontsize=14)         
    plt.close()
def GoF_inlier_outlier(df:pd.DataFrame, outfile:str):
    #####GoF
    ####Check if 
    l_subcol = ["ninl", "noutl", "rchi2"]
    if not all(col in list(df.columns) for col  in l_subcol):
        return f"Check if {l_subcol} in {list(df.columns)}"
    
    range_gof= [0,10]

    chi2_noutl = { n : df[df["noutl"] == n]["rchi2"] for n in np.arange(0,6)}
    chi2_noutl[f">={6}"] = df[df["noutl"] >= 6]["rchi2"]
    fig = plt.figure(constrained_layout=True)
    ax = fig.subplot_mosaic([["outliers", "inliers"]], sharey=True)
    nbins=50
    entries, x = np.histogram(df["rchi2"], bins=nbins, range=range_gof)
    xc = (x[1:] + x[:-1]) /2
    w = x[1:] - x[:-1]
    norm = np.sum(entries)
    #GoF(ax=ax["outliers"], df=df, is_gold=True, column='rchi2', bins=nbins, range=[0,10])
    ax["outliers"].bar(xc, entries/norm, w ,color="blue", alpha=0.5, fill=False, edgecolor="blue" )
    bottom = np.zeros(len(entries))
    for i, ((k,chi2), color) in enumerate(zip(chi2_noutl.items(), [ "black", "purple", "blue", "green", "yellow", "orange", "red"] )):
        e, x = np.histogram(chi2, bins=nbins, range=range_gof )
        e_norm =e/norm
        ax["outliers"].bar(xc, e/norm, w ,color=color, alpha=0.5, label=k, bottom=bottom) #bottom=e[-1]
        bottom += e_norm
    ax["outliers"].legend(fontsize=14, title="#outliers")
    ax["outliers"].set_xlabel("GoF")
    ax["outliers"].set_ylabel("probability density")
    chi2_ninl = { n : df[df["ninl"] == n]["rchi2"] for n in np.arange(3,6)}
    chi2_ninl[f">={6}"] = df[df["ninl"] >= 6]["rchi2"]
    bottom = np.zeros(len(entries))
    for i, ((k,chi2), color) in enumerate(zip(chi2_ninl.items(), [ "green", "yellow", "orange", "red"] )):
        e, x = np.histogram(chi2, bins=nbins, range=range_gof )
        e_norm =e/norm
        ax["inliers"].bar(xc, e/norm, w ,color=color, alpha=0.5, label=k, bottom=bottom) #bottom=e[-1]
        bottom += e_norm

    ax["inliers"].legend(fontsize=14, title="#inliers")
    ax["inliers"].set_xlabel("GoF")
    plt.savefig(outfile)
    plt.close()
    
class EvtRate:
    def __init__(self, df:pd.DataFrame, dt_gap:int=3600):
        self.df = df
        if 'timestamp_s' not in self.df.columns : raise KeyError("Dataframe has no 'timestamp' column.")
        self.run_duration = 0
        time =self.df['timestamp_s'] + self.df['timestamp_ns']*10**(-8)
        self.nevts = len(time)
        time_sort = np.sort(time)
        dtime = np.diff(time_sort) 
        self.run_duration = np.sum(dtime[dtime < dt_gap])  # in second
        self.mean = 0
    def __call__(self, ax, width:float=3600, label:str="", tlim=None, t_off:float=0.):
        if tlim is None: tlim =  ( 0, 
        int(datetime(2032, 4, 2, hour=16,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp())   )
        t_min, t_max = tlim
        if t_min > t_max: raise ArgumentError("t_min > t_max")
        time = self.df['timestamp_s'][(t_min <= self.df['timestamp_s']) & (self.df['timestamp_s'] <= t_max)].values
        #mask = ((t_start <= time) & (time <= t_end))
        t_start = int(np.min(time))
        t_end = int(np.max(time))
        date_tot = [datetime.fromtimestamp(t+t_off) for t in time] #from timestamp in s to proper date
        date_start = datetime.fromtimestamp(t_start+t_off)
        date_start = date(date_start.year, date_start.month, date_start.day )#str(datetime.fromtimestamp(data_tomo[:, 1][0]))
        date_end = datetime.fromtimestamp(t_end+t_off)
        date_end = date(date_end.year, date_end.month, date_end.day )#str(datetime.fromtimestamp(data_tomo[:, 1][-1]))
        self.date_start, self.date_end=  date_start, date_end
        ntimebins = int(abs(t_end - t_start)/width) #hour
#        fig, ax = plt.subplots(figsize=(16,9))
        myFmt = mdates.DateFormatter('%d/%m %H:%M')
        ax.xaxis.set_major_formatter(myFmt) 
        print(f"run duration = {self.run_duration:1.3e}s = {self.run_duration/(3600):1.3e}h = {self.run_duration/(24*3600):1.3e}d")
        print(f"ntimebins = {ntimebins}")
        (nevt, dtbin, patches) = ax.hist(date_tot, bins=ntimebins, edgecolor='None', alpha=0.5, label=f"{label}\nnevts={len(time):1.3e}")
        self.mean = np.mean(nevt)#np.sum(dtbin_centers*nevt)/np.sum(nevt)
        self.std = np.std(nevt)#np.sum(nevt*(dtbin_centers-self.mean)**2)/(np.sum(nevt)-1)
        ax.set_ylabel('events', fontsize=23)
        ax.set_xlabel("time", fontsize=22)
        plt.figtext(.5,.95, f"Event time distribution from {str(date_start)} to {str(date_end)}", fontsize=14, ha='center')
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
       
        
        
        

class AnaHitMap: 
    def __init__(self, anabase:AnaBase, input_type:InputType, panels:List[Panel], evtIDs:list=None):#,  binsXY:tuple, rangeXY:tuple, binsDXDY:tuple, rangeDXDY:tuple):
        self.df_reco= anabase.df_reco
        self.input_type = input_type
        self.panels = panels
        self.label = anabase.label
        self.evtIDs = evtIDs       
        tel = anabase.recofile.telescope
        self.sconfig = list(tel.configurations.keys()) #panels config in telescope (i.e 4-panel configuration = 2*3p + 1*4p)
        self.DXDY = [] 
        self.hDXDY = np.ndarray
        #brut (X,Y) hit maps
        self.binsXY = [(pan.matrix.nbarsX, pan.matrix.nbarsY) for pan in self.panels]
        self.rangeXY = [(  ( 0, int(pan.matrix.nbarsX) * float(pan.matrix.scintillator.width) ), (0, int(pan.matrix.nbarsY) * float(pan.matrix.scintillator.width) ) ) for pan in self.panels] 
        #(DX,DY) maps : hits per telescope pixel (=line-of-sight) r_(DX,DY) 
        self.binsDXDY  = [ (los.shape[0], los.shape[1]) for _, los in tel.los.items()]
        w = tel.panels[0].matrix.scintillator.width
        self.rangeDXDY = [ ((np.min(los[:,:,0])*w, np.max(los[:,:,0])*w), (np.min(los[:,:,1])*w, np.max(los[:,:,1])*w) ) for conf,los in tel.los.items()] 
        
        colnames= ['gold', 'timestamp_s', 'timestamp_ns','time-of-flight','npts']#,'residuals']
        
        self.df=self.df_reco.copy()
        #####N.B different events might share the same index evtID, so better reindex the dataframe with the row loc
#        self.df.index = np.arange(0, len(self.df.index))
        self.df_DXDY = self.df[colnames]
    
        self.fill_dxdy()
    
        
    def fill_dxdy(self):
        pos_panels = [(f"X_{p.position.loc}", f"Y_{p.position.loc}") for p in self.panels] 
        df=self.df.copy()
        X = np.array([ df[x].values for i, (x,y) in enumerate(pos_panels) ])
        Y = np.array([ df[y].values for i, (x,y) in enumerate(pos_panels) ])
        self.XY = [X, Y]
        pos_panels = [(f"X_{p.position.loc}", f"Y_{p.position.loc}") for p in self.panels]
#        X,Y = self.XY
        s = np.array([ ( (Xmin< df[xpos]) & (df[xpos]<Xmax) &  (Ymin< df[ypos]) & (df[ypos]<Ymax) ) for i, ( (xpos,ypos), ((Xmin, Xmax), (Ymin, Ymax)) ) in enumerate( zip(pos_panels, self.rangeXY) ) ])
        if len(self.panels) == 4:  
            idx = [ df[s[0] & s[2]].index, df[s[1] & s[-1]].index, df[s[0] & s[-1]].index ]
            DX = np.array([X[0][s[0] & s[2] ] -  X[2][ s[0] & s[2] ], X[1][ s[1] & s[-1] ] -  X[-1][s[1] & s[-1] ], X[0][s[0] & s[-1]] -  X[-1][s[0] & s[-1]] ], dtype=object)
            DY = np.array([Y[0][s[0] & s[2] ] -  Y[2][ s[0] & s[2] ], Y[1][ s[1] & s[-1] ] -  Y[-1][s[1] & s[-1] ], Y[0][s[0] & s[-1]] -  Y[-1][s[0] & s[-1]] ], dtype=object)
            
        else : 
            idx = [ df[s[0] & s[-1]].index ]
            DX, DY = np.array([X[0][s[0] & s[-1]] -  X[-1][s[0] & s[-1]]]) , np.array([Y[0][s[0] & s[-1]] -  Y[-1][s[0] & s[-1]] ], dtype=object)

        pd.options.mode.chained_assignment = None  # default='warn'
        
        #print(df.loc[df.index.duplicated(keep=False)==True].sort_index().head)
        
        for i, c in enumerate(self.sconfig): 
            #print(c)
            #print("okokok")
            #print(len(idx[i]), len(DX[i]), len(DY[i]))
            #print(len(df.loc[(df.index.duplicated()==True ) & (df.index.isin(idx[i]))]))
            #print(len(df.loc[idx[i]]))
            #print(len(df.loc[df.index.isin(idx[i])]))
            #print(len(df.loc[df.index.duplicated()==True]))
            #print(len(df.loc[idx[i]].loc[idx[i].duplicated()].index))
            #print(len(set(df.loc[idx[i]])))
            #print(len(self.df_DXDY.loc[idx[i]]))
            #print(np.sort(idx[i]))
            #print(np.sort(df.index))
            #print(np.sort(self.df_DXDY.index))
            #print(len(list(set(idx[i]) & set(self.df_DXDY.index))))
            #print(len(list(set(idx[i]) & set(df.index))))
            #print(len(list((set(idx[i]) & set(df.index)))))
            #print(len(list(set(idx[i]).symmetric_difference(set(df.index)))))#& set(idx[i]))))
            #print(type(DX[i]))
            self.df_DXDY.loc[idx[i], f'DX_{c}'], self.df_DXDY.loc[idx[i], f'DY_{c}'] = DX[i], DY[i]
        
        self.sel = s
        self.DXDY = [DX, DY]
        self.hDXDY = { conf : np.histogram2d(DX[i], DY[i], bins=[bdx,bdy], range=[dxlim, dylim] )[0] for i, (conf,(bdx,bdy), (dxlim, dylim)) in enumerate( zip(self.sconfig, self.binsDXDY, self.rangeDXDY) ) }
        
class PlotHitMap:
    """Class to plot reconstructed trajectories hit maps."""
    def __init__(self, hitmaps:List[AnaHitMap], outdir:str, mask:dict=None) :#, reco_file:RecoData, outdir:str, label:str ):
        self.hitmaps = hitmaps
        self.sconfig = self.hitmaps[0].sconfig
        self.panels  = self.hitmaps[0].panels
        self.outdir = outdir
        self.mask = mask
        
    def XY_map(self, invert_yaxis:bool=True, transpose:bool=False):
        """Plot hit map for reconstructed primaries and all primaries"""

        if len(self.hitmaps)==0:
            raise Exception("Fill all XY vectors first")
       
        fig = plt.figure(0, figsize=(16,9))
        gs = GridSpec(len(self.hitmaps), len(self.panels), left=0.05, right=0.95, wspace=0.2, hspace=0.5)
       
        labels = [ hm.label for hm in self.hitmaps]
        #XYmaps = [ hm.XY for hm in self.hitmaps]
        for l, (name, hm) in enumerate(zip(labels, self.hitmaps)) : #dict_XY.items()):
            X, Y = hm.XY
            create_subtitle(fig, gs[l, ::], f'{name}')
            for i, p in enumerate(self.panels):
                ax = fig.add_subplot(gs[l,i], aspect='equal')
                if i== 0: ax.set_ylabel('Y')
                ax.set_xlabel('X')
                ax.get_yaxis().set_visible(False)
                #ax.set_title(f"\n {p.position.}")
                ax.grid(False)
                s = hm.sel[i]
                counts, xedges, yedges, im1 = ax.hist2d( X[i][s], Y[i][s],cmap='viridis', bins=hm.binsXY[i], range=hm.rangeXY[i] ) #im1 = ax.imshow(hXY[i])
                if invert_yaxis: ax.invert_yaxis()
                if transpose: 
                    if i== 0: ax.set_ylabel('X')
                    ax.set_xlabel('Y')
                    ax.hist2d(Y[i][s], X[i][s], cmap='viridis', bins=hm.binsXY[i], range=hm.rangeXY[i] ) #im1 = ax.imshow(hXY[i])
                divider1 = make_axes_locatable(ax)
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im1, cax=cax1, format='%.0e')
                #cbar.ax.tick_params(labelsize=8)        

        gs.tight_layout(fig)
        plt.savefig(
            os.path.join(self.outdir,"", f"brut_hit_maps.png")
        ) 
        plt.close()
        
    def DXDY_map(self, invert_xaxis:bool=True, invert_yaxis:bool=False, transpose:bool=False, fliplr:bool=False):
        if len(self.hitmaps)==0:
            raise Exception("Fill all DXDY vectors first")
        labels = [ hm.label for hm in self.hitmaps]
        fig = plt.figure(1, figsize= (16,9))
        nconfigs = len(self.sconfig)
        gs = GridSpec(len(self.hitmaps), nconfigs , left=0.05, right=0.95, wspace=0.2, hspace=0.1)
        for l, (name, hm) in enumerate(zip(labels, self.hitmaps)):
            DX, DY = hm.DXDY
            create_subtitle(fig, gs[l, ::], f'{name}')
            for c, conf in enumerate(self.sconfig):
                ax1 = fig.add_subplot(gs[l,c], aspect='equal')
                #if c == 0 : 
                #    ax1.set_ylabel('$\\Delta$Y [mm]', fontsize=16)
                #else : ax1.get_yaxis().set_visible(False)
                ax1.set_ylabel('$\\Delta$X [mm]', fontsize=16)
                ax1.set_xlabel('$\\Delta$Y [mm]', fontsize=16)
                DX_min, DX_max = hm.rangeDXDY[c][0]
                DY_min, DY_max = hm.rangeDXDY[c][1]
                h = hm.hDXDY[conf]
                if transpose: h = h.T
                if fliplr : h = np.fliplr(h)
                h[h==0] = np.nan 
                im1 = ax1.imshow(h, cmap='viridis', norm=LogNorm(vmin=1, vmax=np.max(h[~np.isnan(h)])), extent=[DX_min, DX_max, DY_min, DY_max] )
                ax1.grid(False)
                #hist, xedges, yedges, im1 = ax1.hist2d( DY[c], DX[c], edgecolor='black', linewidth=0., bins=hm.binsDXDY[c], range=hm.rangeDXDY[c], weights=None, cmap='viridis', norm=LogNorm(vmin=1, vmax=np.max(hm.hDXDY[c]) ) ) #    
                if invert_xaxis:  ax1.invert_xaxis()
                if invert_yaxis:  ax1.invert_yaxis()
                divider1 = make_axes_locatable(ax1)
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im1, cax=cax1, extend='max')
        gs.tight_layout(fig)
        plt.savefig(
            os.path.join(self.outdir, f"NHits_dXdY_map.png")
        )
        plt.close()
     
    def DXDY_map_az_ze(self, az=None, ze=None, invert_yaxis:bool=False, transpose:bool=False, fliplr:bool=False):
        ####NHits map in (Az,Ze) plane
        # ze_min, ze_max = np.min(ze), np.max(ze)
        # az_min, az_max = np.min(az), np.max(az)
        if len(self.hitmaps)==0:
            raise Exception("Fill DXDY histograms first")
        nconfigs= len(self.hitmaps[0].hDXDY)
        gs = GridSpec(len(self.hitmaps),nconfigs)

        #sns.set_style("whitegrid")
        fig = plt.figure(1, figsize= (16,9))
        for l,  hm in enumerate(self.hitmaps):
            hist = hm.hDXDY
            name = hm.label
            create_subtitle(fig, gs[l, ::], f'{name}')
            for c, conf in enumerate(self.sconfig):
                ax = fig.add_subplot(gs[l,c], aspect='equal')
                h = hist[conf]
                if transpose: h = h.T
                if fliplr : h = np.fliplr(h)
                im1 = ax.imshow(h, cmap='viridis', norm=LogNorm())
                #ax.grid(color='b', linestyle='-', linewidth=0.25)
                ax.grid(False)
                locs = ax.get_xticks()[1:-1]  # Get the current locations and labels.
                #print(az[c])
                new_x = [str(int(az[conf][int(l)])) for l in locs]
                new_y = [str(int(ze[conf][int(l)])) for l in locs]
                ax.set_xlabel('$\\varphi$ [deg]', fontsize=16)
                ax.set_xticks(locs, new_x)
                ax.set_ylabel('$\\theta$ [deg]', fontsize=16)
                ax.set_yticks(locs, new_y)
                if invert_yaxis : ax.invert_yaxis()
                ax.set_title(conf, fontsize=16)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im1, cax=cax, orientation='vertical')
        gs.tight_layout(fig)
        label = '_'.join([hm.label for hm in self.hitmaps ])
        plt.savefig(
            os.path.join(self.outdir, f"NHits_AzZe_map_{label}.png")
        )
        plt.close()


class Perf:
    
    def __init__(self, HitMapProcess:AnaHitMap, HitMapPrim:AnaHitMap, HitMapAllPrim:AnaHitMap, outdir:str):
       
        self.hitmap_process =  HitMapProcess
        self.hitmap_prim =  HitMapPrim
        self.hitmap_allprim = HitMapAllPrim
        self.label = HitMapProcess.label  
        self.outdir = outdir
        DX_reco, DY_reco = np.asarray(self.hitmap_process.DXDY)
        
        DX_prim, DY_prim = np.asarray(self.hitmap_prim.DXDY)
        self.nconfigs = DX_reco.shape[0]
        
        self.hDXDY_reco = self.hitmap_process.hDXDY#np.array([np.histogram2d(DX_reco[i], DY_reco[i], bins=[self.binsDX,self.binsDY], range=[self.DXlim, self.DYlim] )[0] for i in range(DX_reco.shape[0] )  ])
        self.hDXDY_prim = self.hitmap_prim.hDXDY #np.array([np.histogram2d(DX_prim[i], DY_prim[i], bins=[self.binsDX,self.binsDY], range=[self.DXlim, self.DYlim] )[0] for i in range(DX_reco.shape[0] )  ])
        self.hDXDY_allprim = self.hitmap_allprim.hDXDY
        self.reff = np.ndarray #reconstruction efficiency

    def plot_reco_efficiency(self, az, ze):    
        hDXDY_allprim = np.copy(self.hDXDY_allprim)
        hDXDY_allprim[hDXDY_allprim == 0.] = np.nan
        self.reff =  self.hDXDY_reco/hDXDY_allprim
        self.reff[np.isnan(self.reff)] = 0.
        fig = plt.figure(13, figsize= (12,6))
        gs = GridSpec(1, len(self.reff))
        
        for i, eff  in enumerate(self.reff) :
            DX_min, DX_max = self.hitmap_process.rangeDXDY[0]#np.min(ze),np.max(ze) #
            DY_min, DY_max = self.hitmap_process.rangeDXDY[1]#np.min(az),np.max(az)#self.hitmap_process.rangeDXDY[1]
        
            dx = np.linspace(DX_min, DX_max, self.hitmap_process.binsDXDY[0])
            dy = np.linspace(DY_min, DY_max, self.hitmap_process.binsDXDY[1])
            ax = fig.add_subplot(gs[0,i], aspect='equal')
           
            ax.grid(False)
            #ax.set_ylabel('$\\Delta$X', fontsize=16) #ax.set_ylabel('zenith $\\theta$ ($^{\circ}$)', fontsize=16)
            ax.set_ylabel('Zenith [deg]', fontsize=14)
            # ax.set_ylabel('azimuth $\\varphi$ ($^{\circ}$)', fontsize=16)
            
            #ax.get_yaxis().set_visible(False)
            #ax.set_xlabel('$\\Delta$Y', fontsize=16)
            ax.set_xlabel('Azimuth [deg]', fontsize=14)
            
            mean_eff = np.mean(eff)
            mean_err = self.error_eff(x=self.hDXDY_reco[i], y=self.hDXDY_allprim[i])
            print(f'mean_eff={mean_eff:.2f}$\\pm${mean_err:.2f}')
            #c = ax.pcolor(dy, dx, eff, cmap='jet',  vmin=eff.min(), vmax=1., label='$\\epsilon_{reco}$='+f'{mean_eff:.2f}') #norm=LogNorm(vmin=F_min, vmax=F_max)
            c = ax.pcolor(eff, cmap='jet',  vmin=eff.min(), vmax=1., label='$\\epsilon_{reco}$='+f'{mean_eff:.2f}') #norm=LogNorm(vmin=F_min, vmax=F_max)
            locs = ax.get_xticks()[:-1]  # Get the current locations and labels.
            new_x = np.array([int(az[i][int(l)]) for l in locs])#np.linspace(np.min(az[i]), np.max(az[i]), len(locs))
            new_y = np.array([int(ze[i][int(l)]) for l in locs])
            ax.set_xticks(locs, new_x)
            ax.set_yticks(locs, new_y)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(c, cax=cax, extend='max')
            #ax.legend(loc='upper right', fontsize=16)
            #ax.set_title('$\\langle$$\\epsilon_{reco}$$\\rangle$='+f'{mean_eff:.2f}$\\pm${mean_err:.2f}', fontsize=16)
        gs.tight_layout(fig)
        plt.figtext(.5,.95, "Muon reconstruction efficiency" , fontsize=12, ha='center')
        plt.savefig(
            os.path.join(self.outdir, f"reco_eff_mu.png")
        )
        plt.close()
        
    def error_eff(self, x, y):
        self.cut ((x!=0 ) & (y!=0 ))
        dx,dy = np.sqrt(x[cut]), np.sqrt(y[cut]) #poisson error
        new_x, new_y = x[cut], y[cut]
        err = new_x/new_y * np.sqrt((dx/new_x)**2+(dy/new_y)**2) 
      
        mean_err = np.mean(err)
        return mean_err
    
    
    def plot_angular_resolution(self, az:list, ze:list):
        
        DDX, DDY = self.DDXDDY
        nconfigs = DDX.shape[0]
       
        DX_min, DX_max = self.hitmap_process.rangeDXDY[0]
        DY_min, DY_max = self.hitmap_process.rangeDXDY[1]
        
        fig = plt.figure(13, figsize= (16,9))
        gs = GridSpec(2,nconfigs, left=0.05, right=0.95, wspace=0.25, hspace=0.25)  
        c = 'blue'
        for i in range(nconfigs):
            ax1 = fig.add_subplot(gs[0,i])#, aspect='equal')
            ax1.set_title(f"Config {i+1}")
            ax1.grid(False)
            ax1.set_xlabel('$\\Delta(\\Delta$X) = $\\Delta$X$_{reco}$ - $\\Delta$X$_{prim}$ [mm]', fontsize=16)
            ax1.set_ylabel('entries', fontsize=16)
            entries, bins, _ = ax1.hist(DDX[i], bins=100, range=[DX_min, DX_max], color=c)
            ax2= fig.add_subplot(gs[1,i])#, aspect='equal')
            ax2.grid(False)
            ax2.set_xlabel('$\\Delta(\\Delta$Y) = $\\Delta$Y$_{reco}$ - $\\Delta$Y$_{prim}$ [mm]', fontsize=16)
            ax2.set_ylabel('entries', fontsize=16)
            entries, bins, _ = ax2.hist(DDY[i], bins=100, range=[DY_min, DY_max], color=c)
            
        plt.figtext(.5,.95, "Muon primaries vs RANSAC reconstructed tracks" , fontsize=12, ha='center')
        plt.savefig(
            os.path.join(self.outdir, "", f"ecart_pixel_mm_prim_reco.png")
        )
        plt.close()
    
        dx = np.linspace(DX_min, DX_max, self.hitmap_process.binsDXDY[0])
        dy = np.linspace(DY_min, DY_max, self.hitmap_process.binsDXDY[1])
        #convert mm to deg
        #convert bar to deg
        fx = [interpolate.interp1d(dx, z) for z in ze ]
        fy = [interpolate.interp1d(dy, a) for a in az ]
        ####ANGULAR RESOLUTION
        fig = plt.figure(13, figsize= (16,9))
        gs = GridSpec(2, nconfigs, left=0.05, right=0.95, wspace=0.25, hspace=0.25)
        Ze_max = 0.25 * np.max(ze)
        Az_max = 0.25 * np.max(az)
        rangeZe  = (-Ze_max,Ze_max)
        rangeAz  = (-Az_max,Az_max) 
        #sns.set_theme(style="darkgrid")
        for i in range(nconfigs):
            cutx = (( DX_min<DDX[i] ) & (DDX[i] < DX_max ))
            cuty = (( DY_min<DDY[i] ) & (DDY[i] < DY_max ))
            DDX_new = fx[i]( DDX[i][cutx]) # in deg, theta (zenith) resolution
            DDY_new = fy[i]( DDY[i][cuty]) # in deg, phi (azimuth) resolution

            #Zenith
            ax1 = fig.add_subplot(gs[0,i])#, aspect='equal')
            ax1.set_title(f"Config {i+1}")
            ax1.grid(False)
            ax1.set_xlabel('$\\Delta$$\\theta$ [$^{\circ}$]', fontsize=16)
            ax1.set_ylabel('entries', fontsize=16)
            y, bins, _ = ax1.hist(DDX_new, bins=50, range=rangeZe, color=c)
            x = np.array([ (bins[i+1]+bins[i])/2 for i in range(len(bins)-1) ])
            fitrange = ( ( -Ze_max< x) & (x < Ze_max ) ) 
            new_x = x[fitrange]
            new_y = y[fitrange]
            N=len(x)
            amp = np.max(y)  
            mean = 0
            sigma = np.sqrt( sum( y*(x - mean)**2  )  /  ((N-1)/N*sum(y))  )
            popt,pcov  = curve_fit(gaus,new_x,new_y,p0=[amp,mean,sigma])
            perr = np.sqrt(np.diag(pcov))
            ax1.plot(new_x, gaus(new_x,*popt), label='$\\sigma$$_{\\theta}$='+f'{popt[2]:.2f}'+ '$^{\circ}$\n'+f'$\\pm${perr[2]:.2f}'+ '$^{\circ}$') #$\\mu$={popt[1]:.1f}'+ '$^{\circ}$'+'\n
            ax1.legend(loc='center right', fontsize=16)
            
            #Azimuth
            ax2= fig.add_subplot(gs[1,i])#, aspect='equal')
            ax2.grid(False)
            ax2.set_xlabel('$\\Delta$$\\varphi$ [$^{\circ}$]', fontsize=16)
            ax2.set_ylabel('entries', fontsize=16)
            y, bins, _ = ax2.hist(DDY_new, bins=50, range=rangeAz, color=c)# range=[-100, 100])
            x = np.array([ (bins[i+1]+bins[i])/2 for i in range(len(bins)-1) ])
            fitrange = ( (-Az_max < x) & (x < Az_max ) ) 
            new_x = x[fitrange]
            new_y = y[fitrange]
            N=len(x)
            amp = np.max(y)  
            mean = 0
            sigma = np.sqrt( sum( y*(x - mean)**2  )  /  ((N-1)/N*sum(y))  )
            popt,pcov  = curve_fit(gaus,new_x,new_y,p0=[amp,mean,sigma])
            perr = np.sqrt(np.diag(pcov))
            ax2.plot(new_x, gaus(new_x,*popt), label='$\\sigma$$_{\\varphi}$='+f'{popt[2]:.2f}'+ '$^{\circ}$\n'+f'$\\pm${perr[2]:.2f}'+ '$^{\circ}$')
            ax2.legend(loc='center right', fontsize=16)
            
        plt.figtext(.5,.95, "Muon reconstruction Angular resolution" , fontsize=12, ha='center')
        plt.savefig( 
            os.path.join(self.outdir, "", f"angular_resolution.png")
        )
        plt.close()
    
    

class AnaCharge:
    """Class to analyze charge deposits in scintillator panels"""
    def __init__(self, inlier_file:RecoData,  outlier_file:RecoData, outdir:str, label:str, evttype:EventType=None):
        self.inlier_file  = inlier_file
        self.outlier_file = outlier_file
        self.df_inlier  = self.inlier_file.df
        self.df_outlier = self.outlier_file.df
        #self.evttype = evttype.name
        #if  self.evttype == "GOLD" : 
        #    self.df_inlier = self.df_inlier.loc[self.df_inlier['gold']==1.]
        #    self.df_outlier = self.df_outlier.loc[self.df_outlier['gold']==1.]
        self.outdir = outdir 
        self.evtNo_reco = list(self.df_inlier.index)
        self.label = label 
        self.panels = inlier_file.telescope.panels
        self.binsDX = 2*self.panels[0].matrix.nbarsX-1
        self.binsDY = 2*self.panels[0].matrix.nbarsY-1
        self.ADC_XY_inlier  = [] #sum(ADC_X+ADC_Y) 
        self.ADC_XY_outlier = []
        self.fill_adc_arrays()
        
    def fill_adc_arrays(self):
        #pos_panels = [(f"X_{p.position}", f"Y_{p.position}") for p in self.panels]
        Z = np.sort(list(set(self.df_inlier['Z'])))
        sumADC_XY_in = self.df_inlier['ADC_X'] + self.df_inlier['ADC_Y'] 
        self.df_inlier = self.df_inlier.assign(sumADC_XY=pd.Series(sumADC_XY_in).values)
        sumADC_XY_out = self.df_outlier['ADC_X'] + self.df_outlier['ADC_Y'] 
        self.df_outlier = self.df_outlier.assign(sumADC_XY=pd.Series(sumADC_XY_out).values)
        
        self.ADC_XY_inlier = [ np.array(self.df_inlier.loc[self.df_inlier['Z'] ==z]["sumADC_XY"].values) for z in Z]
        self.ADC_XY_outlier = [ np.array(self.df_outlier.loc[self.df_outlier['Z'] ==z]["sumADC_XY"].values) for z in Z]
        
    def langau(x, mpv, eta, sigma, amp): return pylandau.langau(x, mpv, eta, sigma, amp, scale_langau=True) 
            

    def plot_adc_panels(self, ADC:dict, nbins:int=100, fcal:dict=None, unc_fcal:dict=None, xlabel:str='dQ [ADC]') : 

        fig, f_axs = plt.subplots(ncols=len(self.panels), nrows=1, figsize=(16,9))

        fontsize = 16#36
        ticksize = 12#26
        legendsize = 12#40
        
        if fcal is None: fcal = {p.ID : 1 for p in self.panels}
        
        for (tag, color, do_fit), adc in ADC.items():
            dict_par = {} ###fit parameters (value, error) / panel
            df_entries= pd.DataFrame(index=np.arange(0, nbins)) ###(entries, bin) / panel 
            for col, panel in enumerate(self.panels): 
               
                if len(adc[col]) == 0 : continue
                ax = f_axs[col]
                xmax_fig = np.mean(adc[col]) + 5*np.std(adc[col])
                ax.set_xlim(0, xmax_fig)
                xmax_fit = xmax_fig
                entries, bins_adc = np.histogram(adc[col],  range=(0,  xmax_fit), bins =  nbins)

                bins = bins_adc/fcal[panel.ID]
                widths = np.diff(bins)
                
                ax.bar(bins[:-1], entries, widths,color='None', edgecolor=color, label=f"{tag}")
                
                ax.set_xlabel(xlabel, fontsize=fontsize) 
                
                if fcal is not None and unc_fcal is not None: 
                    ax.set_xlabel('dE [MIP fraction]', fontsize=fontsize) 
                    ax.axvspan(1-unc_fcal[panel.ID]/fcal[panel.ID], 1+unc_fcal[panel.ID]/fcal[panel.ID], color='orange', alpha=0.2)
                    
                ax.set_ylabel('entries', fontsize=fontsize)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
                ax.tick_params(axis='both', labelsize=ticksize)
                #ax.legend(loc='upper right', fontsize=legendsize, title=f"{panel.position[0].name} ID{panel.ID}")
                ax.legend(loc='upper right', fontsize=legendsize, title=f"{panel.position.loc} ID{panel.ID}")
                #PYLANDAU FIT
                
                if not do_fit : continue 
                bin_centers = np.array([ (bins[i+1]+bins[i])/2 for i in range(len(bins)-1)])
            
                
                N = len(bin_centers)     
                mean = sum( bin_centers * entries) / sum(entries)#(n*max(nentries))#sum(np.multiply(bin_centers, nentries)) / n
                sigma = np.sqrt( sum( entries*(bin_centers - mean)**2  )  /  ((N-1)/N*sum(entries))  )
                rough_max = np.max( bin_centers[bin_centers>0][entries.argmax()] )#bin_centers[np.where(entries==max(entries))] )
                #fitrange =  ( ( rough_max*0.2 < bin_centers ) & (bin_centers< 3*rough_max ) )
                fitrange =  ( ( rough_max*0.2 < bin_centers ) & (bin_centers < xmax_fit ) )
                yerr = np.array([np.sqrt(n) for n in entries[fitrange] ]) 
                yerr[entries[fitrange]<1] = 1
                xfit = bin_centers[fitrange]
                yfit = entries[fitrange]
                bin_w = np.diff(bin_centers[fitrange] )
                #ofile = os.path.join(self.outdir, f"entries_dE_panel_{panel.ID}_{self.label}_{tag}_{self.evttype}.csv")
                #d_conv = {'entry': entries, 'bin': bin_centers}
                #df = pd.DataFrame(d_conv) 
                #df.to_csv(ofile)
                df_entries.loc[fitrange,f'bin_{panel.ID}'] = xfit
                df_entries.loc[fitrange,f'entry_{panel.ID}'] = yfit
                #mpv, eta, amp = int(rough_max), 0.5*sigma, np.max(entries)
                mpv, eta, amp = int(rough_max), sigma, np.max(entries)
                print(mpv, eta, sigma, amp)
                # coeff, pcov = curve_fit(pylandau.landau, xfit, yfit,
                #                 sigma=yerr,
                #                 absolute_sigma=False,
                #                 p0=(mpv, eta, amp))
                #                 # bounds=(1, 10000) )
                # perr = np.sqrt(np.diag(pcov))
       
                #ax.plot(xfit, pylandau.landau(xfit, *coeff), "-", color=color, label =f'MPV={coeff[0]:.0f}$\\pm${perr[0]:.0f}ADC\n$\\eta$={coeff[1]:.1f}$\\pm${perr[1]:.1f}ADC\nA={coeff[2]:.0f}$\\pm${perr[2]:.0f}') #
                #f_axs[col].plot(x, pylandau.landau(x, *coef), '-')
                # Fit the data with numerical exact error estimation
            # Taking into account correlations
                
                x=xfit
                y=yfit
                values, errors, m = fit_landau_migrad(
                                                x,
                                                y,
                                                p0=[mpv, eta, sigma, amp],#
                                                limit_mpv=(rough_max*0.8,rough_max*1.2), #(10., 100.)
                                                limit_eta=(0.3*eta,1.5*eta), #(0.8*eta,1.2*eta)
                                                limit_sigma=(0.3*sigma,1.5*sigma), #(0.8*sigma,1.2*sigma)
                                                limit_A=(0.8*amp,1.2*amp) #(0.8*amp,1.2*amp)
                                                ) 
               
                
                ## PLOT and SAVE fit result, and histogram entries
                entries = pylandau.langau(x, *values)
                yerr = np.sqrt(entries)
                ax.plot(x,y)
                ax.errorbar(x, y, yerr, fmt='.', color='r')
                param = ['MPV', 'eta', 'sigma', 'A']
                sym_err = [np.max(np.abs(e)) for e in errors] 
                for par, val, err in zip(param, values, sym_err) : dict_par[(str(panel.ID), par)]= [f'{val:.5f}', f'{err:.5f}']
                dict_par[(str(panel.ID), "xmin")]= (np.min(xfit),bin_w[0])
                dict_par[(str(panel.ID), "xmax")]= (np.max(xfit),bin_w[-1])
                #dict_par['entries'] = (int(np.sum(entries)), 0)
                param = ['MPV', '$\\eta$', '$\\sigma$', 'A']
                label = ""#"$\\bf{"+ name +"}$\n"
                label += ' '.join('{}={:0.1f}$\\pm${:0.1f} ADC\n'.format(p, value, error) for p, value, error in zip(param[:-1], values[:-1], sym_err[:-1]  )   )
                ax.plot(x, pylandau.langau(x, *values), '-', label=label+'{}={:0.1f}$\\pm${:0.1f}'.format(param[-1], values[-1],sym_err[-1]), color=color )
            
                ax.legend(loc='best', fontsize=legendsize, title=f"Panel {panel.ID}")
                ax.set_ylim(0, 1.2*amp)
            
            xmax_fig = xmax_fit
            #ax.set_xlim(0, xmax_fig)
            
            ofile_ent = os.path.join(self.outdir, f"entries_dQ_{tag}.csv")
            df_entries.to_csv( ofile_ent, sep='\t')
            ofile_par = os.path.join(self.outdir, f'fit_dQ_{self.label}_{tag}.csv')
            df_par = pd.DataFrame.from_dict(dict_par, columns=['value', 'error'],orient='index')
            df_par.to_csv( ofile_par, sep='\t')
            
        fig.tight_layout()
        plt.savefig(
            os.path.join(self.outdir, "",  f"charge_distributions.png")
        )
        plt.close()


    def scatter_plot_dQ(self, dQx:dict, dQy:dict, rangex:tuple=None, rangey:tuple=None, nbins:int=100) : 
        for i, (((tagx, colorx, do_fitx), valx), ((tagy, colory, do_fity),valy))  in enumerate( zip(dQx.items(), dQy.items() )) :
          
            fig = plt.figure(0, figsize= (16,9))
            gs = fig.add_gridspec(2, 2,  width_ratios=(7, 2), height_ratios=(2, 7),
                            left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.05, hspace=0.05)

            ax = fig.add_subplot(gs[1, 0])
            ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            #if fcal[panel.ID]!=1 : 
                #ax.set_xlabel('dE [MIP fraction]', fontsize=fontsize) 
            atx = AnchoredText('dEx',
                        prop=dict(size=14), frameon=True,
                        loc='upper right',
                        )
            atx.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax_histx.add_artist(atx) 
            ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
            aty = AnchoredText('dEy',
                        prop=dict(size=14), frameon=True,
                        loc='upper right',
                        )
            aty.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax_histy.add_artist(aty) 
            ax_histx.tick_params(axis="x", labelbottom=False)
            ax_histx.set_ylabel("entries", fontsize=10)
            ax_histx.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            ax_histx.set_yscale('log')
            ax_histy.tick_params(axis="y", labelleft=False)
            ax_histy.set_xlabel("entries", fontsize=10)
            ax_histy.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            ax_histy.set_xscale('log')
        # the scatter plot:
        #data = np.vstack([x, y])
        #kde = gaussian_kde(data)
    
        
        #Xgrid, Ygrid = np.meshgrid(x, y)
        #Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    # Plot the result as an image
        #ax.imshow(Z.reshape(Xgrid.shape),
            #    origin='lower', aspect='auto',
            #    extent=[0, max(x), 0, max(y)],
            #    cmap='Blues')
        #cb = plt.colorbar()
        #cb.set_label("density")
        
    
            
            if rangex is None and rangey is None : 
                rangex, rangey= (-1,np.max(valx)), (-1,np.max(valy))

  
            entries_x, bins_x, _ = ax_histx.hist(valx, bins=nbins, range =rangex, color ='lightgreen', alpha=1., label='X', edgecolor='none')
            entries_y, bins_y, _ = ax_histy.hist(valy, bins=nbins, range =rangey, orientation='horizontal', color = 'lightgreen', alpha=1., label='Y', edgecolor='none')
            bins_center_x = np.array([ (bins_x[i+1]+bins_x[i])/2 for i in range(len(bins_x)-1)])
            bins_center_y = np.array([ (bins_y[i+1]+bins_y[i])/2 for i in range(len(bins_y)-1)])
            
            gamma = 0.3
            ax.hist2d(valx, valy, bins=nbins, range=None, norm=mcolors.PowerNorm(gamma), cmap='viridis') #mcolors.LogNorm(vmin=1, vmax=max_)
            plt.figtext(.5,.95, self.label, fontsize=12, ha='center')
            plt.savefig(
                os.path.join(self.outdir, "",  "scatter_ADC_.png")
            )
            
            plt.close()



    
if __name__ == '__main__':
    print("ok")