#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 08 2021
@author: Raphaël Bajou
"""
# Librairies
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from datetime import datetime, timezone
import time 
import os
import inspect
import sys
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import seaborn as sns
import re
sns.set()
sns.set_theme(style="whitegrid", palette="pastel")
import logging
#personal modules
#Get location of script
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(filename))
config_path = os.path.normpath(script_path + os.sep + os.pardir)
import analysis as ana
from configuration import dict_tel, str2telescope
from processing import InputType
from tools.tools import create_subtitle


class Acceptance():
    def __init__(self, hitmap:ana.AnaHitMap, outdir:str, opensky_flux:np.ndarray, evttype:ana.EventType, theoric:dict=None ):
        #super().__init__(process_file=process_file, evttype=evttype,  label=label)
        self.hm = hitmap
        self.df = self.hm.df_DXDY
        self.label = self.hm.label
        self.outdir = outdir
        self.os_flux = opensky_flux
        self.evttype = evttype.name
     
        self.sconfig = [ c[3:] for c in self.df.columns if re.search('DX_(.+?)', c)]
       

        self.acc_dir=outdir
        
        self.acceptance = {conf: np.zeros(shape=self.hm.binsDXDY[c])  for c,conf  in enumerate(self.sconfig)}
        self.unc = {conf: np.zeros(shape=self.hm.binsDXDY[c])  for c,conf  in enumerate(self.sconfig)}

        self.ts = np.array(self.hm.df_reco['timestamp_s'].values)
        self.tns = np.array(self.hm.df_reco['timestamp_ns'].values)
        self.compute_acceptance()
        self.theoric = theoric

        #self.ratio = {conf: np.zeros(shape=self.hm.binsDXDY[c])  for c,conf in enumerate(sconfig)}

        logging.basicConfig(filename=os.path.join(self.outdir,f'acceptance_{self.label}_{time.strftime("%d%m%Y")}.log'), level=logging.INFO, filemode='w')
        timestr = time.strftime("%d%m%Y-%H%M%S")
        logging.info(timestr)
        logging.info(sys.argv)
        
    def compute_acceptance(self) -> None:
        time =self.ts + self.tns*10**(-8)
        time_sort = np.sort(time)
        dtime = np.diff(time_sort) 
        runDuration = np.sum(dtime[dtime < 3600])  # in second
        for conf, hDXDY in self.hm.hDXDY.items():
            evt_rate = hDXDY / runDuration
            self.acceptance[conf]= evt_rate/self.os_flux[conf]
            u_DXDY = np.sqrt(hDXDY)
            u_dt = 1 #s
            self.unc[conf] = self.acceptance[conf] *  np.sqrt((u_DXDY/hDXDY)**2 + (u_dt/runDuration)**2) 
            np.savetxt(os.path.join(self.acc_dir, '', f'acceptance_{conf}.txt'), self.acceptance[conf], delimiter='\t', fmt='%.5e')
            np.savetxt(os.path.join(self.acc_dir, '', f'unc_acc_{conf}.txt'), self.unc[conf], delimiter='\t', fmt='%.5e')
        # ####format dict of np.ndarray to dict of list to save in json
        # dict_out= { k:v.tolist() for k, v in self.unc.items()}
        # json_err = json.dumps(dict_out)
        # with open(self.outjson, 'w') as fp:
        #     fp.write(json_err)
    
    def plot_acceptance(self, az, ze):
        #if self.acceptance.size == 0: raise Exception("Please fill acceptance vector first.")
        fig = plt.figure(1, figsize= (16,9))
        nconfigs= self.acceptance.shape[0]
        gs = GridSpec(1,nconfigs)#, left=0.02, right=0.98, wspace=0.1, hspace=0.5)
        sns.set_style("whitegrid")
       # max_acc = np.max([np.max(acc) for acc in self.acceptance])
        create_subtitle(fig, gs[0, ::], f'Experimental acceptances: {self.label}')
    
        for i, (_,acc) in enumerate(self.acceptance.items()):
            max_acc = np.nanmax(acc)
            ax = fig.add_subplot(gs[0,i], aspect='equal')#, projection='3d') 
            #ax = Axes3D(fig)
            #ax.view_init(elev=25., azim=45)       
     
            im1 = ax.pcolor(np.arange(0,self.hm.binsDXDY[0]), np.arange(0,self.hm.binsDXDY[1]), 
                           acc,edgecolor='black', linewidth=0.5, cmap='viridis',  
                           shading='auto', vmin=0, vmax=max_acc) #norm=LogNorm(vmin=np.min(acc) , vmax=np.max(acc)))
            # im1 = ax.plot_surface(
            #     ze[i],
            #     az[i],
            #     acc,
            #     cmap="jet", #cm.coolwarm
            #     linewidth=0,
            #     antialiased=False,
            #     alpha=1
            # )
            
           
            ####2D plot
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im1, cax=cax, format='%.0e', orientation="vertical") # shrink=0.75,
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(label=u'Acceptance (cm².sr)', size=12)
        
            
            ax.set_xlabel('zenith $\\theta$ ($^{\circ}$)', fontsize=12)
            ax.set_ylabel('azimuth $\\phi$ ($^{\circ}$)', fontsize=12)
            ax.set_title(f'{self.sconfig[i]} config')
        
        gs.tight_layout(fig)       
        #plt.figtext(.5,.95, f"Experimental acceptances : {self.label}", fontsize=12, ha='center')
        plt.savefig(
            os.path.join(self.acc_dir,"", f"acceptance.png")
        )
        
        
    
    def plot_acceptance_3D(self, acc_exp, AZ, ZE, acc_th=None,label=None):
        
        if label is None: label=self.label
        #if self.acceptance.size == 0: raise Exception("Please fill acceptance vector first.")
        fig = plt.figure(1, figsize= (16,9))
        #, left=0.02, right=0.98, wspace=0.1, hspace=0.5)
        #gs = GridSpec(1,2)#, left=0.02, right=0.98, wspace=0.1, hspace=0.5)
        sns.set_style("whitegrid")
       # max_acc = np.max([np.max(acc) for acc in self.acceptance])
        #create_subtitle(fig, gs[0, ::], f'Experimental acceptances: {self.label}')
        
       # print(acc)
        max_acc_exp = np.nanmax(acc_exp)
        max_acc = max_acc_exp
        
        gs = GridSpec(1,2)
        
        if acc_th is not None:
           
            max_acc_th =  np.nanmax(acc_th)
            ratio = acc_exp / acc_th
            #u_ratio = ratio * err / acc_exp
            variance = acc_exp*(acc_th-acc_exp)/(acc_th**3) ####Binomial variance
            nz = (variance>=0)
            wi = 1/variance[nz]
            #print(variance)
            mean_ratio = np.sum(ratio[nz] * wi ) / np.sum(wi)
            #u_ratio = np.sqrt(variance) ##poisson unc
            mean_u_ratio = 1/np.sqrt(np.sum(wi))
            str_max = f'max_A_exp/max_A_th = {max_acc_exp:.2f}/{max_acc_th:.2f} = {max_acc_exp/max_acc_th:.2f}' 
            str_mean = f'<A_exp/A_th> +/- = {mean_ratio:.4f} +/- {2*mean_u_ratio:.4f} 95% C.L.'
            print(str_max, '\n', str_mean)
            logging.info(str_max)
            logging.info(str_mean)
            ax0 = fig.add_subplot(gs[0,0], projection='3d') 
            
            #ax0 = Axes3D(fig)
            ax0.view_init(elev=15., azim=45)       
           
            im0 = ax0.plot_surface(
                ZE,
                AZ,
                acc_th,
                cmap="jet", #cm.coolwarm
                linewidth=0,
                antialiased=False,
                alpha=1
            )
            max_acc= max_acc_th
            ax0.set_xlabel('zenith $\\theta$ ($^{\circ}$)', fontsize=14)
            ax0.set_ylabel('azimuth $\\varphi$ ($^{\circ}$)', fontsize=14)

        
        ax1 = fig.add_subplot(gs[0,1], projection='3d') 
        ax1.view_init(elev=15., azim=45)      
        im1 = ax1.plot_surface(
            ZE,
            AZ,
            acc_exp,
            cmap="jet", #cm.coolwarm
            linewidth=0,
            antialiased=False,
            alpha=1,
            vmin=0, vmax=max_acc
        )
    
        #3D plot 
        ax1.set_zlim(0, max_acc )
        #ax1.get_zaxis().set_visible(False)
        cbar = plt.colorbar(im1,  shrink=0.5, orientation="vertical")
        cbar.ax.tick_params(labelsize=12)
        if acc_th is not None: cbar.set_label(label='Integrated acceptance (cm².sr)', size=14)
        else: cbar.set_label(label='Experimental Acceptance (cm².sr)', size=14)

        
        ax1.set_xlabel('zenith $\\theta$ ($^{\circ}$)', fontsize=14)
        ax1.set_ylabel('azimuth $\\varphi$ ($^{\circ}$)', fontsize=14)
            
        gs.tight_layout(fig)       
        plt.savefig(
            os.path.join(self.acc_dir,"", f"acceptance_3D_{label}_{self.evttype}.png")
        )
        plt.close()
    
    
    def plot_ratio_acc(self, acc_exp, acc_th, az=None, ze=None, label=None):
        """Ratio maps between experimental and theoritical acceptances""" 
        if label is None: label=self.label
        r =  acc_exp/acc_th 

        #var = [ k*(n-k)/(n**3) for k, n in zip(acc_exp,acc_th) ] 
        #weight = [1/v for v in var]
        
        #nnan = [ ((~np.isnan(r)) & (r!=np.inf) & (~np.isnan(wi)) & (wi!=np.Inf)) for r, wi in zip(self.ratio, weight) ]
        #ratio_mean = [ np.sum(r[c] * wi[c]) / np.sum(wi[c]) for r, wi, c in zip(self.ratio, weight, nnan) 
        #u_ratio =   [ 1/np.sqrt(np.sum(wi[c])) for wi, c in zip( weight, nnan)]
        #str_eff = f'r_acc={eff_mean:.4f}+/-{u_eff:.4f}'
        
        fig= plt.figure(14, figsize= (16,9))
        gs = GridSpec(1, 1, left=0.05, right=0.95, wspace=0.2, hspace=0.5)
        #fig, axs = plt.subplots(nrows=1, ncols=len(ratios))
        fontsize = 36
        ticksize = 26
        labelbarsize = 40
     
        ax = fig.add_subplot(gs[0,0], aspect='equal')
        im = ax.imshow(r, cmap='jet',  vmin=0 , vmax=1 ) #shading='auto',
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)# format='%.0e')
        cbar.ax.tick_params(labelsize=ticksize)
        #cbar.set_label(label='$\\mathbf{\\mathcal{T}_{exp}}$ / $\\mathbf{\\mathcal{T}_{th}}$', size=labelbarsize)
        cbar.set_label(label='$\\mathcal{T}_{exp}$ / $\\mathcal{T}_{th}$', size=labelbarsize)
        ax.grid(False)
        
        locs = ax.get_xticks()[1:-1]  # Get the current locations and labels.
        new_x = [str(int(az[int(l)])) for l in locs]
        new_y = [str(int(ze[int(l)])) for l in locs]
        ax.set_xticks(locs)
        ax.set_xticklabels(new_x)
        ax.set_yticks(locs)
        ax.set_yticklabels(new_y)
        ax.tick_params(axis='both', labelsize=ticksize)
        ax.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=fontsize)
        ax.set_ylabel('zenith $\\theta$ [deg]', fontsize=fontsize)
        gs.tight_layout(fig)
        plt.savefig(
            os.path.join(self.acc_dir,"", f"ratio_acc_exp_vs_theo.pdf")
        )
        plt.close()
        
if __name__ == '__main__':
    # ==================================================================================================================
    _start_time = time.time()
    print("Start: ", time.strftime("%H:%M:%S", time.localtime()))
    main_dir = os.environ['HOME'] 
    parser=argparse.ArgumentParser(
    description='''This script computes the telescope acceptance for the 4-panels and the two 3-panels configurations. It outputs .txt file for each config.\nIt takes as optional inputs the .txt file from track reconstruction of open-sky calibration XY '--in_file', the '--param_dir' for Corsika flux and angular parameters, and the acceptance and plot output directories '--acc_dir' .''')
    #epilog="""All is well that ends well.""")
    parser.add_argument('--telescope', '-t', default=dict_tel['SNJ'], type=str2telescope, help='/path/to/calib.txt') 
    parser.add_argument('--input_calib', '-i', type=str, help='/path/to/output_CALIB_reco.txt', required=True) 
    parser.add_argument('--param_dir', default=os.path.join(config_path,'', 'AcquisitionParams' ,''), type=str, help='/path/to/matlab/fluxfiles/')
#    parser.add_argument('--out_dir', '-o', default=, type=str, help='/path/to/out/dir/')
    parser.add_argument('--label', '-l', default='test', help='')
    args=parser.parse_args()

    tel = args.telescope
    print(tel.name)
    inCalib = args.input_calib
    label = args.label
#    outDir = args.out_dir 
#    Path(outDir).mkdir(parents=True, exist_ok=True)

    param_dir = os.path.join(script_path, 'AcquisitionParams', tel.name)
    Corsika_OpenSkyFlux = sio.loadmat(os.path.join(param_dir, 'ExpectedOpenSkyFlux.mat'))
    OSFlux_calib_3p =Corsika_OpenSkyFlux['ExpectedFlux_calib_3p']#open-sky flux
#    OSFlux_calib_4p = Corsika_OpenSkyFlux['ExpectedFlux_calib_4p']
    #OSFlux_calib_3p = np.loadtxt(os.path.join(param_dir, 'ExpectedOpenSkyFlux_3p_guan.txt')) #
    #OSFlux_calib_4p = np.loadtxt(os.path.join(param_dir, 'ExpectedOpenSkyFlux_4p_guan.txt')) #
    #OSFlux_calib_4p = np.loadtxt("/Users/raphael/cosmic_flux/matlab/ExpectedFlux_gaisser.txt")


    sconfig = tel.configurations 
    nc = len(sconfig)
    
    #os_flux = { sconfig[0]: OSFlux_calib_3p,  sconfig[1]:OSFlux_calib_3p,  sconfig[2]:OSFlux_calib_4p }
   
    evttype = ana.EventType.MAIN
    input_type = InputType.SIM
    
    anaBaseCal = ana.AnaBase(recofile=ana.RecoData(file=inCalib, 
                                                                   telescope=tel,  
                                                                   input_type=InputType.REAL), 
                        evttype=evttype,
                        label=args.label)
    
    tlim_cal1 = ( int(datetime(2017, 4, 2, hour=00,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp()) , 
       int(datetime(2017, 4, 2, hour=16,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp())   )
    tlim_cal2 = ( int(datetime(2017, 4, 3, hour=00,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp()) , 
       int(datetime(2017, 4, 3, hour=18,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp())   )
    cut_tlim1 = ana.Cut("timestamp_s",vmin=tlim_cal1[0], vmax=tlim_cal1[1], label="dt")
    
    cut_tlim2 = ana.Cut("timestamp_s",vmin=tlim_cal2[0], vmax=tlim_cal2[1], label="dt")
    
    df = anaBaseCal.df_reco
    sigma=50
    res = df['quadsumres']/sigma**2
    ndf = df['npts']-2
    gof = res/ndf
    t = df['timestamp_s']
    cut_time = (( tlim_cal1[0] < t ) &  ( t < tlim_cal1[1] )) | (( tlim_cal2[0] < t ) &  ( t < tlim_cal2[1] )  )
    cut_gof = (( 0 < gof ) &  ( gof < 3.0 ))
    dfcut = df[cut_time & cut_gof]
    
    print(len(df.index), len(dfcut.index))
    #df[]
    #print(len(df.index))
    #df1 = cut_tlim1(anaBaseCal.df_reco)
    #df2 = cut_tlim2(anaBaseCal.df_reco)
    #print(len(df1.index)+len(df2.index))
    #df = pd.concat([df1,df2])
    #rate1 = ana.EvtRate(df1)
    #rate2 = ana.EvtRate(df2)
    
    rate_tot  = ana.EvtRate(df)
    rate_cut  = ana.EvtRate(dfcut)
    fig, ax = plt.subplots()
    ana.GoF(ax,dfcut)
    #rate_tot(ax)
    #rate_cut(ax)
    #rate1(ax)
    #rate2(ax)
    plt.show()
    exit()
    #####################
    hmCalib = ana.AnaHitMap(anabase=anaBaseCal, 
                                    input_type=InputType.REAL, 
                                    panels=tel.panels
                                    )
    
    acc_th_120 = np.loadtxt(os.path.join(script_path, "", "Acceptance/A_theo_16x16_120cm.txt"))
    acc_th_180 = np.loadtxt(os.path.join(script_path, "", "Acceptance/A_theo_16x16_180cm.txt"))
    acc_th = np.array([acc_th_120, acc_th_120, acc_th_180])
    A = Acceptance(hitmap=hmCalib,
                            outdir=outDir, 
                            evttype=evttype, 
                            opensky_flux=os_flux,
                            theoric=acc_th)
    angles = ana.AngCoord(telescope=tel)
    
    l_az_calib  = angles.az_os_1d
    L_AZ_CALIB = angles.AZ_OS_2D
    l_ze_calib   = angles.ze_os_1d
    L_ZE_CALIB = angles.ZE_OS_2D
    
    p = ana.PlotHitMap(hitmaps=[hmCalib], evttype=evttype, outdir=outDir)
    p.XY_map()
    p.DXDY_map()
    A.plot_acceptance_3D(acc_exp=A.acceptance['4p'], acc_th=acc_th[-1], err=A.unc['4p'], az =l_az_calib[-1], ze=l_ze_calib[-1])
    print(A.acceptance['4p'].shape, acc_th.shape)
    A.plot_ratio_acc(acc_exp=A.acceptance['4p'], acc_th=acc_th, az =l_az_calib, ze=l_ze_calib)                             
    r_mean, u_r = np.mean(A.ratio_mean), np.mean(A.u_ratio)
    str_r = f"r_mean_4p={r_mean:.4f}+/-{u_r:.4f}"
    print(str_r)
    logging.info(str_r)

        
    

    
    
    
    
   
    