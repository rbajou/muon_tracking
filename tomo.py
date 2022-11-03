#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphaël Bajou
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from scipy.interpolate import griddata
import scipy.ndimage
import os
import inspect
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from acceptance import Acceptance
from configuration import Telescope

#Get location of script
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(filename))
config_path = os.path.normpath(script_path + os.sep + os.pardir)

import analysis as ana
import acceptance as acc



class ImageFeature:
    def __init__(self, telescope:Telescope, file_3p, file_4p=None):
        self.tel = telescope
        self.file_3p = file_3p
        self.file_4p = file_4p
        self.dict = dict
        if self.file_3p is not None : self.get() 
    def get(self)-> None:
        val_3p = np.genfromtxt(self.file_3p, filling_values=0)
        #val_3p = ( np.isnan(arr_file_3p) )
        val_4p = val_3p
        if self.file_4p is not None: 
            val_4p = np.genfromtxt(self.file_4p, filling_values=0)
            #val_4p = ( np.isnan(arr_file_4p) )
        values = np.array([val_3p, val_3p, val_4p])
        self.dict   = { c:m for c, m in zip(self.tel.configurations, values)} 

class Topography:
    def __init__(self, telescope:Telescope, file_3p, file_4p=None):
        self.tel = telescope
        self.file_3p = file_3p
        self.file_4p = file_4p
        self.profile = np.ndarray
        if self.file_3p is not None : self.get()
        
    def get(self)-> None:
        prof_3p = np.loadtxt(self.file_3p) * 180/np.pi
        prof_4p = prof_3p
        if self.file_4p is not None: 
            prof_4p = np.loadtxt(self.file_4p)  * 180/np.pi
        profiles = np.array([prof_3p, prof_3p, prof_4p])
        self.profile = { c:p for c, p in zip(self.tel.configurations, profiles)} 
      

class Tomo:
    def __init__(self, telescope:Telescope,  hitmap:dict=None, label:str=None, outdir:str=None, info:dict=None, acceptance:dict=None, mask:dict=None, topography:dict=None,  evttype:ana.EventType=None):
        self.tel = telescope
        self.label = label
#        self.evttype = evttype.name
        self.acceptance = acceptance
        self.mask = mask
        self.topography = topography
        self.sconfig = list(telescope.configurations.keys())
        if outdir is not None:
            self.outdir = outdir
            self.flux_dir = os.path.join(self.outdir, "flux")
            self.op_dir   = os.path.join(self.outdir, "opacity")
            self.de_dir   = os.path.join(self.outdir, "density")
            Path(self.flux_dir).mkdir(parents=True, exist_ok=True)
            Path(self.op_dir).mkdir(parents=True, exist_ok=True)
            Path(self.de_dir).mkdir(parents=True, exist_ok=True)
        if hitmap is not None:
            self.hm = hitmap
            self.pixels  = { conf : self.tel.los[conf].shape[:-1] for conf in self.sconfig}
            self.flux    = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.unc_flux    = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.opacity = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.unc_opacity = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.density = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
            self.unc_density = { conf : np.zeros(shape=self.pixels[conf]) for conf in self.sconfig } 
        if info is not None: 
            self.runDuration = float(info['run_duration'])
            print(f'run_duration={self.runDuration/(24*3600):.1f} jours')

    def compute_flux(self, efficiency:dict=None, unc_eff:dict=None, hitmap:dict=None,  dT:float=None, acceptance:dict=None, unc_acc:dict=None):   
        if efficiency is None: efficiency = {c:np.ones(shape=self.pixels[c]) for c in self.sconfig}
        if hitmap is None: hitmap = self.hm
        if dT is None: dT = self.runDuration
        if acceptance is None:   
            acceptance=self.acceptance
        for ( (c, n), (_, acc), (_, eff) ) in zip(hitmap.items(), acceptance.items(), efficiency.items()):
            s = ((acc!=0.) & (eff!=0.))
            acc[~s], unc_acc[c][~s] = np.nan, np.nan 
            eff[~s], unc_eff[c][~s] = np.nan, np.nan 
            self.flux[c]= n / ( dT * acc * eff) 
            np.savetxt(os.path.join(self.flux_dir, '', f'flux_{c}.txt'), self.flux[c], delimiter='\t', fmt='%.5e')
            self.unc_flux[c] = self.flux[c] * np.sqrt((unc_acc[c]/acc)**2 + (np.sqrt(n)/n)**2 )#+ (unc_eff[c]/eff)**2 ) 
            np.savetxt(os.path.join(self.flux_dir, '', f'unc_flux_{c}.txt'), self.unc_flux[c], delimiter='\t', fmt='%.5e')
   

    def plot_flux(self, flux:dict, range:tuple, az:dict, ze:dict, topography:dict=None, mask:dict=None, colorbar:bool=True, sigma:list=[1,1], mode:str='constant', outdir:str=None, label:str=None):
        fontsize = 22#36
        ticksize = 16#26
        legendsize = 18#40
        if outdir is None: outdir= self.flux_dir
        if label is None: label=self.label
        for i, (c,f) in enumerate(flux.items()):
            fig0,ax0 = plt.subplots(figsize= (12,8))
            fig1,ax1 = plt.subplots(figsize= (12,8))
            ax0.grid(False)
            ax1.grid(False) 
            f[~np.isfinite(f)] =  np.nan
            if mask is not None : f[mask[c]] = np.nan
            #sigma = [1, 1] #[2, 2]
            fgaus = scipy.ndimage.filters.gaussian_filter(f, sigma, mode=mode)
            # Diplay filtered array
            fmin,fmax  = range
            fgaus[fgaus<=fmin]=np.nan
            a, z = az[c], ze[c]
            im0 = ax0.pcolor(a,z, f, cmap='jet_r',  shading='auto', norm=LogNorm(vmin=fmin, vmax=fmax)) #norm=LogNorm(vmin=np.min(f), vmax=np.max(f))#vmin=ZTrue_Tomo_Flux_3p1.min(), vmax=ZTrue_Tomo_Flux_3p1.max(),
            im1 = ax1.pcolor(a,z, fgaus, cmap='jet_r',  shading='auto', norm=LogNorm(vmin=fmin, vmax=fmax)) #norm=LogNorm(vmin=np.min(f), vmax=np.max(f))#vmin=ZTrue_Tomo_Flux_3p1.min(), vmax=ZTrue_Tomo_Flux_3p1.max(),
            ax0.set(ylim=[50, 90]) #deg
            ax1.set(ylim=[50, 90]) #deg
            if c=='4p': 
                ax0.set(ylim=[55, 85]) #deg
                ax1.set(ylim=[55, 85]) #deg
            if  topography is not None: 
                ax0.plot(topography[c][:,0], topography[c][:,1], linewidth=3, color='black')
                ax1.plot(topography[c][:,0], topography[c][:,1], linewidth=3, color='black')
            if colorbar : 
                #cax =  fig0.add_axes([0.87, 0.15, 0.03, 0.7]) #[0.15, 0.15, 0.5, 0.05]) #[left, bottom, length/width, height]
                divider0 = make_axes_locatable(ax0)
                cax0 = divider0.append_axes("right", size="5%", pad=0.05)
                cbar = fig0.colorbar(im0, cax=cax0, orientation="vertical")
                cbar.ax.tick_params(labelsize=ticksize)
                cbar.set_label(label='Flux [cm$^{-2}$.s$^{-1}$.sr$^{-1}$]', size=fontsize)
                fig0.subplots_adjust(right=0.85)
                divider1 = make_axes_locatable(ax1)
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                cbar1 = fig1.colorbar(im1, cax=cax1, orientation="vertical")
                cbar1.ax.tick_params(labelsize=ticksize)
                cbar1.set_label(label=u'Flux [cm$^{-2}$.s$^{-1}$.sr$^{-1}$]', size=fontsize)
                fig1.subplots_adjust(right=0.85)
                
            ax0.set_ylabel('zenith $\\theta$ [deg]', fontsize=fontsize)
            ax0.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=fontsize)
            ax0.tick_params(axis='both', which='major', labelsize=ticksize)
            ax0.invert_yaxis()
            ax0.set(frame_on=False)
            ax1.set_ylabel('zenith $\\theta$ [deg]', fontsize=fontsize)
            ax1.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=fontsize)
            ax1.tick_params(axis='both', which='major', labelsize=ticksize)
            ax1.invert_yaxis()
            ax1.set(frame_on=False)
            #fig0.tight_layout()
            #fig1.tight_layout()
            fig0.savefig(
                os.path.join(outdir,"", f"tomo_flux_{c}.png")
            )
            
            fig1.savefig(
                os.path.join(outdir,"", f"tomo_flux_gausfilter_{c}.png")
            )
            
        
        
        
        
    def plot_thickness(self, az:dict, ze:dict, app_thick:dict):
        fig = plt.figure(figsize= (12,8))
      
        gs = GridSpec(1, len(self.opacity))#, left=0.04, right=0.99, wspace=0.1, hspace=0.5)
        thick_min = min([ t[~np.isnan(t)].min() for _,t in app_thick.items()])
        thick_max = min([ t[~np.isnan(t)].max() for _,t in app_thick.items()])
        
        for i, (c,thick) in enumerate(app_thick.items()):
            ax = fig.add_subplot(gs[0,i],aspect="equal")
            ax.grid(False)
            a, z = az[c], ze[c]
            A, Z = np.meshgrid(np.linspace(a.min(), a.max(), 31  )  , np.linspace(z.min(), z.max(), 31  ))
            c = ax.pcolor(a, z, thick, cmap='jet', shading='auto', vmin=thick_min , vmax=thick_max )
            ax.invert_yaxis()
            cbar = fig.colorbar(c, ax=ax, shrink=0.75, format='%.0e', orientation="horizontal")
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(label=u'thickness [m]', size=12)
            #cbar.set_
            if i==0 :  ax.set_ylabel('zenith $\\theta$ [deg]', fontsize=12)
            ax.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=12)
            ax.set_title(f'{self.sconfig[i]} config')
            ax.set(frame_on=False)
        gs.tight_layout(fig)
        plt.figtext(.5,.95, f"thickness : {self.label}", fontsize=12, ha='center')
        plt.savefig(
            os.path.join(self.op_dir,"", f"thickness.png")
        )
        plt.close()
        
    def interpolate_opacity_bis(self, tomo_ze:dict, simu_ze:np.ndarray, op:np.ndarray, tomo_flux:dict, unc_tomo_flux:dict, simu_flux:np.ndarray, app_thick:dict, unc_tl:float=5, *args,**kwargs):
        points = np.zeros(shape=(60000, 2))#60000
        points[:, 0] = simu_ze.flatten() #zenith 1D array
        points[:, 1] = simu_flux.flatten()
        values = np.exp( np.log(10) * op ) #op=exp(log10 varrho)
        zmin, zmax = np.min(points[:, 0]),np.max(points[:, 0])
        fmin, fmax = np.min(points[:, 1]),np.max(points[:, 1])
        
        for (conf,ze), (_,flux), (_,uflux), (_,tl) in zip(tomo_ze.items(), tomo_flux.items(), unc_tomo_flux.items(), app_thick.items()):  
            shape= (self.tel.los[conf].shape[0], self.tel.los[conf].shape[1])
            grid_op =  np.zeros(shape=shape)
            in_range = np.logical_and((fmin < flux) & (flux < fmax), ~(np.isnan(flux)) & ~(np.isnan(tl)))
            
            grid_tmp = np.zeros(shape=flux.shape)
            grid_x, grid_y = ze, flux
            grid_tmp[in_range] = griddata(points, values.flatten(), (grid_x[in_range], grid_y[in_range]), *args, **kwargs) 
            #flux[~in_range] = np.nan
            grid_tmp[~in_range] = np.nan
            grid_op = grid_tmp 
            
            self.opacity[conf] = grid_op
            #grid_std = np.std(grid_op)
            self.mask = in_range
            # self.opacity[conf] = mean_op
            mean_rho = grid_op/tl
            # print(f"mean_rho==\n\n\n{mean_rho}")
            mean_rho[mean_rho==0] = np.nan
            # self.unc_opacity[conf] = std_op
            self.density[conf] = mean_rho
            
            
            # self.unc_density[conf] = unc_rho
            #return grid_op
    def interpolate_opacity(self, tomo_ze:dict, simu_ze:np.ndarray, op:np.ndarray, tomo_flux:dict, unc_tomo_flux:dict, simu_flux:np.ndarray, app_thick:dict, N:int=10, unc_tl:float=5, *args,**kwargs):
        points = np.zeros(shape=(60000, 2))#60000
        points[:, 0] = simu_ze.flatten() #zenith 1D array
        points[:, 1] = simu_flux.flatten()
        values = np.exp( np.log(10) * op ) #op=exp(log10 varrho)
        zmin, zmax = np.min(points[:, 0]),np.max(points[:, 0])
        fmin, fmax = np.min(points[:, 1]),np.max(points[:, 1])
        
        for (conf,ze), (_,flux), (_,uflux), (_,tl) in zip(tomo_ze.items(), tomo_flux.items(), unc_tomo_flux.items(), app_thick.items()):  
            shape= (self.tel.los[conf].shape[0], self.tel.los[conf].shape[1], N)
            rand_flux = np.zeros(shape=shape)
            grid_op =  np.zeros(shape=shape)
            mask_tl = (np.isnan(tl))
            simu_flux_edges = [ np.zeros(shape=flux.shape),  np.zeros(shape=flux.shape) ]
            ####flux data range
            in_range = np.logical_and((fmin < flux) & (flux < fmax), ~(np.isnan(flux)))
            flux[~in_range] = np.nan
            simu_flux_edges[0][~in_range], simu_flux_edges[1][~in_range] = np.nan, np.nan
            for i in range(flux.shape[0]):
                for j in range(flux.shape[1]):
                    if not np.isnan(flux[i,j]):
                        simu_flux_edges[0][i,j], simu_flux_edges[1][i,j]= simu_flux[simu_flux < flux[i,j] ].max(), simu_flux[simu_flux > flux[i,j] ].min()
            #print(simu_flux_edges)
            flux_width = simu_flux_edges[1] - simu_flux_edges[0]
            #print(flux_width)
            flux_c =  (simu_flux_edges[0]+simu_flux_edges[1])/2
            #print(flux_c)
            for n in range(N):
                rflux = np.copy(flux)
                rflux = np.random.normal(loc=flux, scale=uflux, size=flux.shape) 
                rflux[~in_range]=np.nan
                rflux.reshape(flux.shape)
#               in_range = np.logical_and((fmax<rflux), (rflux<fmin), (np.isnan(rflux)))
                #rflux[]=0
                grid_x, grid_y = ze, rflux
                rand_flux[:,:,n] = rflux
                #print(rflux)
                grid_x.reshape(self.tel.los[conf].shape[0],self.tel.los[conf].shape[1])
                grid_tmp = np.zeros(shape=rflux.shape)
                grid_tmp[in_range] = griddata(points, values.flatten(), (grid_x[in_range], grid_y[in_range]), *args, **kwargs) 
                grid_tmp[mask_tl] = np.nan
                grid_op[:,:,n]  = grid_tmp        
            mean_op = np.mean(grid_op,axis=2)
            std_op = np.std(grid_op,axis=2)
            mean_rho = mean_op/tl
            # print(f"mean_rho==\n\n\n{mean_rho}")
            mean_rho[mean_rho==0] = np.nan
            unc_rho = np.copy(mean_rho)
            s = ~np.isnan(mean_rho)
            unc_rho[s] =  mean_rho[s] * np.sqrt((unc_tl/tl[s])**2 + (std_op[s]/mean_op[s])**2 )
            unc_rho.reshape(self.tel.los[conf].shape[0],self.tel.los[conf].shape[1])
            self.opacity[conf] = mean_op
            self.unc_opacity[conf] = std_op
            self.density[conf] = mean_rho
            self.unc_density[conf] = unc_rho
            
            #return mean_op

    
    
    def plot_mean_density(self, quantity:str, val:dict, range:tuple, az:dict, ze:dict, topography:dict=None, sigma:tuple=None, mask:dict=None, outdir:str=None, label:str=None, cmap:str="jet", lognorm:bool=False, mode:str='mirror', threshold:float=None, crater:dict=None):
        """ Mean density maps """
        fontsize = 22#36
        ticksize = 16#26
        legendsize = 18#40
        vmin, vmax = range
        if outdir is None: outdir= self.op_dir
        if label is None: label=self.label
        for i, (c,rho) in enumerate(val.items()):
            fig0,ax0 = plt.subplots(figsize= (12,8))
            fig1,ax1 = plt.subplots(figsize= (12,8))
            a, z = az[c], ze[c]
            ax0.grid(False)
            ax1.grid(False)
            dphi = 35 #deg
            xlim, ylim = [np.median(a)-dphi, np.median(a)+dphi],[50, 90]
            ax0.set(xlim=xlim, ylim=ylim) #deg
            ax1.set(xlim=xlim, ylim=ylim) #deg
            if mask is not None: 
                if threshold is not None: 
                    aberrant = (rho >= threshold) 
                    rho[aberrant] = np.nan

                #https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
                rho[mask[c]] = np.nan
                rho_cp = np.copy(rho)
                rho_cp[np.isnan(rho)]=0
                rho_cp2 = scipy.ndimage.filters.gaussian_filter(rho_cp, sigma, mode=mode)
                rho_cp3= 0*rho.copy()+1
                rho_cp3[np.isnan(rho)]=0
                rho_cp4=scipy.ndimage.gaussian_filter(rho_cp3,sigma=sigma, mode=mode)
                rho_gaus=rho_cp2/rho_cp4
                rho.reshape(self.tel.los[c].shape[0], self.tel.los[c].shape[1])
                rho_gaus.reshape(self.tel.los[c].shape[0], self.tel.los[c].shape[1])
                rho_gaus[mask[c]] = np.nan
            else:rho_gaus = scipy.ndimage.filters.gaussian_filter(rho, sigma, mode=mode)
            
            
            
            if lognorm : 
                im0 = ax0.pcolor(a,z, rho, cmap=cmap,  shading='auto', norm=LogNorm(vmin=vmin, vmax=vmax))#vmin=ZTrue_Tomo_Flux_3p1.min(), vmax=ZTrue_Tomo_Flux_3p1.max(),
                im1 = ax1.pcolor(a,z, rho_gaus, cmap=cmap,  shading='auto', norm=LogNorm(vmin=vmin, vmax=vmax))#vmin=ZTrue_Tomo_Flux_3p1.min(), vmax=ZTrue_Tomo_Flux_3p1.max(),
            else: 
                im0 = ax0.pcolor(a,z, rho, cmap=cmap,  shading='auto', vmin=vmin, vmax=vmax)
                im1 = ax1.pcolor(a,z, rho_gaus, cmap=cmap,  shading='auto', vmin=vmin, vmax=vmax)
            np.savetxt(os.path.join(outdir, '', f'mean_density_gausfilter_{c}.txt'), rho_gaus , fmt='%.5e', delimiter='\t') 
            if  topography is not None: 
                topo = topography[c]
                ax0.plot(topo[:,0],topo[:,1], linewidth=3, color='black')
                ax1.plot(topo[:,0], topo[:,1], linewidth=3, color='black')
            if crater is not None:
                marker=MarkerStyle("*")
                dz=-0.5
                size=35
                for key, value in crater.items():    
                    if key =="SC": mark_col = "magenta"
                    elif key =="TAR": mark_col="white"
                    elif key =="G56": mark_col="lightgrey"
                    elif key =="BLK": mark_col="grey"
                    else: mark_col="black"
                    az_cr, ze_cr =  value[0], value[1]
                    ax0.plot(az_cr,ze_cr+dz, marker=marker, color=mark_col, markersize=size, markeredgewidth=1.5, markeredgecolor="black")
                    ax1.plot(az_cr,ze_cr+dz, marker=marker, color=mark_col, markersize=size, markeredgewidth=1.5, markeredgecolor="black")
                    ax0.annotate(key, ((az_cr+0.5, ze_cr+dz-1)), fontsize=14)
                    ax1.annotate(key, ((az_cr+0.5, ze_cr+dz-1)), fontsize=14)
                    az_topo = topo[:,0]
                    #arg = np.argmin(abs(az_topo-az_cr))
                    #min_phi, max_phi = az_topo[arg-1], az_topo[arg+1]
                    #ax0.axvspan(min_phi, max_phi)
                    #ax0.plot(az_cr, ze_cr, label=key, marker="*")
                        
            
            divider0 = make_axes_locatable(ax0)
            cax0 = divider0.append_axes("right", size="5%", pad=0.05)
            cbar = fig0.colorbar(im0, cax=cax0, orientation="vertical")
            cbar.ax.tick_params(labelsize=ticksize)
            cbar.set_label(label=quantity, size=fontsize)
            fig0.subplots_adjust(right=0.85)
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            cbar1 = fig1.colorbar(im1, cax=cax1, orientation="vertical")
            cbar1.ax.tick_params(labelsize=ticksize)
            cbar1.set_label(label=quantity, size=fontsize)
            fig1.subplots_adjust(right=0.85)
            ax0.set_ylabel('zenith $\\theta$ [deg]', fontsize=fontsize)
            ax0.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=fontsize)
            ax0.tick_params(axis='both', which='major', labelsize=ticksize)
            ax0.grid(True, which='both',linestyle='-', linewidth="0.25", color='grey')
            ax0.invert_yaxis()
            ax0.set(frame_on=True)
            ax1.set_ylabel('zenith $\\theta$ [deg]', fontsize=fontsize)
            ax1.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=fontsize)
            ax1.tick_params(axis='both', which='major', labelsize=ticksize)
            ax1.grid(True, which='both',linestyle='-', linewidth="0.25", color='grey')
            ax1.invert_yaxis()
            ax1.set(frame_on=True)
            fig0.savefig(
                os.path.join(outdir,"", f"mean_density_{c}.pdf")
            )
            fig1.savefig(
                os.path.join(outdir,"", f"mean_density_gausfilter_{c}.pdf")
            )
            
            #fig0.figtext(.5,.95, f"Mean density : {self.label}" , fontsize=12, ha='center')
            #fig1.figtext(.5,.95, f"Mean density : {self.label}" , fontsize=12, ha='center')
          
        plt.close()
        
        
  
    
        
    
    
    # def plot_density_variations(self, density:dict, range:tuple, az:dict, ze:dict, rho0:float=2.1, sigma:tuple=(1,1), outdir:str=None, label:str=None):
    #     """ Average density [g.cm$^{-3}$] maps """
        
    #     fig = plt.figure(figsize= (12,8))
    #     gs = GridSpec(1, len(density))#, left=0.04, right=0.99, wspace=0.1, hspace=0.5)
    #  
        
        
    #     #D_min = min([ np.min(D[~np.isnan(D)]) for _,D in self.density.items()])
    #     #D_max = min([ np.max(D[~np.isnan(D)]) for _,D in self.density.items()])
  
    #     vmin, vmax = range
    #     if outdir is None: outdir= self.op_dir
    #     if label is None: label=self.label
        
    #     for i, (conf,rho) in enumerate(density.items()):
    #         ax = fig.add_subplot(gs[0,i], aspect="equal")
    #         ax.grid(False)
    #         a, z = az[conf], ze[conf]
    #         m = ( ( np.isnan(rho) ) & ( ~np.isfinite(rho) ) )
    #         rho[m] = np.nan
    #         #c=ax.imshow(y, cmap='jet', interpolation='nearest')
    #         #A, Z = np.meshgrid(np.linspace(a.min(), a.max(), 31  )  , np.linspace(z.min(), z.max(), 31  ))
    #         rel_var = rho-rho0/rho0
    #         rel_var[m] = np.nan
    #         rv_gaus = scipy.ndimage.filters.gaussian_filter(rel_var, sigma, mode='constant')
    #         np.savetxt(os.path.join(self.op_dir, '', f'relvar_density_{self.label}_{self.evttype}_{conf}.txt'), rel_var , fmt='%.5e', delimiter='\t')
    #         # Diplay filtered array
    #         #rv_gaus[rv_gaus<=var_min]=np.nan
    #         #var_min, var_max = np.min(rel_var[~np.isnan(rel_var)]), np.max(rel_var[~np.isnan(rel_var)])
    #         c = ax.pcolor(a, z, rv_gaus, cmap='jet', shading='auto', vmin=vmin , vmax=vmax )
    #         if self.topography is not None: 
    #             ax.plot(self.topography[conf][:,0], self.topography[conf][:,1], linewidth=2, color='red')
    #         #ax.invert_yaxis()
    #         ax.set(ylim=[50, 90]) #deg
    #         #ticks = np.linspace(np.around(d_min,0),d_max, 6 )
    #         ticks = np.linspace(vmin, vmax,10)
    #         cbar = fig.colorbar(c, ax=ax, shrink=0.75, format='%.0e', orientation="horizontal", ticks=ticks )
    #         labels = [f"{t:.1f}" for t in ticks]
    #         cbar.ax.set_xticklabels(labels, fontsize=12)
    #         cbar.ax.tick_params(labelsize=12)
    #         #cbar.set_label(label=u'mean density $\\varrho/L$ [g.cm$^{-3}$]', size=12)#[mwe.m$^{-1}$]', size=12)
    #         cbar.set_label(label='relative variation mean density ($\\overline{\\rho}$-$\\rho_{0}$)/$\\rho_{0}$', size=12)#[mwe.m$^{-1}$]', size=12)
    #         ax.set_ylabel('zenith $\\theta$ [deg]', fontsize=12)
    #         ax.set_xlabel('azimuth $\\varphi$ [deg]', fontsize=12)
    #         ax.set_title(f'{self.sconfig[i]} config')
    #         ax.invert_yaxis()
    #         ax.set(frame_on=False)
    #     gs.tight_layout(fig)    
    #     plt.figtext(.5,.95, 'Relative mean density $\\overline{\\rho}$ variation : '+f'{label}'+'\n$\\rho_{0}$='+f'{rho0}'+' g.cm$^{-3}$' , fontsize=12, ha='center')
    #     plt.savefig(
    #         os.path.join(outdir,"", f"relvar_mean_dens_{label}_{self.evttype}.pdf")
    #     )
    #     plt.close()
    
     
        
