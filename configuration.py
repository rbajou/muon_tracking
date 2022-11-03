#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dataclasses import dataclass, field
from typing import List
from enum import Enum, auto
import numpy as np 
import os
import inspect
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(filename))
import argparse
import json
    
@dataclass(frozen=True)
class Scintillator:
    type: str 
    #dimensions in mm
    length: float 
    width: float
    thickness: float
    def __str__(self): return f"{self.type}"

@dataclass
class ChannelMap:
    file: str #= os.path.join(script_path, "", "ChannelNoXYMaps/mapping16x16.dat")
    
    def map(self):
        """
        Read the .dat mapping file and fill a dictionary 'dict_ch_to_bar' :
        { keys = channel No (in [0;63]) : values = X or Y coordinate (e.g 'X01') }
        """
        ####convert ch to chmap
        with open(self.file, 'r') as fmap:
           self.dict_ch_to_bar = json.loads(fmap.read())
        self.dict_ch_to_bar = {int(k):v for k,v in self.dict_ch_to_bar.items()}
        self.dict_bar_to_ch = { v: k for k,v in self.dict_ch_to_bar.items()  }
        #OLD
        # self.dict_ch_to_bar =dict()
        # with open( self.file )as f:
        #     lines = f.readlines()[0:]
        #     #print(lines)
        #     for line in lines:
        #         l = line.split(" ")
        #         l[7] = l[7].rstrip('\n')
        #         try:
        #             self.dict_ch_to_bar[int(l[2])].append(l[0])
        #             self.dict_ch_to_bar[int(l[7])].append(l[5])
        #         except KeyError:
        #             self.dict_ch_to_bar[int(l[2])] = l[0]
        #             self.dict_ch_to_bar[int(l[7])] = l[5]
        
        #     self.dict_bar_to_ch = { v: k for k,v in self.dict_ch_to_bar.items()  } # if len(v)==2 else v[1]: k 
        
        self.channels = list(self.dict_ch_to_bar.keys())
        self.bars = list(self.dict_ch_to_bar.values())
                
    def __post_init__(self):
        self.map()

    def __str__(self):
        return f'ChannelMap: {self.dict_ch_to_bar}'
    
@dataclass(frozen=True)
class Matrix:
    version : int
    scintillator : Scintillator
    nbarsX : int
    nbarsY : int  
    wls_type : str
    fiber_out : str
    def __str__(self):
        return f"v{self.version} with ({self.nbarsX}, {self.nbarsY}) {self.scintillator} scintillators"

class PositionEnum(Enum):
    Front = auto()
    Middle1 = auto()
    Middle2 = auto()
    Rear = auto()

class Position:
    def __init__(self, loc:PositionEnum, z:float):
        self.loc = loc.name 
        self.z = z  #in mm

@dataclass(frozen=True)
class Panel:
    matrix : Matrix 
    ID : int
    channelmap : ChannelMap 
    position: Position #Tuple[PositionEnum, float] 

@dataclass(frozen=True)
class PMT:
    ID : int
    panel : List[Panel] 
    channelmap : ChannelMap
    type : str = field(default='MAPMT')
        
@dataclass
class Telescope:
    name : str
    panels : List[Panel]
    PMTs : List[PMT] = field(default_factory=list)
    utm : np.ndarray = field(init=False) #coordinates (easting, northing, altitude)
    azimuth : float = field(init=False)
    elevation : float = field(init=False)
    
    def __str__(self):
        matrices = [p.matrix for p in self.panels]
        versions = [m.version for m in matrices]
        v = {f'v{v}': versions.count(v) for v in  set(versions)}
        return f"Telescope {self.name}: {len(self.panels)} matrices {v}"#type {','.join(set([m.version for m in matrices]))} "
    def __post_init__(self):
        z_front, z_rear =  self.panels[0].position.z, self.panels[-1].position.z
        object.__setattr__(self, 'length',  abs(z_front-z_rear) )
        if len(self.panels) == 3:
            object.__setattr__(self, 'configurations',  {'3p1':self.panels} )
        elif len(self.panels) == 4:
            object.__setattr__(self, 'configurations',  {'3p1':self.panels[:3], 
                                                         '3p2':self.panels[1:], 
                                                         '4p':self.panels} )
        else: raise ValueError("Unknown panel configuration...")
        #lines-of-sight
        object.__setattr__(self, 'los', {conf: self.get_los(pan[0],pan[-1]) for conf, pan in self.configurations.items() })
        #object.__setattr__(self, 'pixel_xy', {conf: self.get_pixel_xy(pan[0],pan[-1]) for conf, pan in self.configurations.items() })
    
    def get_los(self,front_panel, rear_panel):
        """
        Lines-of-sight referenced as (DX,DY) couples
        """
        nbarsXf, nbarsYf  = front_panel.matrix.nbarsX,front_panel.matrix.nbarsY
        nbarsXr, nbarsYr  = rear_panel.matrix.nbarsX,rear_panel.matrix.nbarsY
        barNoXf, barNoYf = np.arange(1, nbarsXf+1),np.arange(1, nbarsYf+1)
        barNoXr, barNoYr = np.arange(1, nbarsXr+1),np.arange(1, nbarsYr+1)
        DX_min, DX_max = np.min(barNoXf) - np.max(barNoXr) ,  np.max(barNoXf) - np.min(barNoXr) 
        DY_min, DY_max = np.min(barNoYf) - np.max(barNoYr) ,  np.max(barNoYf) - np.min(barNoYr) 
        mat_los = np.fliplr( np.mgrid[DX_min:DX_max+1:1, DY_min:DY_max+1:1].reshape(2,-1).T.reshape(2*nbarsXf-1,2*nbarsYf-1,2) )
        return mat_los

    '''
    def get_pixel_xy(self,front_panel, rear_panel):
        """
        Position XY pixels
        """
        func = lambda xf,xr: xf-xr
        nbarsXf, nbarsYf  = front_panel.matrix.nbarsX,front_panel.matrix.nbarsY
        nbarsXr, nbarsYr  = rear_panel.matrix.nbarsX,rear_panel.matrix.nbarsY
        barNoXf, barNoYf = np.arange(1, nbarsXf+1),np.arange(1, nbarsYf+1)
        barNoXr, barNoYr = np.arange(1, nbarsXr+1),np.arange(1, nbarsYr+1)
        DX_min, DX_max = np.min(barNoXf) - np.max(barNoXr) ,  np.max(barNoXf) - np.min(barNoXr) 
        DY_min, DY_max = np.min(barNoYf) - np.max(barNoYr) ,  np.max(barNoYf) - np.min(barNoYr) 
        res_dx = np.tile(np.mgrid[DX_min:DX_max+1:1],  (2*nbarsXf-1, 1)).T
        res_dy = np.tile(np.mgrid[DY_min:DY_max+1:1],  (2*nbarsYf-1, 1))
        mat_dx, mat_dy = np.zeros(shape=(res_dx.shape[0],res_dx.shape[1],2)), np.zeros(shape=(res_dy.shape[0],res_dy.shape[1],2))
        for i in range(1,nbarsXf+1): 
            for j in np.flip(range(1,nbarsXf+1)): 
                mat_dx[res_dx==func(i,j),:] = [i,j]
                mat_dy[res_dy==func(i,j),:] = [j,i]
        mat = np.concatenate((mat_dx, mat_dy), axis=2)
        return mat
    '''
    
    def plot3D(self, ax, position):
        zticks=[]
        for p in self.panels:
            w  = float(p.matrix.scintillator.width)
            nbarsX = int(p.matrix.nbarsX)
            nbarsY = int(p.matrix.nbarsY)
            sx = w*nbarsX 
            sy = w*nbarsY
            X  = np.linspace(position[0], position[0]+sx , nbarsX+1 )
            Y = np.linspace(position[1], position[1]+sy,  nbarsY+1 )
            X, Y = np.meshgrid(X, Y)
            zpos = position[2]
            Z = np.ones(shape=X.shape)*(zpos + p.position.z)
            ax.plot_surface(X,Y,Z, alpha=0.2, color='darkturquoise', edgecolor='turquoise' )
            zticks.append(Z[0,0])
        
        panel_side=float(self.panels[0].matrix.nbarsX)*float(self.panels[0].matrix.scintillator.width)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_xticks(np.linspace(0, panel_side, 5))
        ax.set_yticks(np.linspace(0, panel_side, 5))
        ax.set_zlabel("Z [mm]")
        ax.set(xlim=[0,panel_side], ylim=[0,panel_side], zlim=[0, self.length])   
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.invert_zaxis()
        ax.set_zticks(zticks)
        return ax



scint_Fermi = Scintillator(type="Fermilab", length=800, width=50, thickness=7 )
scint_JINR = Scintillator(type="JINR", length=800, width=25, thickness=7 )
matrixv1_1 = Matrix(version="1.1",  scintillator=scint_Fermi, nbarsX=16, nbarsY= 16, wls_type="BCF91A",fiber_out="TR644 POM")
matrixv2_0 = Matrix(version="2.0",  scintillator=scint_JINR, nbarsX=32, nbarsY= 32, wls_type="Y11",fiber_out="TR644 POM")
ChMap16 = ChannelMap(file=os.path.join(script_path, "", "ChannelNoXYMaps/mapping16x16.json"))
ChMap32 = ChannelMap(file=os.path.join(script_path, "", "ChannelNoXYMaps/mapping32x32.json"))
####SNJ: SuperNainJaune GW Parking 2019
npanels_SNJ=4
matrix_SNJ = [matrixv1_1 for _ in range(npanels_SNJ)]
pos_SNJ = [ Position(PositionEnum.Front,0), Position(PositionEnum.Middle1, 600), Position(PositionEnum.Middle2, 1200), Position(PositionEnum.Rear, 1800)]
panels_SNJ = [ Panel(matrix=m, ID=9+i, position=pos, channelmap=ChMap16) for i, (m, pos) in enumerate(zip(matrix_SNJ, pos_SNJ)) ]
pmt_SNJ =  [ PMT(ID=i+9, panel=pan, channelmap=ChMap16) for i, pan in enumerate(panels_SNJ) ]
tel_SNJ= Telescope(name = "SNJ", panels=panels_SNJ, PMTs=pmt_SNJ)

#####SBR: SuperBaronRouge GW Rocher Fendu 2021-2022 4 matrices = 4 * v1.1
ChMapSBR = ChannelMap(file=os.path.join(script_path, "", "ChannelNoXYMaps/mapping16x16_SBR.json"))
npanels_SBR = 4
matrix_SBR = [matrixv1_1 for _ in range(npanels_SBR)]
pos_SBR = [ Position(PositionEnum.Front,0), Position(PositionEnum.Middle1, 600), Position(PositionEnum.Middle2, 1200), Position(PositionEnum.Rear, 1800) ]
panel_id_SBR = [0, 2, 1, 3]
panels_SBR = [ Panel(matrix=m, ID=panID, position=pos, channelmap=ChMapSBR) for (m, pos, panID) in zip(matrix_SBR, pos_SBR, panel_id_SBR) ]
pmt_SBR =  [ PMT(ID=6, panel=[panels_SBR[0], panels_SBR[1]], channelmap=ChMapSBR), PMT(ID=7, panel=[panels_SBR[2],panels_SBR[-1]], channelmap=ChMapSBR) ]
tel_SBR= Telescope(name = "SBR", panels=panels_SBR, PMTs = pmt_SBR)

#####COPAHUE LaPLata 2021
ChMap_COP = [ChMap32, ChMap16, ChMap32]
npanels_COP = 3
matrix_COP = [matrixv2_0, matrixv1_1, matrixv2_0]
pos_COP = [ Position(PositionEnum.Front,0), Position(PositionEnum.Middle1, 600), Position(PositionEnum.Middle2, 1200)]
panel_id_COP = [0, 1, 2] 
panels_COP = [ Panel(ID=panID, matrix=m, position=pos, channelmap=map) for (m, pos, panID, map) in zip(matrix_COP, pos_COP, panel_id_COP, ChMap_COP ) ]
pmt_COP =  [ PMT(ID=int(pan.ID), panel=pan, channelmap=pan.channelmap) for pan in panels_COP ]
tel_COP= Telescope(name = "COP", panels=panels_COP, PMTs = pmt_COP)


##BR: BaronRouge GW Rocher Fendu 2015-2019 3 matrices = 1 * v1.1 + 2 * v2.0 (mapping might be wrong)
ChMap_BR = [ChMap32, ChMap16, ChMap32]
npanels_BR = 3
matrix_BR = [matrixv2_0, matrixv1_1, matrixv2_0]
pos_BR = [ Position(PositionEnum.Front,0), Position(PositionEnum.Middle1, 600), Position(PositionEnum.Middle2, 1200)]
panel_id_BR = [23, 24, 25] 
panels_BR = [ Panel(ID=panID, matrix=m, position=pos, channelmap=map) for (m, pos, panID, map) in zip(matrix_BR, pos_BR, panel_id_BR, ChMap_BR ) ]
pmt_BR = [ PMT(ID=int(pan.ID), panel=pan, channelmap=pan.channelmap) for pan in panels_BR ]
tel_BR= Telescope(name = "BR", panels=panels_BR, PMTs = pmt_BR)


dict_tel = { 'SNJ': tel_SNJ, 'SBR': tel_SBR, 'COP':tel_COP, 'BR': tel_BR }
##GV: GeantVert GW Matylis 2015-2019  (mapping might be wrong)
# matrix_GV = [matrixv1_1 for _ in range(3)]
# pos_GV = [PositionEnum.Front, PositionEnum.Middle1, PositionEnum.Rear]
# panels_GV = list(Panel(matrix=m, ID=9+i, position=pos, channelmap=ChMap16) for i, (m, pos) in enumerate(zip(matrix_GV, pos_GV)) )
# tel_GV = Telescope(name = "GeantVert", panels=panels_GV, spacing=600)
# channelmaps_GV= { p.ID :  p.channelmap.map() for p  in panels_GV}
# barwidths_GV = { p.ID :  int(m.scintillator.width) for p,m  in zip(panels_GV, matrix_GV)} #in mm
# zpos_GV = { p.ID : tel_GV.spacing * i for i, p in enumerate(panels_GV)}

#####OM: OrangeMecanique GW Fente du Nord 2017-2019 3 matrices = 1 * v1.1 + 2 * v2.0
# matrix_OM = [matrixv2_0, matrixv1_1, matrixv2_0]
# pos_OM = [PositionEnum.Front, PositionEnum.Middle1, PositionEnum.Rear]
# panels_OM = list(Panel(matrix=m, ID=9+i, position=pos, channelmap=ChMap16) for i, (m, pos) in enumerate(zip(matrix_OM, pos_OM)) )
# tel_OM= Telescope(name = "OrangeMecanique", panels=panels_GV, spacing=600)
# channelmaps_OM= { p.ID :  p.channelmap.map() for p  in panels_OM}
# barwidths_OM = { p.ID :  int(m.scintillator.width) for p,m  in zip(panels_OM, matrix_OM)} #in mm
# zpos_OM = { p.ID : tel_OM.spacing * i for i, p in enumerate(panels_OM)}
##ND: NoirDesir
# matrix_ND = [matrixv2_0, matrixv1_1, matrixv2_0]
# pos_ND = [PositionEnum.Front, PositionEnum.Middle1, PositionEnum.Rear]
# panels_ND = list(Panel(matrix=m, ID=9+i, position=pos, channelmap=ChMap32) for i, (m, pos) in enumerate(zip(matrix_ND, pos_ND)) )
# tel_ND = Telescope(name = "NoirDesir", panels=panels_ND, spacing=600)
# channelmaps_ND= { p.ID :  p.channelmap.map() for p  in panels_ND}
# barwidths_ND = { p.ID :  int(m.scintillator.width) for p,m  in zip(panels_ND, matrix_ND)} #in mm
# zpos_ND = { p.ID : tel_ND.spacing * i for i, p in enumerate(panels_ND)}


def str2telescope(v):
    if isinstance(v, Telescope):
       return v
    #print(v)
    if v in list(dict_tel.keys()):
        return dict_tel[v]
    elif v in [f"tel_{k}" for k in list(dict_tel.keys()) ]:
        return dict_tel[v[4:]]
    elif v in [ k.lower() for k in list(dict_tel.keys())]:
        return dict_tel[v.upper()]
    elif v in [f"tel_{k.lower()}" for k in list(dict_tel.keys()) ]:
        return dict_tel[v[4:].upper()]
    else:
        raise argparse.ArgumentTypeError('Input telescope does not exist.')


if __name__ == '__main__':
    #data = Data(telescope=tel_SNJ, input="/Users/raphael/simu/telescope_SNJ/analysis/test_cry1evt.dat")#, url="https://cours.ip2i.in2p3.fr/marteau/muography/ZENITH/OM/", save_path="/Users/raphael/Desktop/test/")
    
    c, _ = list(tel_SNJ.configurations.keys()), tel_SNJ.configurations.values()
    
    pos = Position(loc=PositionEnum.Front, z=0)
#    print(pos.)
    print(tel_SNJ.panels[-1].position.loc)
    print(tel_SNJ.panels[-1].position.z)
    
#    print(tel_SNJ.los['3p1'].shape[:-1])
 #   arr = tel_SNJ.get_los(tel_SNJ.panels[0], tel_SNJ.panels[-2])
 #   arr_rs =  arr.reshape(arr.shape[0], -1)
#    np.savetxt("/Users/raphael/Desktop/test_dxdy.txt", arr_rs, fmt="%.0f")
    #os.system("code /Users/raphael/Desktop/test_dxdy.txt")
    # arr = tel_SNJ.get_pixel_xy(tel_SNJ.panels[0], tel_SNJ.panels[-2])
    # print(arr[15,15,:])
    # print(arr.shape)
    # arr_rs =  arr.reshape(arr.shape[0], -1)
    # np.savetxt("/Users/raphael/Desktop/test_pix_xy.txt", arr_rs, fmt="%.0f")
    #os.system("code /Users/raphael/Desktop/test_pix_xy.txt")
    #print(tel_SNJ.panels[1].position[1])
 #   print(tel_BR.los['3p1'])
    #print(channelmaps_SBR)
    # pos = Position.Front
    # pos.value = 60 
    pass