#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import inspect
import scipy.io as sio
import argparse
import time
import glob
import pandas as pd
import json
from datetime import datetime, timezone
#personal modules
main_dir = os.environ["HOME"]
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(filename))
simu_path = os.path.abspath(os.path.join(script_path, os.pardir))
from configuration import str2telescope
from processing import InputType
from tools.tools import str2bool
from analysis import EvtRate, EventType, RecoData, Cut, AnaBase, AnaHitMap, AnaCharge, PlotHitMap
from acceptance import Acceptance


def_args={}
run_type = "calib" #"calib" or "tomo"
'''
with open(os.path.join(script_path, 'config_files', 'configSNJ.yaml') ) as fyaml:
    try:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        def_args = yaml.load(fyaml, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)
'''   

start_time = time.time()
print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time
t_start = time.perf_counter()
home_path = os.environ["HOME"]

parser=argparse.ArgumentParser(
description='''For a given muon telescope configuration, this script allows to format .root simu output file, process the format data and analyze it.''', epilog="""All is well that ends well.""")
parser.add_argument('--telescope', '-tel', default=def_args["telescope"], help='Input telescope name (e.g "SNJ"). It provides the associated configuration.',  type=str2telescope)
parser.add_argument('--reco_dir', '-i', default=def_args[f"reco_{run_type}"], help='/path/to/out_<label>/', type=str)
parser.add_argument('--out_dir', '-o', default=def_args[f"out_dir_{run_type}"], help='Path to processing output', type=str)
parser.add_argument('--label', '-l', default=def_args[f"label_{run_type}"], help='Label of the dataset',type=str)
parser.add_argument('--do_fit', '-fit', default=True, help='Path to processing output', type=str2bool) 
parser.add_argument('--input_type', '-it', default='DATA',  help="'DATA' or 'MC'", type=str)
args=parser.parse_args()

tel = args.telescope
sconfig = list(tel.configurations.keys())
nc = len(sconfig)
label = args.label
outDir = Path(args.out_dir )
outDir.mkdir(parents=True, exist_ok=True)
intype = args.input_type

try:
    freco = glob.glob(os.path.join(args.reco_dir,  '', f'*_reco.csv*'))[0]
    if label is None: label = os.path.basename(os.path.dirname(freco))
    finlier = glob.glob(os.path.join(args.reco_dir,  '', f'*_inlier.csv*') )[0] 
    foutlier = glob.glob(os.path.join(args.reco_dir,  '', f'*_outlier.csv*') )[0] 

except AttributeError:
    exit("No inlier/outlier files found.")

print("ANALYSIS")
start_time = time.time()
input_type = InputType.REAL
evttype = EventType.MAIN
finfo  = Path(args.out_dir) / f"info_{label}.json"
run_info = {}
recoFile = RecoData(file=freco, 
                                        telescope=tel,  
                                        input_type=input_type)
print(f"load recoFile --- {(time.time() - start_time):.3f}  s ---")   

# tlim = ( int(datetime(2019, 3, 1, hour=00,minute=00,second=00).replace(tzinfo=timezone.utc).timestamp()) , 
#         int(datetime(2019, 5, 1, hour=23,minute=59,second=59).replace(tzinfo=timezone.utc).timestamp())   )

ab = AnaBase(recofile=recoFile, 
                    evttype=evttype,
                    label=label, tlim=None, cuts=None)


#####Hitmap
hmDir = outDir/ "hitmap"
hmFiles = glob.glob( str(hmDir/"hitmap*.txt") )
hmDXDY  = {}
#if len(hmFiles)==0:
print("Hit maps (DXDY)")
hmTomo = AnaHitMap(anabase=ab, input_type=InputType.REAL, panels=tel.panels)
hmDXDY = hmTomo.hDXDY  
hmDir.mkdir(parents=True,exist_ok=True)
# with open(str(hmDir /f'hitmap_{label}_{evttype.name}.pkl'), 'wb') as out:
#         pickle.dump(hmTomo, out, pickle.HIGHEST_PROTOCOL)
for c, hm in hmDXDY.items():  np.savetxt(str(hmDir / f'hitmap_{label}_{evttype.name}_{c}.txt'), hm, delimiter='\t', fmt='%.5e')
pl = PlotHitMap(hitmaps=[hmTomo], evttype=evttype, outdir=str(hmDir)) #m.mask
pl.XY_map()
pl.DXDY_map()
print("save:\n")
os.system(f"ls -lh {hmDir}/*")


sigma = 50#mm
res = ab.df_reco['quadsumres']/sigma**2
ndf = ab.df_reco['npts']-2
gof = res/ndf
ab.df_reco['rchi2'] = gof



plt.close()
#####GoF
range_gof= [0,10]

chi2_noutl = { n : ab.df_reco[ab.df_reco["noutl"] == n]["rchi2"] for n in np.arange(0,6)}
chi2_noutl[f">={6}"] = ab.df_reco[ab.df_reco["noutl"] >= 6]["rchi2"]
fig = plt.figure(constrained_layout=True)
ax = fig.subplot_mosaic([["outliers", "inliers"]], sharey=True)
nbins=50
entries, x = np.histogram(ab.df_reco["rchi2"], bins=nbins, range=range_gof)
xc = (x[1:] + x[:-1]) /2
w = x[1:] - x[:-1]
print(len(w))
norm = np.sum(entries)
#GoF(ax=ax["outliers"], df=ab.df_reco, is_gold=True, column='rchi2', bins=nbins, range=[0,10])
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
chi2_ninl = { n : ab.df_reco[ab.df_reco["ninl"] == n]["rchi2"] for n in np.arange(3,6)}
chi2_ninl[f">={6}"] = ab.df_reco[ab.df_reco["ninl"] >= 6]["rchi2"]
bottom = np.zeros(len(entries))
for i, ((k,chi2), color) in enumerate(zip(chi2_ninl.items(), [ "green", "yellow", "orange", "red"] )):
    e, x = np.histogram(chi2, bins=nbins, range=range_gof )
    e_norm =e/norm
    ax["inliers"].bar(xc, e/norm, w ,color=color, alpha=0.5, label=k, bottom=bottom) #bottom=e[-1]
    bottom += e_norm

ax["inliers"].legend(fontsize=14, title="#inliers")
ax["inliers"].set_xlabel("GoF")
plt.savefig(str(outDir /f"gof.png"))
os.system(f"code {str(outDir/'gof.png')}")



#print(f"loss={cut_chi2.loss*100:.2f}")
#print(len(ab.df_reco.index))
#print(len(ab.df_reco.index)/cut_chi2.loss)


evtIDs = list(ab.evtIDs)
evtIDs_gold = list(ab.df_reco_gold.index)
print("n_evts=",len(evtIDs))
print("n_evts_gold=",len(evtIDs_gold))
print(f"load AnaBase --- {(time.time() - start_time):.3f}  s ---")   


if intype=="DATA": 
    evtrate = EvtRate(df=ab.df_reco)
    df_reco_tof =cut_tof(ab.df_reco)
    evtrate_tof = EvtRate(df=df_reco_tof)
    fig, ax = plt.subplots(figsize=(16,9))
    evtrate(ax, label="all")
    evtrate_tof(ax, label="tof")

    cut_noutl = Cut(column="noutl", vmin=0, vmax=6)
    evtrate_noutl = EvtRate(df=cut_noutl(ab.df_reco)) 
    evtrate_noutl(ax, label="#outl")

    cut_noutl = Cut(column="npts", vmin=0, vmax=10)
    evtrate_noutl = EvtRate(df=cut_noutl(ab.df_reco)) 
    evtrate_noutl(ax, label="#npts")

    ax.legend(loc='best')
    plt.savefig(
        str(outDir/ f"event_rate_{args.label}.png")
    )
    plt.close()
    print(f"save {str(outDir/ f'event_rate_{args.label}.png')}")



    run_info['Nevts'] = evtrate.nevts 
    run_info['run_duration'] = f"{evtrate.run_duration:.0f}" #s
    run_info['mean_evtrate'] = f"{evtrate.mean/3600:.5f}" #evt.s^-1
    outstr=json.dumps(run_info)
    with open(finfo, 'w') as f: f.write(outstr)


# os.system(f"display {args.out_dir}/event*png &")
# exit()


#exit()

    #pl.DXDY_map_az_ze(az=az, ze=ze, invert_yaxis=False,transpose=False, fliplr=True)
# else: 
#     print("Get hit maps (DXDY)")
#     ####Save and load class object with pickle (very slowwwwww....)
#     #with open(hmDir /f'hitmap_{label}_{evttype.name}.pkl', 'rb') as hm:
#         #hmTomo = pickle.load(hm)
#     for c in sconfig:
#         with open(hmDir /f'hitmap_{label}_{evttype.name}_{c}.txt', 'rb') as f:
#             hm = np.loadtxt(str(hmDir / f'hitmap_{label}_{evttype.name}_{c}.txt'), f, delimiter='\t', fmt='%.5e')
#             hmDXDY[c] = hm

#os.system(f"display {args.out_dir}/NHits*{args.label}*png &")
#exit()
# ######CHARGE SIGNALS
inlierData=RecoData(file=finlier, telescope=tel, input_type=input_type, index=evtIDs)

inlierData.df['ADC_SUM'] = inlierData.df['ADC_X'] + inlierData.df['ADC_Y']
outlierData= RecoData(file=foutlier, telescope=tel, input_type=input_type, index=evtIDs)
print(f"load in/outlier files --- {(time.time() - start_time):.3f}  s ---")   



# ####CHARGE CALIBRATION WITH GOLDEN EVENTS = MINIMUM IONIZING PARTICLES(MIP) = MUONS
path_calib_mip = Path(os.environ['HOME'])/f"muon_data/{tel.name}/ana/{def_args['label_calib']}/charge/gold" #Path(args.out_dir)/"calib_gold"
path_calib_mip.mkdir(parents=True, exist_ok=True)
fn_calib = str(path_calib_mip/f"fit_dQ_{args.label}_Inlier_GOLD.csv")
print(fn_calib)

q_dir = Path(args.out_dir) / "charge"
(q_dir/"outliers").mkdir(parents=True, exist_ok=True)
(q_dir/"gold").mkdir(parents=True, exist_ok=True)
if len(evtIDs_gold) != 0: 
    anaChargeGold= AnaCharge(inlier_file=inlierData,
                                        outlier_file=outlierData,
                                        evttype=EventType.GOLD,
                                        outdir=str(q_dir/"gold"), 
                                        label=args.label
                                    ) 
    dict_ADC_gold = {('Inlier', 'green', args.do_fit):anaChargeGold.ADC_XY_inlier, ('Outlier', 'red', False):anaChargeGold.ADC_XY_outlier}


anaChargeMain= AnaCharge(inlier_file=inlierData,
                                    outlier_file=outlierData,
                                    evttype=evttype,
                                    outdir=str(q_dir/"outliers"), 
                                    label=args.label
                                    ) 

dict_ADC = {('Inlier', 'green',False):anaChargeMain.ADC_XY_inlier, ('Outlier', 'red', False):anaChargeMain.ADC_XY_outlier}

if os.path.exists(fn_calib):
    dfCalib = pd.read_csv(glob.glob(fn_calib)[0], sep="\t", index_col=0)
    #dfCalib= dfCalib.set_index('Unnamed: 0')
    print(dfCalib.head())
    fCalib = { pan.ID: dfCalib.loc[f"('{pan.ID}', 'MPV')"]['value'] for pan in tel.panels}
    unc_fCalib = { pan.ID: dfCalib.loc[f"('{pan.ID}', 'MPV')"]['error'] for pan in tel.panels}
    #print(fCalib)
    if len(evtIDs_gold) != 0: anaChargeGold.plot_adc_panels(ADC=dict_ADC_gold, fcal=fCalib, unc_fcal=unc_fCalib )
    anaChargeMain.plot_adc_panels(ADC=dict_ADC, fcal=fCalib, unc_fcal=unc_fCalib)
else: 
    if len(evtIDs_gold) != 0: anaChargeGold.plot_adc_panels(dict_ADC_gold)
    anaChargeMain.plot_adc_panels(ADC=dict_ADC)

#os.system(f"display {q_dir}/main/*png &")


print(f"plot ADC --- {(time.time() - start_time):.3f}  s ---")   
#inlier_evtID_panel = [ set(inlierData.df.loc[ inlierData.df['Z'] == pan.position[1] ].index) for pan in tel.panels ]
#evtID_allpan = set.intersection(*inlier_evtID_panel)
#df_inlier_max = inlierData.df.groupby(['evtID','Z'])['ADC_SUM'].max().loc[evtID_allpan]
# dqx = df_inlier_max.xs(tel.panels[0].position[1], level=1, drop_level=False).values  / fCalib[tel.panels[0].ID]
# dQx = {('Inlier Front', 'green',False): dqx}
# dqy = df_inlier_max.xs(tel.panels[1].position[1], level=1, drop_level=False).values / fCalib[tel.panels[1].ID]
# dQy = {('Inlier Rear', 'green',False): dqy}
# anaChargeMain.scatter_plot_dQ(dQx=dQx, dQy=dQy)

print("\noutfiles:")
os.system(f"ls -lh {args.out_dir}/*")
#os.system(f"display {args.out_dir}/charge/charge_distributions_{args.label}_MAIN.png &")
#os.system(f"for f in {args.out_dir}/*png; do display $f &; done;")
#os.system(f"for f in {args.out_dir}/*/*png; do display $f &; done;")
exit()

#####ACCEPTANCE
if run_type == "calib":
    param_dir = os.path.join(script_path,'', 'AcquisitionParams')
    Corsika_OpenSkyFlux = sio.loadmat(os.path.join(param_dir, f'{tel.name}', 'ExpectedOpenSkyFlux.mat'))
    OSFlux_calib_3p = Corsika_OpenSkyFlux['ExpectedFlux_calib_3p']#open-sky flux
    OSFlux_calib_4p = Corsika_OpenSkyFlux['ExpectedFlux_calib_4p']
    os_flux = { sconfig[0]: OSFlux_calib_3p,  sconfig[1]:OSFlux_calib_3p,  sconfig[2]:OSFlux_calib_4p }
    accDir = os.path.join(outDir, 'acceptance')
    accFiles = glob.glob(os.path.join(accDir, f"acceptance_*_{label}.txt"))
    uaccFiles = glob.glob(os.path.join(accDir, f"error_*_{label}.txt"))
    acc_th_120 = np.loadtxt(os.path.join(script_path, "", "Acceptance/A_theo_16x16_120cm.txt"))
    acc_th_180 = np.loadtxt(os.path.join(script_path, "", "Acceptance/A_theo_16x16_180cm.txt"))
    acc_th = {sconfig[0]: acc_th_120, sconfig[1]:acc_th_120, sconfig[2]:acc_th_180}
    AcqVars = Path(param_dir)/ f'{tel.name}' / 'angular_coord.pkl'

    if AcqVars.exists():
        with open(AcqVars, 'rb') as fin:
            angles = pickle.load(fin)
    else: 
        angles = AngCoord(telescope=tel, tomo=True)
        with open( str( AcqVars ), 'wb') as fout:
            pickle.dump(angles, fout, pickle.HIGHEST_PROTOCOL)
                
    az_calib  = angles.az_os_1d
    AZ_CALIB = angles.AZ_OS_2D
    ze_calib   = angles.ze_os_1d
    ZE_CALIB = angles.ZE_OS_2D
    
    if len(accFiles)==0:
        anaBaseCal = AnaBase(recofile=recoFile,
                evttype=evttype,
                label=label)
        hmCalib = AnaHitMap(anabase=anaBaseCal, 
                            input_type=InputType.REAL, 
                            panels=tel.panels
                            )                
        
        A = Acceptance(hitmap=hmCalib,
                                    outdir=outDir, 
                                    evttype=evttype, 
                                    opensky_flux=os_flux,
                                    theoric=acc_th)
        acceptance = {conf: v for conf,v in A.acceptance.items()}
        err_acc = {conf: v for conf,v in A.error.items()}
        # A.plot_acceptance(az=AZ_CALIB, ze=ZE_CALIB)
        # A.plot_ratio_acc(az =az_calib, ze=ze_calib)
        for (conf, aexp), (_, ath), (_,AZ), (_,ZE) in zip(acceptance.items(), acc_th.items(), AZ_CALIB.items(), ZE_CALIB.items()): 
            A.plot_acceptance_3D(acc_exp=aexp, acc_th=ath, AZ=AZ, ZE=ZE, label=conf)
            A.plot_ratio_acc(acc_exp=aexp, acc_th=ath, az=angles.az_os_1d[conf], ze=angles.ze_os_1d[conf], label=conf)
    else : 
        acceptance = {conf: np.loadtxt(f) for conf, f in zip(sconfig, accFiles)}
        err_acc = {conf: np.loadtxt(f) for conf, f in zip(sconfig, uaccFiles)}










# N_outliers_panel = [len(outlier) for outlier in anaChargeMain.ADC_XY_outlier]
# N_inliers_panel = [len(inlier) for inlier in anaChargeMain.ADC_XY_inlier]
# sum_N_outliers_tot = np.sum(N_outliers_panel)
# sum_N_inliers_tot = np.sum(N_inliers_panel)
    
# for i, p in enumerate(tel.panels):
#     #print(p.position.name)
#     N_outliers = N_outliers_panel[i]
#     N_inliers = N_inliers_panel[i]
#     print(f'ratio_out_in={N_outliers}/{N_inliers}={N_outliers/N_inliers:.2f}' )

exit()


    ######EFFICIENCY
# recoData = RecoData(file=freco, input_type=input_type, telescope=args.telescope)
# eff=Efficiency(rawdata=None, seldata = fselection, inlierdata=inlierData, label=label, outdir=outDir, nbins=50, outlierdata=outlierData, ene_max=1200) #ADC

# l_evtID = eff.get_only_sel_adc(keys=['4p'])
# print(l_evtID)
# print(inlierData.df.index)
# print(inlierData.df[l_evtID])

# sel = eff.sel_adc

# dict_sel_ADC = {(id_panel, 'blue', False): adc for id_panel, adc in eff.sel_adc.items() }

# hSelADC, bin_ADC = eff.plot_adc(name='selection',ADC=dict_sel_ADC) 

# eff.get_adc(df=inlierData.df, outdict=eff.inlier_adc, evtID=l_evtID)
# eff.get_adc(df=outlierData.df, outdict=eff.outlier_adc, evtID=l_evtID)
# reco = eff.inlier_adc#eff.inlier_adc_ch
# outl = eff.outlier_adc
# t_sec = round(time.time() - start_time)
# (t_min, t_sec) = divmod(t_sec,60)
# (t_hour,t_min) = divmod(t_min,60)
# dict_inlier_adc = {(id_panel, 'green', False): adc for id_panel, adc in reco.items() }
# dict_outl_ADC = {(id_panel, 'red', False): adc for id_panel, adc in outl.items() }
# hInlADC, _ = eff.plot_adc(name='inlier',ADC=dict_inlier_adc)
# hOutlADC, _ = eff.plot_adc(name='oulier',ADC=dict_outl_ADC)
# dict_sum_ADC = {(id_panel, 'orange', False): np.concatenate((adc_in,adc_ou), axis=None) for (id_panel, adc_in), (_, adc_ou) in zip(reco.items(),outl.items()) }
# hSumADC, _ = eff.plot_adc(name='inlier+oulier',ADC=dict_sum_ADC)
# dict_ADC = { ('SELECTION', 'blue', False) : dict_sel_ADC, ('Inlier', 'green', False) : dict_inlier_adc, ('Outlier', 'red', False) : dict_outl_ADC}  
# eff.plot_adc_diff(ADC=dict_ADC, name=args.label)
# eff_in, var_in = eff.compute_efficiency_per_panel(raw=hSelADC, reco=hInlADC, label='inlier')
# eff_out, var_out = eff.compute_efficiency_per_panel(raw=hSelADC, reco=hOutlADC, label='outlier')
# eff_sum, var_sum = eff.compute_efficiency_per_panel(raw=hSelADC, reco=hSumADC, label='cumul inlier and outlier')
# print('compute efficiency: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))
# ###key = (label:str, color:str, do_fit:bool), val:np.ndarray
# dict_efficiency = {  ('Outlier', 'red', False) : eff_out, ('Inlier', 'green', False) : eff_in, ('Inlier+Outlier', 'orange', False) : eff_sum }
# dict_variance = {  ('Outlier', 'red', False) : var_out, ('Inlier', 'green', False) : var_in, ('Inlier+Outlier', 'orange', False) : var_sum}#,
# eff.plot_efficiency_vs_energy(efficiency=dict_efficiency, variance=dict_variance, energy=bin_ADC, name='ransac_vs_selection')


exit()





#exit()


