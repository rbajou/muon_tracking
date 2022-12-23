#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path
from configuration import dict_tel, Telescope,  str2telescope
from processing import InputType, Data
import inspect
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(filename))
wd_path = os.path.abspath(os.path.join(script_path, os.pardir))#working directory path
import argparse
import time
import logging
#personal modules
import processing as pr
from tools.tools import str2bool


_start_time = time.time()
t0 = time.strftime("%H:%M:%S", time.localtime())
print("Start: ", t0)#start time
t_start = time.perf_counter()
home_path = os.environ["HOME"]
parser=argparse.ArgumentParser(
description='''For a given muon telescope configuration, this script allows to perform RANSAC tracking and outputs trajectrory-panel crossing XY coordinates''', epilog="""All is well that ends well.""")
parser.add_argument('--telescope', '-tel', default=dict_tel["SNJ"], help='Input telescope name. It provides the associated configuration.', type=str2telescope)
parser.add_argument('--input_data', '-dat', default=[], nargs="*", help='/path/to/datafile/  One can input a data directory, a single datfile, or a list of data files e.g "--input_data <file1.dat> <file2.dat>"', type=str)
parser.add_argument('--input_type', '-it', default=None,  help="'DATA' or 'MC'", type=str)
parser.add_argument('--out_dir', '-o', default='/Users/raphael/simu/telescope_SNJ/analysis/RANSACtest/', help='Path to processing output', type=str) 
parser.add_argument('--label', '-l', default=None, help='Label of the dataset', required=True, type=str)
parser.add_argument('--max_nfiles', '-max', default=1, help='Maximum number of dataset files to process.', type=int)
parser.add_argument('--residual_threshold', '-rt', default=50, help='RANSAC "distance-to-model" parameter: "residual_threshold" in mm.',type=float)
parser.add_argument('--min_samples', '-ms', default=2, help='RANSAC size of the initial sample: "min_samples".',type=int)
parser.add_argument('--max_trials', '-N', default=100, help='RANSAC number of iterations: "max_trials".',type=int)
parser.add_argument('--fit_intersections', '-intersect', default=False, help='if true record line model intersection points on panel; else record closest XY points to model',type=str2bool)
parser.add_argument('--info', '-info', default=None, help='Additional info',type=str)
args=parser.parse_args()
tel = args.telescope
print(f"telescope : {tel}")
inData = args.input_data
if len(args.input_data)==1 : 
    inData = f'{args.input_data[0]}'    
    ####In case we input a .list file containing rawdata file paths
    if Path(inData).is_file() & inData.endswith(".list"): 
        print("Input is .list file")
        listfiles = []
        with open(inData, "r") as f: 
            for line in f.readlines():
                l = line.split("\n")[0]
                listfiles.append(l)
        inData=listfiles

if args.input_type == 'DATA': input_type=InputType.DATA
elif args.input_type == 'MC': input_type=InputType.MC
else: raise argparse.ArgumentTypeError("--input_type should be 'DATA' or 'MC'.")

label = args.label
print(f"Input data : {inData}")
outDir = os.path.join(args.out_dir, '' )    
Path(outDir).mkdir(parents=True, exist_ok=True)
print("PROCESSING...")
_start_time = time.time()
recoDir = outDir+ f"out_{label}/"
print(f"Output dir : {recoDir}")
Path(recoDir).mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=recoDir+f'out.log', level=logging.INFO, filemode='w')
logging.info(sys.argv)
logging.info(t0)
logging.info(args.info)

recoData = Data(telescope=tel, input=inData, type=input_type ,label=label, max_nfiles=args.max_nfiles)
recoData.builddataset()
process_reco = pr.Processing(data=recoData, outdir=recoDir)
nPM = len(tel.PMTs)
rt = args.residual_threshold#mm
ms = args.min_samples
N  = args.max_trials
is_fit_intersect = args.fit_intersections
s= f"is_fit_intersect={is_fit_intersect}"
#print(s)
logging.info(s)

s = f'RANSAC(residual_threshold={rt}mm, min_samples={ms}, max_trials={N})'
print(s)
logging.info(s)
if input_type==InputType.MC:
    max_outliers = None
    logging.info(f"max_outliers={max_outliers}")
    eff=1.
    logging.info(f"scint_eff={eff}")
    process_reco.ransac_reco_pmt(residual_threshold=rt, min_samples=ms, max_trials=N, scint_eff=eff, max_outliers=max_outliers, is_fit_intersect=is_fit_intersect)
elif input_type==InputType.DATA: 
    max_outliers=None #dict={"3p":4, "4p":4}
    logging.info(f"max_outliers={max_outliers}")
    langau=None
    logging.info(f"langau={langau}")
    process_reco.ransac_reco_pmt(residual_threshold=rt, min_samples=ms, max_trials=N, langau=langau, max_outliers=max_outliers, is_fit_intersect=is_fit_intersect)
else: raise ValueError
#######
######
process_reco.to_csv() ##save reco 

t_sec = round(time.time() - _start_time)
(t_min, t_sec) = divmod(t_sec,60)
(t_hour,t_min) = divmod(t_min,60)
t_end = 'runtime processing : {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec)
print(t_end)
logging.info(t_end)
