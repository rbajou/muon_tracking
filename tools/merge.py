#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import inspect
import argparse
import sys
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = os.path.dirname(os.path.abspath(filename))
wd_path = os.path.abspath(os.path.join(script_path, os.pardir))#working directory path
sys.path.append(wd_path)
from tools import merge_files
import shutil

if __name__=='__main__':

    parser=argparse.ArgumentParser( description='''Merge job output files''')
    parser.add_argument('--reco_dir', '-i', default=os.environ['HOME'], help="Directory where job outputs are /path/to/<jobid>", type=str) #/gpfs/users/bajou/stella/muon_data/reco
    parser.add_argument('--out_dir', '-o',default="/path/to/out_<data_label>", help="Output directory for merged job outputs", type=str)
    parser.add_argument('--type', '-t', default=[],nargs="*", help="'reco, 'inlier', 'outlier'", type=str)
    parser.add_argument('--ext', '-ext',  default="csv.gz", help="'csv' or 'txt', else modify merge function", type=str)
    parser.add_argument('--prefix', '-p',  default="out", help="prefix folders name containing files to be merged", type=str)
    args=parser.parse_args()
    print(args.reco_dir, args.out_dir)
    out_dir = os.path.join(args.out_dir,'')
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f'Out merge dir : {out_dir}')
    filename = out_path.name
    print(f'Filename : {out_path.name}')
    
    for t in args.type:
    #     #filename, file_extension = os.path.splitext('/path/to/somefile.ext')
        ext=""
        if t == 'selection': ext="json.gz"
        else: ext = args.ext
        out_file = merge_files(recodir=args.reco_dir, filename=filename, type=t, ext=ext, prefix=args.prefix)
        print(f"Output file : '{out_file}'")
        shutil.move(out_file, args.out_dir)
        print(f"'{out_file}' moved to '{args.out_dir}'")
