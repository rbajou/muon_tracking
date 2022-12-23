The repository contains python modules and scripts to reconstruct particles trajectories  (in ```processing.py```) using the RANSAC fit method in muon telescope from input data in tomography or calibration (open-sky) mode. It also features analysis module (```analysis.py```) to use on reconstruction output.

To process raw data, the user needs to run ```python3 main_processing.py``` providing the following arguments : 
- ```--telescope``` (str2telescope) the telescope name (required): Check the available telescope configurations in  dictionary ```dict_tel``` in ```configuration.py``` or edit/add one in the latter library.
- ```--input_data``` (str or List[str] or ```.list``` file) :  /path/to/datafile/  One can input a data directory, a single datfile, or a list of data files e.g --input_data <file1.dat> <file2.dat>.
- ```--out_dir``` (str) : /path/to/outdir/ where the output directory will be created.
- (optional)```--input_type ``` (str, ```DATA``` or ```MC```) either real data or monte-carlo data (simu)
- (optional) ```--out_dir ``` (str) path to output directory
- (optional) ```--max_nfiles```  (integer, default is ```1```) the maximum number of data files to process.
- (optional) ```--is_ransac```  (bool, default is ```True```)
RANSAC parameters:
- (optional) ```--residual_threshold```  (float, default is ```50```mm, i.e size of detector pixel)
- (optional) ```--min_samples```  (float, default is ```2```) the size of the inital data sample 
- (optional) ```--max_trials```  (float, default is ```100```) maximum number of iterations to find best trajectory model
- ...

The output is a directory with csv.gz and log files containing: 
- ```reco.csv.gz``` : RANSAC output (intersection points XY coordinates between fitted trajectories and each telescope panel) for each filtered track.
- ```inlier.csv.gz``` : RANSAC inliers and outliers XYZ points  coordinates and their associated ADC content (in X and Y)
- ```out.log``` : log file featuring info on track reconstruction processing

Once the processing output is here, you can run the ```check_reco.py``` script to get XY maps and DXDY maps with the following arguments:
- ```--telescope``` (str2telescope)
- ```--reco_dir``` (str) : /path/to/<out_*>/
- ```--out_dir``` (str) : /path/to/outdir/


