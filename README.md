The repository contains python modules and scripts to reconstruct particles trajectories  (in ```processing.py```) using the RANSAC fit method in muon telescope from input data in tomography or calibration (open-sky) mode. It also features analysis module (```analysis.py```) to use on reconstruction output.

To process raw data, the user needs to run ```python3 main_processing.py``` providing following arguments : 
- ```--telescope``` (str2telescope) the telescope name (required): so far available ```SNJ``` (4 panels) , ```SBR``` (4 panels), ```ND``` (3 panels), ```OM``` (3 panels). Possibility to add a telescope configuration in ```configuration.py```
- ```--input_data``` (str or List[str]) :  /path/to/datafile/  One can input a data directory, a single datfile, or a list of data files e.g --input_data <file1.dat> <file2.dat>.
- ```--out_dir /path/to/outdir/``` where the output directory will be created.
- (optional)```--input_type ``` (str, ```DATA``` or ```MC```) either real data or monte-carlo data (simu)
- (optional) ```--out_dir ``` (str) path to output directory
- (optional) ```--max_nfiles```  (integer, default is ```1```) the maximum number of data files to process.
- (optional) ```--is_ransac```  (bool, default is ```True```)
RANSAC parameters:
- (optional) ```--residual_threshold```  (float, default is ```50.```mm)
- (optional) ```--min_samples```  (float, default is ```2```) the size of the inital data sample 
- (optional) ```--max_trials```  (float, default is ```100```) maximum number of iterations to find best trajectory model
- ...

The output is a directory with csv and json files containing: 
- ```*_reco.csv``` : RANSAC output (intersection points XY coordinates between fitted trajectories and each telescope panel) for each filtered track.
- ```*_inlier.csv``` and ```*_outlier.csv```: RANSAC inliers and outliers XYZ points  coordinates and their associated ADC content (in X and Y)

