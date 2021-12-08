Submission Script Readme

Enclosed is a python script sharon_submission.py.  This script assumes that it will be used as follows

USAGE = f'Usage: python sharon_submission.py  <task_id> <cancer type> <input_file/dir_paths> <output_file_path>'

Where: output_file_path is the path where the submission result will be written, in a file called "sharon_submission_task1_<cancer type>.csv
The format of the file is as required in the guidance.

The third parameter is assumed to be a folder in tasks 1 and 3 with all omics of relevant cancer type. 
In case of task 2, it is assumed to be a FILE containing EXP info for the tested cancer type. 
LAML cannot be a cancer type for task 3


This script assumes that all its data files are present in the same folder it is run from. 
If for whatever reason this is not practical, then the variable "model_path" in the last few lines should be modified accordingly.

Also, pickled files use the package "dill" in version 0.3.3. (pip install dill==0.3.3). Other libraries are default.
Python version needed is 3.8.

The submission results are tab delimited as required and best viewed by notepad++.

All predictions measure their execution time including file loading and print out times.