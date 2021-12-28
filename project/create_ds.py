import os
import shutil

import matplotlib
from matplotlib import pylab as plt

import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D


root_path = os.path.join(os.path.dirname(os.getcwd()), "research_dataset", "raw")
print(root_path)

from shutil import copyfile

#transfer scans into single directory
for patient in os.listdir(root_path):
    #print(patient)
    for scan_set in os.listdir(os.path.join(root_path, patient)):
        for scan in os.listdir(os.path.join(root_path, patient, scan_set)):
            print(scan)
            if "T2w" in scan:
                scan_type = "T2w"
            else:
                scan_type = "T1w"
            shutil.copy(os.path.join(root_path, patient, scan_set, scan), os.path.join(os.path.dirname(os.getcwd()), "research_dataset", "ds", scan_type, scan))


