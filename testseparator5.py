## Test Data Separator 5
##### Jan 13, 2021 Developed by Naoya Yamazaki  Copyright @ Naoya Yamazaki
    # Separating TEST DATA for classifier comfirmation later
    ### Separating test files###############

import datetime
import glob
import math
import os
import random
import shutil
import sys
import time
import traceback
from time import sleep

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def testseparator(pj_dir):
    
    label_list = []
    label_key = []
    label_dict = {}
    sampling = 0.1
    key=0
    os.makedirs('{}/test_img'.format(pj_dir),exist_ok=True)
    
    for root, dirs, files in os.walk("{}/img".format(pj_dir)):
        for class_name in dirs:
            files = glob.glob('{}/img/{}/*.jpg'.format(pj_dir, class_name))
            test_files = random.sample(files,math.ceil(len(files)*sampling))
            
            
            for i, file in enumerate(test_files):
                shutil.copy2(file, '{}/test_img/{}{}.jpg'.format(pj_dir, i, class_name))
                os.remove(file)
                
    #######################################################################

    

    
    
    
