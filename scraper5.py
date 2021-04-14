## Image Scraper 5
##### Jan 6, 2021 Developed by Naoya Yamazaki  Copyright @ Naoya Yamazaki
##### Scraping from Microsoft Bing

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

def scraper(trans_list):
    
    print('trans_list = ', trans_list)
    scroll = trans_list[-1]
    scroll = int(scroll)
    pj_dir = trans_list[-2]
    print('trans_list = ', trans_list)
    print(scroll, pj_dir)
    
    class_list = []
    indx_list = []
    for i, word in enumerate(trans_list):
        if trans_list[i][0] == '-':
            class_list.append(trans_list[i])
            indx_list.append(i)
    print('class_list = ', class_list, 'indx_list = ', indx_list)
             
    for i, class_name in enumerate(class_list):
        if i < len(class_list)-1:
            query_list = trans_list[indx_list[i]+1 : indx_list[i+1]]
        else:
            query_list = trans_list[indx_list[i]+1 : -2]
        print('Go to scraper for', class_name, query_list)
        
        start_time = time.time()
        dt_start = datetime.datetime.now()
        print('Processing started from {}'.format(dt_start))
        print('Success to import library')
        print('Start donwloading process for {}'.format(class_list))

        # 1st for loop; 
        # for each query, make chrome path, search query word, get url, save log in excel, 
        # download and save all images

        global d_list
        d_list = []
        global num_of_url
        num_of_url = 0
        
        for query in query_list:
            chrome_path = r'C:\Users\nnroc\MyProgram\IMAGE_DETECTOR_CNN\VERSION_5\chromedriver.exe'
            options = Options()
            options.add_argument('--incognito')
            driver = webdriver.Chrome(executable_path=chrome_path, options=options) # Options ???
            url = 'https://www.bing.com/images'
            driver.get(url)
            sleep(3)   

            print('Chrome driver has been activated')
            
            search_box = driver.find_element_by_class_name('sb_form_q')   #Bingのsearch boxにqueryを入力
            search_box.send_keys(query)
            search_box.submit() 
            sleep(2)
            
            print('Start scraping image URLs for class {} by word {}'.format(class_name, query))

            height = 1000
            print('Scrolling', end="")
            while height < scroll:
                driver.execute_script("window.scrollTo(0, {});".format(height))
                height += 300
                print('.', end="")
                sleep(2)
            print('|')
            print('Scrolling finished for class {} by word {}'.format(class_name, query))
            print('Wait! Getting element from web site')
            elements = driver.find_elements_by_class_name('imgpt')

            print('Making image URL list for class {} by word {}'.format(class_name, query), end="")

            for i, element in enumerate(elements):
                name = f'{query}_{i}'
                img_url = element.find_element_by_tag_name('img').get_attribute('src')
                if img_url[:4] == 'http':
                    d = {'filename': name, 'img_url': img_url}  #dict type

                    d_list.append(d)

                    num_of_url += 1
                    print('.', end="")
                sleep(3)
            print('|')     
            driver.quit()
        # End of for loop
                
        df = pd.DataFrame(d_list)

        print('|')
        print('Number of Image URL = {}'.format(num_of_url))
        print('Completed making list as pandas data frame for class {}'.format(class_name))

        url_dir = '{}/'.format(pj_dir)
        if os.path.isdir(url_dir) == False:
            os.makedirs(url_dir) 

        print('Start saving URL list into CSV for class {}'.format(class_name))

        df.to_csv('{}/image_url_{}.csv'.format(pj_dir, class_name))  

        print('Completed CSV for class {}'.format(class_name))
        print('Start saving image files for class {}'.format(class_name))

        img_dir = '{}/img/{}/'.format(pj_dir, class_name)

        if os.path.isdir(img_dir) == False:
            os.makedirs(img_dir)

        print('Saving image files for class {}'.format(class_name), end="")

        num_of_img_files = 0

        for filename, img_url in zip(df.filename, df.img_url):
            try:
                img=requests.get(img_url)
                with open(img_dir + filename + '.jpg', 'wb') as f:
                    f.write(img.content)
                num_of_img_files += 1
            except Exception as e:
                e_summary = e[:50]
                print('Ignorable Error--already skipped {}----'.format(e_summary))
                pass

        print('.', end="")

        sleep(1)

        print('|')
        print('Number of image files for {} = {}'.format(class_name, num_of_img_files))
        print('Completed saving image files for class {}'.format(class_name))
        
        total_time = time.time()-start_time
        print('TOTAL TIME for {} = {} sec'.format(class_name, total_time))

    print('CONGRATURATIONS!!!')
    


    

    
    
    
