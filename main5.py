## Main5 
##### Jan 10, 2021 Developed by Naoya Yamazaki  Copyright @ Naoya Yamazaki
##### GUI

import datetime
import glob
import os
import pickle
import re
import shutil
import sys
import time
import tkinter as tk
import tkinter.ttk as ttk
import traceback
import urllib.request as req
import warnings
import webbrowser
from io import StringIO
from time import sleep
from tkinter import *
from tkinter import filedialog, messagebox
from tkinter.filedialog import askopenfile
from tkinter.font import BOLD

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import RMSprop
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import np_utils, to_categorical
from PIL import Image, ImageTk
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sklearn.model_selection import train_test_split

import cnn5
import scraper5
import testseparator5

warnings.filterwarnings('ignore')

#####################################################
# Tkinter window
app = Tk()

WW=app.winfo_screenwidth()
WH=app.winfo_screenheight()
WW=int(WW*0.9)
WH=int(WH*0.9)
print('WW= {}   WH= {}'.format(WW, WH))
geo_param = str(WW)+"x"+str(WH)+"+0"+"+0"
print('geo param = ', geo_param)
app.geometry(geo_param)
# app.resizable(WW, WH)

app.title('CNN Image Classifier')

# Functions

def hyper_param_get():
    global pxw
    global pxh
    global ch
    global batchsize
    global epoch
    global dropout
    global early_stopping_enable
    
    hyperparam_df = pd.read_excel('{}/hyperparam.xlsx'.format(pj_dir))
    print('hyperparam df = ', hyperparam_df)
    hyperparam_list0 = hyperparam_df.columns
    hyperparam_list = hyperparam_df.iloc[0]
    print(hyperparam_list0)
    print(hyperparam_list)
    pxw = int(hyperparam_list[1])
    pxh = int(hyperparam_list[2])
    ch = int(hyperparam_list[3])
    batchsize = int(hyperparam_list[4])
    epoch = int(hyperparam_list[5])
    dropout = float(hyperparam_list[6])
    early_stopping_enable = int(hyperparam_list[7])

    print('pxw------dropout =', pxw, pxh, ch, batchsize, epoch, dropout)
    
    cnn_box.delete("1.0","end")
    cnn_box_value.delete("1.0","end")
    
    cnn_box.insert(tk.END, '{}'.format(hyperparam_list0[1]))
    cnn_box.insert(tk.END, '\n{}'.format(hyperparam_list0[2]))
    cnn_box.insert(tk.END, '\n{}'.format(hyperparam_list0[3]))
    cnn_box.insert(tk.END, '\n{}'.format(hyperparam_list0[4]))
    cnn_box.insert(tk.END, '\n{}'.format(hyperparam_list0[5]))
    cnn_box.insert(tk.END, '\n{}'.format(hyperparam_list0[6]))
    cnn_box.insert(tk.END, '\n{}'.format(hyperparam_list0[7]))
    
    cnn_box_value.insert(tk.END, '{}\n{}\n{}\n{}\n{}\n{}\n{}'.format(pxw,pxh,ch,batchsize,epoch,dropout,early_stopping_enable))

def pj_create():                     # pj_dir as project main dir
    global pj_dir
    pj_entry.delete(0, END) 
    ret = filedialog.askdirectory()
    pj_dir = ret
    pj_entry.insert(END, pj_dir) 
    print('ret = ', ret, 'pj_dir = ', pj_dir)
    
    if os.path.exists('{}/hyperparam.xlsx'.format(pj_dir)):
        hyper_param_get()
    else:
        shutil.copyfile('hyperparam_init.xlsx', '{}/hyperparam.xlsx'.format(pj_dir))
        hyper_param_get()
                        
def go_scrape():
    pj_dir = pj_entry.get()
    scroll = scroll_combo.get()
    trans_list = words_box.get(1.0, "end-1c")
    trans_list = trans_list.split('\n')
    trans_list.append(pj_dir)
    trans_list.append(scroll)
    print('trans_list ===== ', trans_list)
    print(type(trans_list))
    scraper5.scraper(trans_list)
    
def test_separate():
    print('Go to Test Data Separation.....')
    testseparator5.testseparator(pj_dir)

########GO LEARNING################                        
# def go_learn_ok():
#     cnn_warning_window.destroy()
#     go_learn()

# def go_learn_ng():
#     cnn_warning_window.destroy()
#     pass



def go_learn_check():

    cnn_warning_window = tk.Toplevel(app)
    
    geo_param_cnn_warning = str(280) + "x" +str(130) + "+" + str(int(WW/2-280/2))+ "+" + str(int(WH/2-130/2))
    
    print('geo_param_cnn_warning = ', geo_param_cnn_warning)
    print(type(geo_param_cnn_warning))
    
    cnn_warning_window.geometry(geo_param_cnn_warning)
    
    cnn_warning_window.title('!WARNING')
    
    cnn_warning = tk.Label(cnn_warning_window, text = "Warning! Check parameters...", 
                           font=('meiryo', 12, 'bold'))
    cnn_ok = tk.Button(cnn_warning_window, text = "OK", font=('meiryo', 12, 'bold'), 
                       command=cnn_warning_window.destroy)
    
    cnn_warning.place(x=10, y=10)
    cnn_ok.place(x=115, y=70)

def go_learn():

    print('Go learn')
    
    try:
        hyperparam_list0 = cnn_box.get(1.0, "end-1c")
        hyperparam_list0 = hyperparam_list0.split('\n')
        
        hyperparam_list = cnn_box_value.get(1.0, "end-1c")
        hyperparam_list = hyperparam_list.split('\n')
        
        print('checkpoint 1 ', str(sys.exc_info()[0]))
        
        hyperparam_list[0] = int(hyperparam_list[0])
        hyperparam_list[1] = int(hyperparam_list[1])
        hyperparam_list[2] = int(hyperparam_list[2])
        hyperparam_list[3] = int(hyperparam_list[3])
        hyperparam_list[4] = int(hyperparam_list[4])
        hyperparam_list[5] = float(hyperparam_list[5])
        hyperparam_list[6] = int(hyperparam_list[6])
        
        print('checkpoint 2 ', str(sys.exc_info()[0]))
        print('naonao hyper list0', hyperparam_list0)
        print('naonao hyper list', hyperparam_list)
        
        hyperparam_dict = dict(zip(hyperparam_list0, hyperparam_list))
        print(hyperparam_dict)

        hyperparam_dict_series = {}
        for k,v in hyperparam_dict.items():   # 一度pd.Seriesに変換
            hyperparam_dict_series[k]=pd.Series(v)
        hyperparam_df = pd.DataFrame(hyperparam_dict_series)
        print('nao3', hyperparam_df)
        hyperparam_df.to_excel('{}/hyperparam.xlsx'.format(pj_dir))
        print('Success to go learning ccn5') 
        cnn5.cnn(pj_dir)  # こっちが正しいんじゃない？
    except:
        print('nao4 go to check', str(sys.exc_info()[0]))
        go_learn_check()
        
    # print('Success to go learning ccn5')    
    # cnn5.cnn(pj_dir)

def img_select(): # Classify module
    img_entry.delete(0, END)
    img_init = Image.open('pics/img_init.jpg')
    img = ImageTk.PhotoImage(img_init)
    img_label = tk.Label(image = img)
    img_label.image = img
    img_label.place(x=866, y=165)
    
    img_file = filedialog.askopenfilename()
    img_entry.insert(END, img_file)
        
    img_original = Image.open(img_file)
        
    w = img_original.width
    h = img_original.height
    canw = 400
    canh = canw / 1.618   # Golden ratio
    rw = w/canw 
    rh = h/canh
    if rw >= rh:
        img = img_original.resize((int(w/rw), int(h/rw)))
    else:
        img = img_original.resize((int(w/rh), int(h/rh)))
    
    img = ImageTk.PhotoImage(img)
    img_label = tk.Label(image = img)
    img_label.image = img
    img_label.place(x=866, y=165)
    
    # classifier
    image = img_original.resize((pxw, pxh))  # Hyper parameter later!!!!
    model = load_model('{}/model.h5'.format(pj_dir))
    print('h5 files ', model, type(model))
    
    np_image = np.array(image)
    np_image = np_image / 255
    np_image = np_image[np.newaxis, :, :, :]
    result = model.predict(np_image)
    print('result = ', result)
    result = result[0]
    print('result (after)= ', result)
    print('len(result = )', len(result))
    
    label_class_df = pd.read_csv('{}/label_class.csv'.format(pj_dir))
        
    # 関数化要　Auto classifierと共用とする    
        
    class_list =[] 
    for i in range(len(result)):
        class_list.append(label_class_df.iloc[0,i+1])
    print('class_list = ', class_list)

    result_dict  = {k:v for k, v in zip(class_list, result)}
    print('result_dict = ', result_dict)

    correct = os.path.basename(img_file)
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.4)
    plt.bar(range(len(result_dict)), result_dict.values())
    plt.xticks(range(len(result_dict)), result_dict.keys())
    plt.title("Probabilities in classes", fontsize=16)
    plt.xlabel("Correct is {}".format(correct), fontsize=16)
    # plt.ylabel("Prbability", fontsize=18)

    plt.tick_params(labelsize=16, rotation=90)
    
    # plt.show()
    fig.savefig('{}/fig_temp.png'.format(pj_dir))
    fig = Image.open('{}/fig_temp.png'.format(pj_dir))

    # figw = fig.width
    # figh = fig.height
    fcw = 400
    fch = 320
    fig = fig.resize((int(fcw), int(fch)))
                     
    # rfw = figw/fcw 
    # rfh = figh/fch
    # if rfw >= rfh:
    #     fig = fig.resize((int(figw/rfw), int(figh/rfw)))
        
    # else:
    #     fig = fig.resize((int(figw/rfh), int(figh/rfh)))
    print('figsize', fig.width, fig.height)
    
    fig = ImageTk.PhotoImage(fig)
    fig_label = tk.Label(image = fig)
    fig_label.image = fig
    fig_label.place(x=866, y=420)
    
def auto_classifier():
    print('Auto classifier started!!!')
    files = glob.glob('{}/test_img/*'.format(pj_dir))
    df = pd.read_csv('{}/label_class.csv'.format(pj_dir))
    # result_list = []
    result_df =pd.DataFrame(columns = ['result_max',
                                       'result_max_index',
                                       'class_max',
                                       'file_name'])
    
    for i, file in enumerate(files):
        print('nao 0122 file = ', file)
        image = Image.open(file)
        print('Image = ', image)
        image = image.resize((pxw, pxh))  
        model = load_model('{}/model.h5'.format(pj_dir))
        np_image = np.array(image)
        np_image = np_image / 255
        np_image = np_image[np.newaxis, :, :, :]
        result = model.predict(np_image)
        result = result[0]
        result = result.tolist()
        print('result = ', result, type(result))
    
        result_max = max(result)
        print('result max = ', result_max)
                
        result_max_index = result.index(result_max)   #max(result))
        
        class_max = df.iloc[0, result_max_index+1]  ###
        file_name = os.path.basename(file)
        print('nao 0122', result_max, result_max_index, class_max, file_name)
        # result_list.append(result_max)
        # result_list.append(result_max_index)
        # result_list.append(class_max)
        # result_list.append(file_name)
        result_df.loc[i] = [result_max, result_max_index, class_max, file_name]
    
    print(result_df)
    result_df.to_excel('{}/result_auto.xlsx'.format(pj_dir))
        
        




# Logo
logo = Image.open('pics/logo2.png')

logo = ImageTk.PhotoImage(logo)
logo_label = tk.Label(image = logo)
logo_label.image = logo
logo_label.place(x=0, y=0)


# Version, Owner
version1 = tk.Label(app, text='Version 5 January 2, 2021', font=('meiryo', 12, 'bold'))
version1.place(x=520, y=15)
version2 = tk.Label(app, text='Developed by Naoya Yamazaki', font=('meiryo', 12, 'bold'))
version2.place(x=520, y=38)
version3 = tk.Label(app, text='Copyright@Naoya Yamazaki', font=('meiryo', 12, 'bold'))
version3.place(x=520, y=61)

## Scraper
# Create/select project
pj_label = Label(app, text='Create/select Project Dir>>>', font=('meiryo', 12))
pj_browse = Button(app, text='Browse...', width=12, 
                   font=('meiryo', 12), command=pj_create)
pj_entry = Entry(app, font=('Franklin Gothic', 14, 'bold'))

pj_label.place(x=10, y=126) 
pj_browse.place(x=245, y=120)
pj_entry.place(x=380, y=122, width=200, height = 35) 

# Instruction to input class, query in textbox
scraper_inst0 = Label(app, text='Write classes/Searching words on the Textbox below', font=('meiryo', 12))
scraper_inst1 = Label(app, text='Add "-" on the head of the class', font=('meiryo', 12))
scraper_inst2 = Label(app, text='[Example]', font=('meiryo', 12))
scraper_inst3 = Label(app, text='-cat                               <--class name', font=('meiryo', 12))
scraper_inst4 = Label(app, text='american short hair       <--searching word "cat"', font=('meiryo', 12))
scraper_inst5 = Label(app, text='persian cat                    <--searching word for "cat"', font=('meiryo', 12))
scraper_inst6 = Label(app, text='-dog                              <--class name', font=('meiryo', 12))
scraper_inst7 = Label(app, text='shepherd                       <--searching word "dog"', font=('meiryo', 12))
scraper_inst8 = Label(app, text='Labrador Retriever        <--searching word for "dog"', font=('meiryo', 12))

scraper_inst0.place(x=10, y=190)
scraper_inst1.place(x=10, y=215)
scraper_inst2.place(x=30, y=205+22*2)
scraper_inst3.place(x=40, y=205+22*3)
scraper_inst4.place(x=40, y=205+22*4)
scraper_inst5.place(x=40, y=205+22*5)
scraper_inst6.place(x=40, y=205+22*6)
scraper_inst7.place(x=40, y=205+22*7)
scraper_inst8.place(x=40, y=205+22*8)
                     
words_box = Text(app, font=('Franklin Gothic', 12, 'bold'))
words_box.place(x=10, y=410, width = 300, height=340)

scrape_btn = tk.Button(app, text='GO SCRAPING', width=15, height=6, 
                       font=('meiryo', 12), command=go_scrape)
scrape_btn.place(x=315, y=502)

# selectbox for scroll
scroll = Label(app, text='Page scrolling [Pxls]', font=('meiryo', 12))
scroll.place(x=315, y=440)

scroll_combo = ttk.Combobox(app, state='readonly', width=12,
                            font=('Franklin Gothic', 14, 'bold'))
scroll_combo["values"] = (
    1000, 2000, 3000, 4000, 5000, 
    6000, 7000, 8000, 9000, 10000, 
    11000, 12000, 13000, 14000, 15000)
scroll_combo.place(x=315, y=470)

## Test Separator
test_separate_btn = tk.Button(app, text='Test Data\nSeparation', width=15, 
                              font=('meiryo', 12), command=test_separate)
test_separate_btn.place(x=315, y=670)

## Learning / CNN
cnn_inst0 = Label(app, text='Setup CNN parameters', font=('meiryo', 12))
cnn_inst0.place(x=495, y=190)

cnn_box = Text(app, font=('Franklin Gothic', 12, 'bold'))
cnn_box.place(x=495, y=227, width = 280, height=450)

cnn_box_value = Text(app, font=('Franklin Gothic', 12, 'bold'))
cnn_box_value.place(x=780, y=227, width = 74, height=450)

learn_btn = tk.Button(app, text='GO LEARNING', width=35,  
                      font=('meiryo', 12), command=go_learn)
learn_btn.place(x=495, y=700)

##Classifier
img_select_label = Label(app, text='Select image to be classified>>>', 
                         font=('meiryo', 12))
img_browse = Button(app, text='Browse...', width=12, 
                    font=('meiryo', 12), command=img_select)
img_entry = Entry(app, font=('Franklin Gothic', 14, 'bold'))

img_select_label.place(x=600, y=126) 
img_browse.place(x=866, y=120)
img_entry.place(x=1000, y=122, width=200, height = 35) 



# Auto classifier
auto_classifier = Button(app, text='AUTO', width=6, 
                    font=('meiryo', 12), command=auto_classifier)
auto_classifier.place(x=1220, y=120)




## Links

readme = Image.open('pics/readme.png')
github = Image.open('pics/github.png')
fb = Image.open('pics/fb.png')

readme = ImageTk.PhotoImage(readme)
readme_label = tk.Label(image = readme)
readme_label.image = readme
readme_label.place(x=866, y=20)
readme_label.bind("<Button-1>", lambda e: webbrowser.open_new("readme5.pptx"))

github = ImageTk.PhotoImage(github)
github_label = tk.Label(image = github)
github_label.image = github
github_label.place(x=1010, y=20)
github_label.bind("<Button-1>", lambda e: webbrowser.open_new("https://github.com/AISTARWORKS"))
                  

fb = ImageTk.PhotoImage(fb)
fb_label = tk.Label(image = fb)
fb_label.image = fb
fb_label.place(x=1100, y=20)
fb_label.bind("<Button-1>", lambda e: webbrowser.open_new("https://www.facebook.com/naoya.yamazaki.988"))

 
app.mainloop()

# cd C:\Users\nnroc\MyProgram\IMAGE_DETECTOR_CNN\VERSION_5
