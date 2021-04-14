## ImageDetector with CNN Version 5
##### Jan 9, 2021 Developed by Naoya Yamazaki  Copyright @ Naoya Yamazaki

import os
import pickle
import re
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import train_test_split


def cnn(pj_dir):

    hyperparam_df = pd.read_excel('{}/hyperparam.xlsx'.format(pj_dir))
    print(hyperparam_df)
    hyperparam_list = hyperparam_df.iloc[0]
    print(hyperparam_list)
    pxw = int(hyperparam_list[1])
    pxh = int(hyperparam_list[2])
    ch = int(hyperparam_list[3])
    batchsize = int(hyperparam_list[4])
    epoch = int(hyperparam_list[5])
    dropout = float(hyperparam_list[6])
    early_stopping_enable = int(hyperparam_list[7])

    print(pxw, pxh, ch, batchsize, epoch, dropout)
    

    ### Plot function
    def plot_history(history, save_graph_img_path, fig_size_width, fig_size_height, lim_font_size):

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.figure(figsize=(fig_size_width, fig_size_height))
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = lim_font_size  # 全体のフォント
        plt.subplot(121)

        # plot accuracy values
        plt.plot(epochs, acc, color = "blue", linestyle = "solid", label = 'train acc')
        plt.plot(epochs, val_acc, color = "green", linestyle = "solid", label= 'valid acc')
        plt.title('Training and Validation acc')
        plt.grid()
        plt.legend()

        # plot loss values
        plt.subplot(122)
        plt.plot(epochs, loss, color = "red", linestyle = "solid" ,label = 'train loss')
        plt.plot(epochs, val_loss, color = "orange", linestyle = "solid" , label= 'valid loss')
        plt.title('Training and Validation loss')
        plt.legend()
        plt.grid()

        plt.savefig(save_graph_img_path)
        plt.close() # バッファ解放


    #### Generating train data

    label_list = []
    label_key = []
    label_dict = {}
    key=0

    for root, dirs, files in os.walk("{}/img".format(pj_dir)):
        for class_name in dirs:
            label_list.append(class_name)
            label_key.append(key)
            label_dict[key] = class_name
            key += 1
            
    num_classes = len(label_list)        
    print('label_list', label_list)
    print('label_dict', label_dict)

    label_dict_sw = {v: k for k, v in label_dict.items()}   # swap keys and values
    print('label_dict_sw', label_dict_sw)

    # Loading img files by keras img_load

    X_data = []
    y_data = []
    for class_name in label_list:
        for root, dirs, files in os.walk("{}/img/".format(pj_dir) + "{}".format(class_name)):
            for file in files:
                img = img_to_array(load_img("{}/img/".format(pj_dir) + "{}/".format(class_name) + "{}".format(file), target_size=(pxw,pxh,ch)))
    #             plt.imshow(img)
    #             plt.show()
                X_data.append(img)
                label = label_dict_sw[class_name]
    #             print(label)
                y_data.append(label)

    # Converting to np array
    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)

    # Splitting train and validation
    X_train, X_validation, y_train, y_validation = train_test_split(X_data, y_data, test_size=0.2)

    # Converting to float32 and normalizing to 0-1
    X_train = X_train.astype('float32')
    X_validation = X_validation.astype('float32')
    X_train = X_train / 255.0
    X_validation = X_validation / 255.0
                
    # Endoding Label to one-hot

    y_train = to_categorical(y_train, num_classes) 
    y_validation = to_categorical(y_validation, num_classes)

    print(X_train.shape, 'X train samples')
    print(X_validation.shape, 'X test samples')
    print(y_train.shape, 'y train samples')
    print(y_validation.shape, 'y validation samples')

    #####################################################################    
    ###### Creating model

    model = Sequential()

    model.add(Conv2D(32,(3,3), 
                padding='same', 
                input_shape=X_train.shape[1:],
                activation='relu'))
    model.add(Conv2D(32,(3,3),
                padding='same',
                activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,(3,3), 
                padding='same', 
                input_shape=X_train.shape[1:],
                activation='relu'))
    model.add(Conv2D(64,(3,3), 
                padding='same', 
                input_shape=X_train.shape[1:],
                activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax')) # number of labels = num_classes

    model.summary()

    with StringIO() as buf:
        # StringIOに書き込む
        model.summary(print_fn=lambda x: buf.write(x + "\n"))
        # StringIOから取得
        text = buf.getvalue()

    print('text??? = ', text)

    ######Learning##############
    
    early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 2)
    
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    
    
    if early_stopping_enable == 1:
        
        history = model.fit(X_train, 
                            y_train, 
                            batch_size=batchsize, 
                            epochs=epoch, 
                            verbose=1, 
                            validation_split=0.1,
                            callbacks = [early_stopping])
    else:
        
        history = model.fit(X_train, 
                            y_train, 
                            batch_size=batchsize, 
                            epochs=epoch, 
                            verbose=1, 
                            validation_split=0.1,)
        

    score = model.evaluate(X_validation, 
                            y_validation,
                            verbose=1
                            )

    # Evaluate Loss, Acurracy
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    ## Plot, save...
    FIG_SIZE_WIDTH = 12
    FIG_SIZE_HEIGHT = 10
    FIG_FONT_SIZE = 25

    plot_history(history, 
                save_graph_img_path = '{}/chart.png'.format(pj_dir), 
                fig_size_width = FIG_SIZE_WIDTH, 
                fig_size_height = FIG_SIZE_HEIGHT, 
                lim_font_size = FIG_FONT_SIZE)

    # Save Model structure
    open('{}/model.json'.format(pj_dir), 'w').write(model.to_json())  

    # Save H5 file
    model.save('{}/model.h5'.format(pj_dir))

    ## Save history
    with open('{}/history.json'.format(pj_dir), 'wb') as f:
        pickle.dump(history.history, f)
        
    ## Save Label_class_dict as pd.DataFram to CSV
    label_dict_series = {}
    for k,v in label_dict.items():   # 一度pd.Seriesに変換
        label_dict_series[k]=pd.Series(v)
    df_label_class = pd.DataFrame(label_dict_series)
    df_label_class.to_csv('{}/label_class.csv'.format(pj_dir))
        
    ###### model for Web scrape
   
    # model.add(Conv2D(32,(3,3), 
    #             padding='same', 
    #             input_shape=X_train.shape[1:],
    #             activation='relu'))
    # model.add(Conv2D(32,(3,3),
    #             padding='same',
    #             activation='relu'))

    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    # model.add(Conv2D(64,(3,3), 
    #             padding='same', 
    #             input_shape=X_train.shape[1:],
    #             activation='relu'))
    # model.add(Conv2D(64,(3,3), 
    #             padding='same', 
    #             input_shape=X_train.shape[1:],
    #             activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(num_classes, activation='softmax')) # number of labels = num_classes

    # model.summary()

    # with StringIO() as buf:
    #     # StringIOに書き込む
    #     model.summary(print_fn=lambda x: buf.write(x + "\n"))
    #     # StringIOから取得
    #     text = buf.getvalue()

    # print('text??? = ', text)

    # ######Learning##############
    # model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    # history = model.fit(X_train, 
    #                     y_train, 
    #                     batch_size=5, 
    #                     epochs=10, 
    #                     verbose=1, 
    #                     validation_split=0.1)
