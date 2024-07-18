#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:40:31 2023

@author: fkxxgis
"""

import os
import glob
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import Model, optimizers, callbacks
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, SimpleRNN
from tensorflow.keras.callbacks import TensorBoard

def DeleteOldFile(file_path):
    all_file = os.listdir(file_path)
    for file in all_file:
        if os.path.isdir(os.path.join(file_path, file)):
            DeleteOldFile(os.path.join(file_path, file))
        else:
            os.remove(os.path.join(file_path, file))

def TestModel(test_y, predict_y):
    new_line = "\n"
    
    pearson_1 = stats.pearsonr(test_y[ : , 0], np.squeeze(predict_y[0]))
    R2_1 = metrics.r2_score(test_y[ : , 0], np.squeeze(predict_y[0]))
    mae_1 = metrics.mean_absolute_error(test_y[ : , 0], np.squeeze(predict_y[0]))
    RMSE_1 = metrics.mean_squared_error(test_y[ : , 0], np.squeeze(predict_y[0]))**0.5
    plt.figure(1)
    plt.scatter(test_y[ : , 0], np.squeeze(predict_y[0]))
    plt.title("Blue")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    maxlims = max(max(test_y[ : , 0]), max(predict_y[0])[0])
    minlims = min(min(test_y[ : , 0]), min(predict_y[0])[0])
    lims = [minlims, maxlims]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    time = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M%S")
    save_path = "D:/Model_data/Picture_new"
    plt.savefig(os.path.join(save_path, "Blue_" + time))
    plt.text(-0.025, 0.015, f"Pearson r = {pearson_1[0]}{new_line}R2 = {R2_1}{new_line}"
             f"MAE = {mae_1}{new_line}RMSE = {RMSE_1}")
    
    #pearson_2 = stats.pearsonr(test_y[ : , 1], predict_y[1])
    #R2_2 = metrics.r2_score(test_y[ : , 1], predict_y[1])
    #mae_2 = metrics.mean_absolute_error(test_y[ : , 1], predict_y[1])
   # RMSE_2 = metrics.mean_squared_error(test_y[ : , 1], predict_y[1])**0.5
    pearson_2 = stats.pearsonr(test_y[ : , 1], np.squeeze(predict_y[1]))
    R2_2 = metrics.r2_score(test_y[ : , 1], np.squeeze(predict_y[1]))
    mae_2 = metrics.mean_absolute_error(test_y[ : , 1], np.squeeze(predict_y[1]))
    RMSE_2 = metrics.mean_squared_error(test_y[ : , 1], np.squeeze(predict_y[1]))**0.5
    plt.figure(1)
    plt.scatter(test_y[ : , 0], np.squeeze(predict_y[1]))
    plt.figure(2)
    plt.scatter(test_y[ : , 1], predict_y[1])
    plt.title("Green")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    maxlims = max(max(test_y[ : , 1]), max(predict_y[1])[0])
    minlims = min(min(test_y[ : , 1]), min(predict_y[1])[0])
    lims = [minlims, maxlims]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    time = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M%S")
    save_path = "D:/Model_data/Picture_new"
    plt.savefig(os.path.join(save_path, "Green_" + time))
    plt.text(-0.025, 0.015, f"Pearson r = {pearson_2[0]}{new_line}R2 = {R2_2}{new_line}"
             f"MAE = {mae_2}{new_line}RMSE = {RMSE_2}")
    
   # pearson_3 = stats.pearsonr(test_y[ : , 2], predict_y[2])
   # R2_3 = metrics.r2_score(test_y[ : , 2], predict_y[2])
   # mae_3 = metrics.mean_absolute_error(test_y[ : , 2], predict_y[2])
   # RMSE_3 = metrics.mean_squared_error(test_y[ : , 2], predict_y[2])**0.5
    pearson_3 = stats.pearsonr(test_y[ : , 2], np.squeeze(predict_y[2]))
    R2_3 = metrics.r2_score(test_y[ : , 2], np.squeeze(predict_y[2]))
    mae_3 = metrics.mean_absolute_error(test_y[ : , 2], np.squeeze(predict_y[2]))
    RMSE_3 = metrics.mean_squared_error(test_y[ : , 2], np.squeeze(predict_y[2]))**0.5
    plt.figure(1)
    plt.scatter(test_y[ : , 0], np.squeeze(predict_y[2]))
    plt.figure(3)
    plt.scatter(test_y[ : , 2], predict_y[2])
    plt.title("Red")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    maxlims = max(max(test_y[ : , 2]), max(predict_y[2])[0])
    minlims = min(min(test_y[ : , 2]), min(predict_y[2])[0])
    lims = [minlims, maxlims]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.text(-0.025, 0.015, f"Pearson r = {pearson_3[0]}{new_line}R2 = {R2_3}{new_line}"
             f"MAE = {mae_3}{new_line}RMSE = {RMSE_3}")
    time = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M%S")
    save_path = "D:/Model_data/Picture_new"
    plt.savefig(os.path.join(save_path, "Red_" + time))
data_path = "D:/Extact_data/15_Train_Model_New/Train_Model_0715_Main -1.csv"
tensorboard_log_path = "D:/Model_data/tensorboard_NoSI"
epoch_path = "D:/Model_data/epoch_new"
model_path = "D:/Model_data/model_new"
sample_random_seed = 21 
train_frac = 0.8
train_x_1_colums = ["blue", "NDVI", "days", "sola", "temp", "prec", "soil", "blue_h_dif", "ndvi_h_dif", "Cro", "DBF", "DNF", "EBF", "ENF", "Gra", "Sav", "Shr"]
train_x_2_colums = ["green", "NDVI", "days", "sola", "temp", "prec", "soil", "green_h_dif", "ndvi_h_dif", "Cro", "DBF", "DNF", "EBF", "ENF", "Gra", "Sav", "Shr"]
train_x_3_colums = ["red", "NDVI", "days", "sola", "temp", "prec", "soil", "red_h_dif", "ndvi_h_dif", "Cro", "DBF", "DNF", "EBF", "ENF", "Gra", "Sav", "Shr"]
initial_learning_rate = 0.1
decay_steps = 3000
decay_rate = 0.99
epochs = 300
batch_size = 1024
validation_split = 0.2
initializer = tf.keras.initializers.GlorotUniform()


data = pd.read_csv(data_path)
data.dropna(inplace = True)
dummies = pd.get_dummies(data["PointType"], dtype = int)
data = pd.concat([data, dummies], axis = 1)
data.drop(["PointType"], axis = 1, inplace = True)
data[["blue", "green", "red", "inf", "NDVI", "NDVI_dif", "days", "sola", "temp", "prec", "soil",
      "blue_h_dif", "green_h_dif", "red_h_dif", "inf_h_dif", "ndvi_h_dif"]] = StandardScaler().fit_transform(data[["blue", "green", "red", "inf",
            "NDVI", "NDVI_dif", "days", "sola", "temp", "prec", "soil",
            "blue_h_dif", "green_h_dif", "red_h_dif", "inf_h_dif", "ndvi_h_dif"]])

data = data.sample(frac = 1, random_state = sample_random_seed)
train_size = int(len(data) * train_frac)
train_data = data[ : train_size]
test_data = data[train_size : ]
train_y = train_data.iloc[ : , 6 : 9].values
train_x_1 = train_data[train_x_1_colums].values
train_x_2 = train_data[train_x_2_colums].values
train_x_3 = train_data[train_x_3_colums].values
test_y = test_data.iloc[ : , 6 : 9].values
test_x_1 = test_data[train_x_1_colums].values
test_x_2 = test_data[train_x_2_colums].values
test_x_3 = test_data[train_x_3_colums].values

DeleteOldFile(tensorboard_log_path)
DeleteOldFile(epoch_path)
DeleteOldFile(model_path)

tensorboard_callback = TensorBoard(log_dir = tensorboard_log_path, write_images = True)
epoch_callback = callbacks.ModelCheckpoint(os.path.join(epoch_path, "weights.hdf5"),
                                           monitor = "val_loss", save_weight_only = False,
                                           save_best_only = True)

input_1 = Input(shape = (17, ), name = "input_1")
input_2 = Input(shape = (17, ), name = "input_2")
input_3 = Input(shape = (17, ), name = "input_3")

dense_1 = Dense(2048, activation = "relu", name = "Dense_1", kernel_initializer = initializer)(input_1)
dense_2 = Dense(2048, activation = "relu", name = "Dense_2", kernel_initializer = initializer)(input_2)
dense_3 = Dense(2048, activation = "relu", name = "Dense_3", kernel_initializer = initializer)(input_3)

dropout_1 = Dropout(rate = 0.1, name = "Dropout_1")(dense_1)
dropout_2 = Dropout(rate = 0.1, name = "Dropout_2")(dense_2)
dropout_3 = Dropout(rate = 0.1, name = "Dropout_3")(dense_3)

concatenate = Concatenate(name = "Concatenate")([dropout_1, dropout_2, dropout_3])

independent_1_1 = Dense(2048, activation = "relu", name = "Independent_1_1", kernel_initializer = initializer)(concatenate)
dropout_1_1 = Dropout(rate = 0.05, name = "Dropout_1_1")(independent_1_1)
independent_1_2 = Dense(2048, activation = "relu", name = "Independent_1_2", kernel_initializer = initializer)(dropout_1_1)
 dropout_1_2 = Dropout(rate = 0.1, name = "Dropout_1_2")(independent_1_2)
independent_1_3 = Dense(1024, activation = "relu", name = "Independent_1_3", kernel_initializer = initializer)(dropout_1_2)
 dropout_1_3 = Dropout(rate = 0.1, name = "Dropout_1_3")(independent_1_3)
independent_1_4 = Dense(512, activation = "relu", name = "Independent_1_4", kernel_initializer = initializer)(dropout_1_3)
independent_1_5 = Dense(512, activation = "relu", name = "Independent_1_5", kernel_initializer = initializer)(independent_1_4)

independent_2_1 = Dense(2048, activation = "relu", name = "Independent_2_1", kernel_initializer = initializer)(concatenate)
dropout_2_1 = Dropout(rate = 0.05, name = "Dropout_2_1")(independent_2_1)
independent_2_2 = Dense(2048, activation = "relu", name = "Independent_2_2", kernel_initializer = initializer)(dropout_2_1)
 dropout_2_2 = Dropout(rate = 0.1, name = "Dropout_2_2")(independent_2_2)
independent_2_3 = Dense(1024, activation = "relu", name = "Independent_2_3", kernel_initializer = initializer)(dropout_2_2 )
dropout_2_3 = Dropout(rate = 0.1, name = "Dropout_2_3")(independent_2_3)
independent_2_4 = Dense(512, activation = "relu", name = "Independent_2_4", kernel_initializer = initializer)(dropout_2_3)
independent_2_5 = Dense(512, activation = "relu", name = "Independent_2_5", kernel_initializer = initializer)(independent_2_4)

independent_3_1 = Dense(2048, activation = "relu",name = "Independent_3_1", kernel_initializer = initializer)(concatenate)
dropout_3_1 = Dropout(rate = 0.05, name = "Dropout_3_1")(independent_3_1)
independent_3_2 = Dense(2048, activation = "relu", name = "Independent_3_2", kernel_initializer = initializer)(dropout_3_1)
 dropout_3_2 = Dropout(rate = 0.1, name = "Dropout_2_2")(independent_2_2)
independent_3_3 = Dense(1024, activation = "relu", name = "Independent_3_3", kernel_initializer = initializer)( dropout_3_2 )
 dropout_3_3 = Dropout(rate = 0.1, name = "Dropout_2_3")(independent_2_3)
independent_3_4 = Dense(512, activation = "relu", name = "Independent_3_4", kernel_initializer = initializer)( dropout_3_3)
independent_3_5 = Dense(512, activation = "relu", name = "Independent_3_5", kernel_initializer = initializer)(independent_3_4)

output_1 = Dense(1, activation = "linear", name = "Output_1")(independent_1_5)
output_2 = Dense(1, activation = "linear", name = "Output_2")(independent_2_5)
output_3 = Dense(1, activation = "linear", name = "Output_3")(independent_3_5)

model = Model(inputs = [input_1, input_2, input_3],
              outputs = [output_1, output_2, output_3])
lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate = initial_learning_rate,
                                                    decay_steps = decay_steps,
                                                    decay_rate = decay_rate,
                                                    staircase = True)
model.compile(optimizer = optimizers.SGD(learning_rate = lr_schedule), loss = ["mse", "mse", "mse"],
              metrics = ["mae"])

history = model.fit(x = [train_x_1, train_x_2, train_x_3],
                    y = train_y, epochs = epochs, batch_size = batch_size,
                    callbacks = [tensorboard_callback, epoch_callback], validation_split = validation_split)
                    # callbacks = tensorboard_callback, validation_split = validation_split)

epoch_best = max(glob.glob(epoch_path + "/*"), key = os.path.getmtime)
epoch_best_model = model.load_weights(epoch_best)
predict_y = model.predict([test_x_1, test_x_2, test_x_3])
TestModel(test_y, predict_y)
model.save(model_path)