import numpy as np
import pandas as pd
import cv2
import matplotlib.image as mpimg
import json

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout
from keras.layers import Activation, Flatten, Lambda, Input, ELU
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

import h5py
import os
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import csv
from DAVE2 import Model
from DatasetGenerator import DatasetGenerator
import time

sample_count = 0
path_to_trainingdir = 'H:/test'

def redo_csv(readfile):
    temp = readfile.split("/")
    writefile = "/".join(temp[:-1]) + "data1.csv"
    #print("writefile:{}".format(writefile))
    with open(readfile) as csvfile:
        with open(writefile, 'w') as csvfile1:
            metadata = csvfile.readlines()
            #metadata = csv.reader(csvfile, delimiter=',')
            for row in metadata:
                #row = row.replace('C:\\Users\\merie\\Documents\\BeamNGpy-master\\BeamNGpy-master\\examples/', '')
                row = row.replace('H:/BeamNG_DAVE2_racetracks/', '')
                row = row.replace('bmp', 'png')
                csvfile1.write(row)
                #print("row:{}".format(row))
    os.remove(readfile)
    os.rename(writefile, readfile)

def process_csv(filename):
    global path_to_trainingdir
    hashmap = {}
    with open(filename) as csvfile:
        metadata = csv.reader(csvfile, delimiter=',')
        for row in metadata:
            imgfile = row[0].replace("\\", "/")
            hashmap[imgfile] = row[1:]
    return hashmap

def process_training_dir(trainingdir, m):
    global sample_count, path_to_trainingdir
    td = os.listdir(trainingdir)
    td.remove("data.csv")
    X_train = []; steering_Y_train = []; throttle_Y_train = []
    hashmap = process_csv("{}\\data.csv".format(trainingdir))
    for img in td:
        img_file = "{}{}".format(trainingdir, img)
        image = cv2.imread(img_file)
        if "bmp" in img_file:
            print("img_file {}".format(img_file))
            compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            temp = image

            # cv2.imwrite(img_file.replace("bmp", "png"), temp, compression_params)
            img = img.replace("bmp", "png")
            Image.fromarray(image).save(img_file.replace("bmp", "png"))
            os.remove(img_file)
        image = m.process_image(image)
        # plt.imshow(np.reshape(image, m.input_shape))
        # plt.pause(0.00001)
        X_train.append(image.copy())
        y = float(hashmap[img][2])
        throttle_Y_train.append(y)
        y = float(hashmap[img][0])
        steering_Y_train.append(y)
        sample_count += 1
    return np.asarray(X_train), np.asarray(steering_Y_train),np.asarray(throttle_Y_train)

def get_endnumber(img_filename):
    img_filename = img_filename.replace(".png", "")
    l = img_filename.split("_")[-1]
    return float(l)


# add dir2 into dir1 until dir1 reaches 10,000 images
def combine_two_datasets(dir1, dir2):
    dir1_files = os.listdir(dir1)
    dir2_files = os.listdir(dir2)
    dir1_files.remove("data.csv")
    dir2_files.remove("data.csv")
    dir1_files.sort(reverse=False, key=get_endnumber)
    dir2_files.sort(reverse=False, key=get_endnumber)
    dir1_csv_filename = "{}/data.csv".format(dir1)
    dir2_csv_filename = "{}/data.csv".format(dir2)

    with open(dir1_csv_filename, "w") as dir1_csv_file:
        with open(dir2_csv_filename, "r") as dir2_csv_file:
            dir2_lines = dir2_csv_file.readlines()
            dir1_count = get_endnumber(dir1_files[-1])
            dir2_processed_count = 0
            while dir1_count + dir2_processed_count < 10000:
                dir2_processed_count += 1
                # change dir2 image name
                new_image_location = dir1 + '{}/hopper_industrial_{}.png'.format(dir1, dir1_count + dir2_processed_count)
                # save image to dir1
                img = cv2.imread("{}".format(dir2_files[0]))
                cv2.imwrite(new_image_location)
                # add corresponding csv line to dir1 csv
                for line in dir2_lines:
                    if dir2_files[0] in line:
                        dir2_line = line
                        dir2_files.remove(dir2_files[0])
                dir2_line.replace()
                dir1_csv_file.write(dir2_line)


def main2():
    global sample_count, path_to_trainingdir
    # Convert training dataframe into images and labels arrays
    # Training data generator with random shear and random brightness
    start_time = time.time()
    # Start of MODEL Definition
    m = Model()
    model = m.define_model()

    # prep training set
    t = os.listdir(path_to_trainingdir)
    training_dirs = ["{}/{}/".format(path_to_trainingdir, training_dir) for training_dir in t]
    # for training_dir in t:
    #     training_dirs.append("{}/{}/".format(path_to_trainingdir, training_dir))

    shape = (0, 1, m.input_shape[0], m.input_shape[1], m.input_shape[2])
    X_train = np.array([]).reshape(shape); y_train = np.array([])
    for d in training_dirs[-1:]:
        print("Processing {}".format(d))
        redo_csv("{}/data.csv".format(d))
        x_temp, steering_y_temp, throttle_y_temp = process_training_dir(d, m)
        print("Concatenating X_train shape:{} x_temp shape:{}".format(X_train.shape, x_temp.shape))
        X_train = np.concatenate((X_train, x_temp), axis=0)
        steering_y_train = np.concatenate((y_train,steering_y_temp), axis=0)
        throttle_y_train = np.concatenate((y_train,throttle_y_temp), axis=0)
    print("Final X_train shape:{} Final y_train shape:{}".format(X_train.shape, steering_y_train.shape))

    # Train and save the model
    BATCH_SIZE = 100
    NB_EPOCH = 9
    NB_SAMPLES = 2*len(X_train)
    # Train steering
    model.fit(x=X_train, y=steering_y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH)
    model_name = 'BeamNGmodel-racetracksteering'
    model.save_weights('{}.h5'.format(model_name))
    with open('{}.json'.format(model_name), 'w') as outfile:
        json.dump(model.to_json(), outfile)
    # Train throttle
    model.fit(x=X_train, y=throttle_y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH)
    model_name = 'BeamNGmodel-racetrackthrottle'
    model.save_weights('{}.h5'.format(model_name))
    with open('{}.json'.format(model_name), 'w') as outfile:
        json.dump(model.to_json(), outfile)
    print("All done :)")
    print("Total training samples: {}".format(sample_count))
    print("Time to train: {}".format(time.time() - start_time))

def main():
    global sample_count, path_to_trainingdir
    # Convert training dataframe into images and labels arrays
    # Training data generator with random shear and random brightness
    start_time = time.time()
    # Start of MODEL Definition
    m = Model()
    model = m.define_model()
    generator = DatasetGenerator(32, path_to_trainingdir, m.input_shape)
    model.fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1,
                        callbacks=None, validation_data=None, validation_steps=None,
                        validation_freq=1, class_weight=None, max_queue_size=10,
                        workers=1, use_multiprocessing=False, shuffle=False, initial_epoch=0)
    print("All done :)")
    print("Time to train: {}".format(time.time() - start_time))

def main3():
    global path_to_trainingdir
    combine_two_datasets(path_to_trainingdir + "/training_images_industrial-racetrackstartinggate4", path_to_trainingdir + "/training_images_industrial-racetrackstartinggate0")

if __name__ == '__main__':
    main3()