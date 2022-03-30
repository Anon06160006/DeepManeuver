import numpy as np
import keras
import os, cv2, csv
from DAVE2 import Model
from PIL import Image
# adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DatasetGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                 feature="steering", shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.feature = feature
        self.shuffle = shuffle
        self.training_dir = 'H:/BeamNG_DAVE2_racetracks/'
        # self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        print(int(np.floor(len(self.list_IDs) / self.batch_size)))
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        return 1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #
        # # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y1, y2 = self.data_generation(index)
        return X, y1, y2

    # def on_epoch_end(self):
    #     'Updates indexes after each epoch'
    #     self.indexes = np.arange(len(self.list_IDs))
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)

    def data_generation(self, i):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        # for i, ID in enumerate(list_IDs_temp):
        #     # Store sample
        #     X[i,] = np.load('data/' + ID + '.npy')
        #
        #     # Store class
        #     y[i] = self.labels[ID]
        # 'H:/BeamNG_DAVE2_racetracks/'
        training_dir = "{}training_images_industrial-racetrackstartinggate{}".format(self.training_dir, i)
        m = Model()
        X, y1, y2 = self.process_training_dir(training_dir, m)
        print(X.shape, y1.shape)
        return X, y1, y2

    def process_csv(self, filename):
        global path_to_trainingdir
        hashmap = {}
        with open(filename) as csvfile:
            metadata = csv.reader(csvfile, delimiter=',')
            for row in metadata:
                imgfile = row[0].replace("\\", "/")
                hashmap[imgfile] = row[1:]
        return hashmap

    # retrieve number of samples in dataset
    def get_dataset_size(self, dir_filename):
        dir1_files = os.listdir(dir_filename)
        dir1_files.remove("data.csv")
        return len(dir1_files)

    # assumes training directory has 10,000 samples
    # resulting size: (10000, 150, 200, 3)
    def process_training_dir(self, trainingdir, m):
        td = os.listdir(trainingdir)
        td.remove("data.csv")
        size = self.get_dataset_size(trainingdir)
        X_train = np.empty((size, 150, 200, 3))
        steering_Y_train = np.empty((size))
        throttle_Y_train = np.empty((size))
        hashmap = self.process_csv("{}\\data.csv".format(trainingdir))
        for index, img in enumerate(td):
            img_file = "{}/{}".format(trainingdir, img)
            image = Image.open(img_file)
            # image = cv2.imwrite(img_file)
            image = m.process_image(np.asarray(image))
            # plt.imshow(np.reshape(image, m.input_shape))
            # plt.pause(0.00001)
            X_train[index] = image.copy()
            y = float(hashmap[img][2])
            throttle_Y_train[index] = y
            y = float(hashmap[img][0])
            steering_Y_train[index] = y
        # return np.asarray(X_train), np.asarray(steering_Y_train), np.asarray(throttle_Y_train)
        return X_train, steering_Y_train, throttle_Y_train
        # print(len(X_train))
        # print(len(steering_Y_train))
        # return np.asarray(X_train), np.asarray(steering_Y_train)


    def process_enumerated_training_dirs(self, filename_root, trainingdir_indices, m):
        X_train = np.empty((10000 * len(trainingdir_indices), 150, 200, 3))
        steering_Y_train = np.empty((10000 * len(trainingdir_indices)))
        throttle_Y_train = np.empty((10000 * len(trainingdir_indices)))
        for i in trainingdir_indices:
            trainingdir = "{}{}".format(filename_root, i)
            td = os.listdir(trainingdir)
            td.remove("data.csv")
            hashmap = self.process_csv("{}\\data.csv".format(trainingdir))
            for index, img in enumerate(td):
                img_file = "{}/{}".format(trainingdir, img)
                image = Image.open(img_file)
                # image = cv2.imwrite(img_file)
                image = m.process_image(np.asarray(image))
                # plt.imshow(np.reshape(image, m.input_shape))
                # plt.pause(0.00001)
                adjusted_index = index + i * 10000
                X_train[adjusted_index] = image.copy()
                steering_Y_train[adjusted_index] = float(hashmap[img][1])
                throttle_Y_train[adjusted_index] = float(hashmap[img][2])
            return X_train, steering_Y_train, throttle_Y_train


    def process_all_training_dirs(self, m):
        rootdir = 'H:/BeamNG_DAVE2_racetracks_all/'
        dirs = os.listdir(rootdir)
        sizes = []
        for d in dirs:
            sizes.append(self.get_dataset_size(rootdir + d))
        X_train = np.empty((sum(sizes), 150, 200, 3))
        steering_Y_train = np.empty((sum(sizes)))
        throttle_Y_train = np.empty((sum(sizes)))
        for i,d in enumerate(dirs):
            trainingdir = "{}{}".format(rootdir, d)
            td = os.listdir(trainingdir)
            td.remove("data.csv")
            hashmap = self.process_csv("{}\\data.csv".format(trainingdir))
            for index, img in enumerate(td):
                img_file = "{}/{}".format(trainingdir, img)
                image = Image.open(img_file)
                image = m.process_image(np.asarray(image))
                adjusted_index = sum(sizes[:i]) + index
                X_train[adjusted_index] = image.copy()
                steering_Y_train[adjusted_index] = float(hashmap[img][1])
                throttle_Y_train[adjusted_index] = float(hashmap[img][2])
            return X_train, steering_Y_train, throttle_Y_train