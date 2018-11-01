#!/usr/bin/python3
# coding: utf-8

# ---- Description ----
""" A file which contains a sample of neural agent """

# ---- Imports ----
from NAP.data_preprocessor import Data_Preprocessor
import keras.backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import VGG19
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import pickle
import os
import numpy as np
import tensorflowjs as tfjs


# ---- Class ----
class Neural_Agent(Data_Preprocessor):
    """ Used to treat some data """

    key_model = 'model'
    key_weights = 'weights'
    key_weights_file = 'weights_file'
    key_checkpoint = 'checkpoint'
    key_instance_save_path = 'instance_save_path'
    default_save_path = 'basic_instance_save'

    def __init__(self, **kwargs):
        """"
        Init a new Neural Agent
        :param kwargs: Dictionary of parameters
        - optimizer
        - loss
        :type kwargs: dict
        """""
        # --- SAVE TESTING ---
        if (os.path.exists(kwargs.get(self.key_instance_save_path, self.default_save_path))
                and os.path.isfile(kwargs.get(self.key_instance_save_path, self.default_save_path))):
            print('Classifier file already exists, loading it...')
            try:
                self.__dict__ = Neural_Agent._load_instance(
                    kwargs.get(self.key_instance_save_path, self.default_save_path)).__dict__.copy()
                self.load_weights(self.weights_file)
                print('Classifier successfully loaded')
                self.display_categories()
                return
            except EOFError as e:
                print('Failed to load classifier: File corrupted (EOFError)\n')

        # --- SUPER INIT ---
        Data_Preprocessor.__init__(self, **kwargs)

        # --- SELF INIT ---
        weights_file = kwargs.get(self.key_weights_file, 'weights.h5')
        if not weights_file.endswith('.h5'):
            weights_file = weights_file.replace('.', '_')  # format secure file name
            weights_file += '.h5'
        self.weights_file = weights_file
        self.instance_save_path = kwargs.get(self.key_instance_save_path, 'basic_instance_save')

        # --- NEURAL INIT ---
        self.input_shape = self.shape + kwargs.get('depth', (3,))  # by default
        if K.image_data_format() == 'channel_first':
            self.input_shape = kwargs.get('depth', (3,)) + self.shape
        self.optimizer = kwargs.get('optimizer', 'Adam')
        self.loss = kwargs.get('loss', 'binary_crossentropy')
        self.checkpoint = kwargs.get(self.key_checkpoint)
        print('Building neural net...')
        self.model = self._total_net(batch_size=kwargs.get('batch_size', 16),
                                     nbr_epochs=kwargs.get('epochs', 1),
                                     loops=kwargs.get('loops', 1),
                                     verbose=kwargs.get('verbose', 1))
        print('Fitting neural net...')
        self.fit_all(batch_size=kwargs.get('batch_size', 16),
                     epochs=kwargs.get('epochs', 1),
                     loops=kwargs.get('loops', 1),
                     verbose=kwargs.get('verbose', 1))

        print('TFJS saving...')
        tfjs.converters.save_keras_model(self.model, 'test.json')

        # --- STATE SAVE ---
        print('Saving instance...')
        self.save_instance()

    # --- PICKLER MANAGEMENT ---
    def __setstate__(self, state):
        state[self.key_model] = model_from_json(state.get(self.key_model))
        state.get(self.key_model).set_weights(state.get(self.key_weights))
        del state[self.key_weights]
        self.__dict__.update(state)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

    def __getstate__(self):
        state = self.__dict__.copy()
        state[self.key_model] = state.get(self.key_model).to_json()
        state[self.key_weights] = self.model.get_weights()
        del state[self.key_checkpoint]
        return state

    # --- NEURAL NETWORK ---
    def _buildnet(self, input_shape):
        """"
        Build the neural network model and load weights if the weights file exists
        :return: keras.models.Model
        :rtype: keras.models.Model
        """""
        model = Sequential(name="neural_net")
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(len(self.categories), activation='softmax'))
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model

    def _fit_top(self, base, top, loops=1, batch_size=16, nbr_epochs=1, verbose=1):
        for i in range(loops):
            while not self.all_scanned():
                data, target = self.pick_data()
                data = base.predict(data)
                top.fit(data, target,
                        batch_size=batch_size, epochs=nbr_epochs,
                        verbose=verbose)
            for cat in self.categories.values():
                cat[self.cat_all_scanned] = False

        # --- VAR DISCHARGE ---
        for cat in self.categories.values():
            try:
                cat.pop(self.cat_temp_batch)
            except KeyError:
                pass
            cat[self.cat_one_time] = False

    def _prepare_base(self):
        base_model = VGG19(include_top=False, weights='imagenet', input_shape=self.input_shape)
        for layer in base_model.layers:
            layer.trainable = False
        return base_model

    def _total_net(self, loops=1, batch_size=16, nbr_epochs=1, verbose=1):
        base = self._prepare_base()
        top = self._buildnet(base.output_shape[1:])
        self._fit_top(base, top, loops, batch_size, nbr_epochs, verbose)
        model = Model(inputs=base.input, outputs=top(base.output))
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return model

    def predict(self, filepath):
        """ Make a prediction from a text file """
        raw_data = self.extract_from_raw(filepath)
        batch = np.array(self.preprocess_data(raw_data))
        prediction = self.model.predict(batch)
        return prediction.reshape(prediction.shape[1:])

    # --- FITTING ---
    def fit_all(self, batch_size=16, epochs=1, loops=1, verbose=1):
        """ Fit neural network onto data files """
        if not self.checkpoint:
            self.checkpoint = ModelCheckpoint(self.weights_file, monitor='loss',
                                              save_best_only=True,
                                              save_weights_only=True,
                                              mode='min',
                                              verbose=verbose)
        for i in range(loops):
            while not self.all_scanned():
                data, target = self.pick_data()
                self.model.fit(data, target,
                               batch_size=batch_size, epochs=epochs,
                               verbose=verbose,
                               callbacks=[self.checkpoint])
            for cat in self.categories.values():
                cat[self.cat_all_scanned] = False

        # --- VAR DISCHARGE ---
        for cat in self.categories.values():
            try:
                cat.pop(self.cat_temp_batch)
            except KeyError:
                pass
            cat[self.cat_one_time] = False

    # --- SAVE MANAGEMENT ---
    def save_instance(self):
        with open(self.instance_save_path, 'wb') as file:
            pickle.Pickler(file).dump(self)

    @staticmethod
    def _load_instance(filename):
        with open(filename, 'rb') as file:
            return pickle.Unpickler(file).load()

    def load_weights(self, filename: str):
        """"
        Load weights from filename
        :param filename: name of the weights file
        :type filename: str
        :return: None
        :rtype: None
        """""
        try:
            self.model.load_weights(filename)
        except OSError as error:
            print(error)
            print("File '{}' doesn't exist".format(filename))

    def save_weights(self, filename: str):
        """" Save weights of the neural network into the file """""
        self.model.save_weights(filename, overwrite=True)

    # --- OVERRIDING ---
    def preprocess_data(self, raw_data):
        return Data_Preprocessor.preprocess_data(self, raw_data)

    def extract_from_raw(self, file_path):  # used if needed to change
        if (os.path.isdir(file_path)
                or (file_path[-3:].lower() not in ('jpg', 'png')
                    and file_path[-4:].lower() not in ('jpeg',))):
            return
        try:
            img = load_img(file_path, target_size=self.shape)
        except (FileNotFoundError, OSError):
            return
        arr_img = img_to_array(img)
        arr_img = arr_img.reshape((1,) + arr_img.shape)
        return arr_img / 255.

    # --- DISPLAY MANAGEMENT ---
    def display(self, filepath, limit=0):
        """ Display prediction """
        # prepare a display name
        name = os.path.join(os.path.basename(os.path.dirname(filepath)), os.path.basename(filepath))
        prediction = self.predict(filepath)  # predict

        results = []
        for i in range(len(prediction)):  # sort predictions
            results.append([prediction[i], self.search_index(self.categories, i)])
        results = sorted(results, key=lambda res: res[0], reverse=True)

        if not limit:
            limit = len(results)

        print('Predictions for:', name)
        for e in results[:limit]:  # display each category with its result
            print("{0:.2f}% - {1}".format(np.round(e[0] * 100, 2), e[1]))
        print()

    def display_prediction(self, file, limit=0):
        """ Display predictions for a list of texts """
        if type(file) in (tuple, list):
            for path in file:
                self.display_prediction(path, limit=limit)
        elif type(file) is str:
            if os.path.isdir(file):
                files = os.listdir(file)
                for i in range(len(files)):
                    files[i] = os.path.join(file, files[i])
                self.display_prediction(files, limit=limit)
            else:
                self.display(file, limit=limit)
