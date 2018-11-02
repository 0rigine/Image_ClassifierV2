#!/usr/bin/python3
# coding: utf-8

# ---- Description ----
""" Class used to generate, organize and use data """

# ---- Imports ----
import numpy as np
import os
import pickle
import abc
import sys
import re
from keras.preprocessing.image import ImageDataGenerator


# ---- Class ----
class Data_Preprocessor:
    """ Class used to generate, organize and use data """

    byte_to_mb = 1048576.
    # --- CATEGORIES ATTRIBUTES ---
    cat_fileIndex = 'fileIndex'
    cat_inFileIndex = 'inFileIndex'
    cat_index = 'cat_index'
    cat_all_scanned = 'all_scanned'
    cat_one_time = 'one_time'  # has scanned all data files in one time
    cat_dir = 'directory'
    cat_temp_batch = 'temp_batch'  # used only if one time is True

    def __init__(self, **kwargs):
        if 'source_path' not in kwargs.keys():
            raise KeyError('Please specify a sample folder path')
        # --- ATTRIBUTES INITIALIZATION ---
        self.data_per_batch = kwargs.get('data_per_batch', 10.) * self.byte_to_mb
        self.data_per_file = kwargs.get('data_per_file', 50.) * self.byte_to_mb
        self.data_dir = kwargs.get('data_dir', 'Datas')
        # if count letters (True) or words (False), for text treatment
        self.letter_mode = kwargs.get('letter_mode', False)
        self.categories = {}  # contains dictionary of attributes for each category
        self.shape = kwargs.get('dimensions', (300, 300))
        self.datagen_loops = kwargs.get('datagen_loops', 20)  # number of loops made by the image data generator

        # --- CATEGORIES INDEXING ---
        print('Listing categories from sources...')
        self.categories_list(kwargs['source_path'])
        self.display_categories()

        # --- DATA MINING ---
        if not os.path.exists(self.data_dir) or len(os.listdir(self.data_dir)) == 0 or kwargs.get('reuse_data') is None:
            print('Mining data from raw files...')
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)
            self.categories_mining(kwargs.get('overwrite', False))

    # --- DECORATOR ---
    @staticmethod
    def decorator_file_security(to_modify_func):
        def modified_func(*args, **kwargs):
            """ Secure a file interaction """
            # --- File security ---
            try:
                raw = to_modify_func(*args, **kwargs)
            except (IsADirectoryError, FileNotFoundError) as error:
                print(error)
                return
            return raw

        return modified_func

    # --- DATA PREPROCESSING ---
    @abc.abstractmethod
    def preprocess_data(self, raw_data):
        """ Preprocess data for treatment """
        return raw_data

    # --- FILES CROSSING ---
    def categories_list(self, source_path):
        """ List all categories from sample folder """
        for cat in os.listdir(source_path):
            temp_path = os.path.join(source_path, cat)
            if os.path.isdir(temp_path):
                self.categories[cat] = {self.cat_all_scanned: False,
                                        self.cat_one_time: False,
                                        self.cat_fileIndex: 0,
                                        self.cat_inFileIndex: 0,
                                        self.cat_index: len(self.categories),
                                        self.cat_dir: temp_path}

    # --- DATA MINING ---
    def categories_mining(self, overwrite=False):
        """ Preprocess raw data for each category """
        index = 0
        total = len(self.categories)
        for cat_name, cat_dict in self.categories.items():
            index += 1
            print("Mining: {0} ({1} / {2})".format(cat_name, index, total))
            self.categories_data_mining(cat_dict.get(self.cat_dir), cat_name, overwrite)

    def categories_data_mining(self, cat_path, cat_name, overwrite=False):
        """ Preprocess raw data from source for a specified category """
        # --- Data generator ---
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        if os.path.isdir(cat_path):
            sources = os.listdir(cat_path)  # list of image files
            data = []
            for t_path in sources:
                # course of source raw files
                t_path = os.path.join(cat_path, t_path)
                if os.path.isfile(t_path):  # security
                    raw = self.extract_from_raw(t_path)  # try to extract raw data from file
                    if raw is not None:  # in case of no error is raised (image corrupted, not found)
                        t_data = self.preprocess_data(raw)  # try to prepocess raw data
                        maxi = 0
                        if len(data) > 0:
                            data = np.append(data, t_data, axis=0)
                        else:
                            data = np.array(t_data)
                        # split data to get batch as great size
                        for batch in datagen.flow(t_data, batch_size=1):
                            # memory security
                            if sys.getsizeof(np.array(data)) >= self.data_per_file:
                                self.save_data(np.array(data), cat_name, overwrite)
                                data = []  # emtpy data batch after save
                            if len(data) > 0:
                                data = np.append(data, batch, axis=0)
                            else:
                                data = np.array(batch)
                            maxi += 1
                            if maxi > self.datagen_loops:
                                break
            if len(data) > 0:
                self.save_data(np.array(data), cat_name, overwrite)

    def extract_from_raw(self, file_path):
        """ Extract raw from file """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    # --- DATA PICKING ---
    @staticmethod
    def format_text(source_text):
        """
        Format text
        :param source_text: string
        :return: string
        """
        for sign in ('\n', '\ufeff', '\r'):
            source_text = source_text.replace(sign, ' ')
        return source_text.lower()

    @staticmethod
    def count_valid(directory, basename):
        """ Count the number of data files for a category """
        return len(Data_Preprocessor.list_category_files(directory, basename))

    @staticmethod
    def list_category_files(directory, basename):
        """ List data files of a category """
        result = []
        basename = os.path.basename(basename)
        expr = re.compile(r'^' + re.escape(basename) + r'[0-9]+$')
        for filename in os.listdir(directory):
            if expr.search(filename):
                result.append(os.path.join(directory, filename))
        return result

    @staticmethod
    def search_index(dico, index):
        for key, values in dico.items():
            if index == values.get(Data_Preprocessor.cat_index):
                return key
        return ''

    def all_scanned(self):
        """ Test if each category has scanned all of its files """
        for cat_dict in self.categories.values():
            if not cat_dict.get(self.cat_all_scanned):
                return False
        return True

    def all_one_shot(self):
        """ Test if each category has scanned all of its file in one time """
        for cat_dict in self.categories.values():
            if not cat_dict.get(self.cat_one_time):
                return False
        return True

    def forge_target(self, **kwargs):
        """ Used to forge target batch """
        target = [0.] * len(self.categories)
        target[kwargs.get('index')] = 1.
        target = [target] * len(kwargs.get('data'))
        return target

    def pick_from_file(self, size_max, cat_name, cat_dict):
        """ Pick data from a file """
        data = np.array([])
        if cat_dict.get(self.cat_fileIndex) < self.count_valid(self.data_dir, cat_name):
            # if still data file to read

            # --- FILEPATH FINDING ---
            filename = os.path.join(self.data_dir, cat_name + str(cat_dict.get(self.cat_fileIndex)))
            file_data = self.load_data(filename)

            # --- FILL BATCH ---
            while (sys.getsizeof(data) < size_max
                   and cat_dict.get(self.cat_inFileIndex) < len(file_data)):
                temp = file_data[cat_dict.get(self.cat_inFileIndex)]
                temp = np.reshape(temp, (1,) + temp.shape)
                if sys.getsizeof(temp) + sys.getsizeof(data) >= size_max:
                    break
                if len(data) > 0:
                    data = np.append(data, temp, axis=0)
                else:
                    data = temp
                cat_dict[self.cat_inFileIndex] += 1

            # --- ADJUST INDEXES ---
            if not cat_dict.get(self.cat_inFileIndex) < len(file_data):
                # if at the end of the data file
                cat_dict[self.cat_fileIndex] += 1
                cat_dict[self.cat_inFileIndex] = 0

        # --- RESULTING BATCH ---
        return data

    def pick_category(self, size_max, cat_name):
        """ Pick data from category data files """
        cat_dict = self.categories.get(cat_name)
        data = cat_dict.get(self.cat_temp_batch, np.array([]))

        if len(data) == 0:
            initial_index = cat_dict.get(self.cat_fileIndex)
            total_valid = self.count_valid(self.data_dir, cat_name)

            # --- FILL DATA LOOP ---
            while (sys.getsizeof(np.array(data)) < size_max
                   and cat_dict.get(self.cat_fileIndex) < total_valid):
                data = self.pick_from_file(size_max, cat_name, cat_dict)

            # --- TEST INDEXES ---
            if cat_dict.get(self.cat_fileIndex) >= total_valid:
                cat_dict[self.cat_all_scanned] = True
                cat_dict[self.cat_inFileIndex] = 0
                cat_dict[self.cat_fileIndex] = 0
                if initial_index == 0:
                    cat_dict[self.cat_one_time] = True

        # --- FORGE TARGET ---
        target = self.forge_target(index=cat_dict.get(self.cat_index), data=data)
        # --- RETURN BATCH ---
        return data, target

    def pick_data(self):
        """ Extract data from data files """
        data, target = [], []
        max_per_category = float(self.data_per_batch) / len(self.categories)
        for category_name in self.categories.keys():  # for each known category
            cat_data, cat_target = self.pick_category(max_per_category, category_name)
            if len(data) > 0:
                data = np.append(data, cat_data, axis=0)
                target = np.append(target, cat_target, axis=0)
            else:
                data = cat_data
                target = cat_target
        return np.array(data), np.array(target)

    # --- DATA SAVING ---
    def save_data(self, data, filename, overwrite=False):
        """ Save data into file """
        # --- PATH CONTROL ---
        if self.data_dir not in filename:
            filename = os.path.join(self.data_dir, filename)
        # --- OVERWRITING CONTROL ---
        if not overwrite:
            num = self.count_valid(self.data_dir, filename)
            match = re.search(r'[0-9]+$', filename)
            if match:
                filename = filename.replace(match.group(), '')
            filename += str(num)
        # --- SAVING ---
        print('Saving data in file:', filename)
        with open(filename, 'wb') as file:
            pickle.Pickler(file).dump(data)

    def load_data(self, filename=None):
        """ Load data from file """
        # --- PATH CONTROL ---
        if filename is None:  # random pick of data file
            dirList = os.listdir(self.data_dir)
            filename = np.random.choice(dirList)
        elif self.data_dir not in filename:  # defined pick of data file
            filename = os.path.join(self.data_dir, filename)
        # --- READING DATAS ---
        with open(filename, 'rb') as file:
            data = pickle.Unpickler(file).load()
        # --- RETURN ---
        return data

    # --- DISPLAY MANAGEMENT ---
    def display_categories(self):
        """" Display all categories treated by the model """""
        print('Total categories: {}'.format(len(self.categories)))
        for category in self.categories.keys():
            end = ' - '
            if self.categories.get(category).get(self.cat_index) == len(self.categories) - 1:
                end = '\n' * 2
            print(category, end=end)
