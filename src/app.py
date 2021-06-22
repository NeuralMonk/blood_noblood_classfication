#################################################
from numpy.lib.function_base import append
import time_util
import normalize_histogram
import tensorflow_text
import dng_to_jpg

import paths_util
import numpy as np
from keras.preprocessing import image
import os
import shutil
import pandas as pd
from pathlib import Path
import sys
from itertools import repeat
import multiprocessing as mp

import argparse
import shutil
import datetime
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
import os

import resize
#################################################

# CONSTANTS FOR WORKING WITH CSV

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INDEX = 'index'
KEY = 'key'
FILE_NAME = 'file_name'
FILE_PATH = 'file_path'
JPG_PATH = 'jpg_path'
FEATURES = 'features'
VECTOR = 'vector'
CLUSTER_INDEX = 'cluster_index'
CONVERT_JPG_PATH = 'convert_jpg_path'
SCALED_JPG_PATH = 'scaled_jpg_path'
NORMALIZED_JPG_PATH = 'normalized_jpg_path'
KEEP_IMAGE = 'keep_image'

#################################################

# CONSTANTS FOR RUNNING APPLICATION

# resize_MAX_LENGTH = 1800
# SIMILARITY_THRESHOLD = 0.93

#################################################

# Initializing a checkpoint_tracker to get performance stats
track_checkpoints = time_util.checkpoint_tracker()

# The Decorator @time_util.checkpoint_tracker().create_checkpoint("header_here")
# adds a checkpoint as seen in the code below.
# Each checkpoint stage should be defined by a unique header.
# This header holds the total time usage of the function the
# Decorator w the unique header is placed on.
# The performance stats can be accessed by:
# checkpoint_tracker().get_performance() which returns a dict.

# Syntactic Sugar to make a prettier decorator
create_checkpoint = track_checkpoints.create_checkpoint


class create_session:
    def __init__(self, images_dir,resize_max_length,similarity_threshold):
        self.print_start()
        self.session = time_util.timestamp_simple()
        print(
            f'--- New Session: {self.session }, Created for Directory: {images_dir} ---')
        self.images_dir = images_dir
        # self.session_out_path = Path("data") / "main_run" / self.session
        self.session_out_path = Path("Data")
        self.images_dataframe = self.queue_images(images_dir)
        self.images_dataframe[JPG_PATH] = self.images_dataframe[FILE_PATH]
        self.preprocess_flag = False
        self.resize_max_length = resize_max_length
        self.similarity_threshold = similarity_threshold



    @create_checkpoint("queue_images")
    def queue_images(self, files_dir):
        """
        input:
            files_dir: Path to the Directory containing files/images.
        output:
            images_dataframe with values initialized
        """
        files_dir = Path(files_dir)

        file_list = list(files_dir.glob("**/*"))

        file_list = [item for item in file_list if not item.stem.startswith('.') and item.is_file()]

        total = len(file_list)

        print(f"[INFO] Running queue_images for session: {self.session}\n")
        temp_list = []
        for i, image_path in enumerate(file_list):
            temp_list.append([i, image_path.stem,
                              image_path.name, str(image_path), str(image_path)])
            paths_util.printProgressBar(i + 1, total)
        images_dataframe = pd.DataFrame(
            data=temp_list,
            columns=[INDEX, KEY, FILE_NAME, FILE_PATH, JPG_PATH])

        #save_csv(images_dataframe, "images_dataset.csv",self.session_out_path)
        return images_dataframe

    @create_checkpoint("preprocessing")
    def preprocess(
            self,
            mode,
            dng_convert_flag=True,
            normalize_histogram_flag=True,
            resize_flag=True
    ):
        """
        input:
            mode: an option to select the way preprocessed output is generated.
                "hard_overwrite" -> Does not care about session, dumps and overwrites
                    all files into: data/processed
                "soft_overwrite" -> Takes Session into account, but dumps and overwrites
                    files into: data/<session>/processed
                "no_overwrite" -> Consumes the most disk space, creates a new folder for
                    each preprocessing step and stores all images at each step
        output:
            stores list of jpg paths for each file in self.imagedataframe["jpg_path"]
        """

        print(f"[INFO] Running Preprocess with mode = {mode}")

        self.preprocess_flag = True

        file_list = self.images_dataframe[FILE_PATH]
        print("file list")
        print(file_list)
        out_path = self.session_out_path / "processed"

        # Defining a pipeline
        # Each step or function is made of a single dictionary
        # Add new steps and fill all values
        # name -> name of the step
        # flag -> variable to boolean controlling weather the step is run or not
        # function -> create a function inside the class and add its refrence here
        # out_path -> default out_path for "no_overwrite", unique path to a folder
        #   inside data/main_run/<session>/processed/<new step folder>
        # key -> key for storing list of paths inside self.images_dataframe

        pipeline = [
            {"name": "dng_to_jpg",
             "flag": dng_convert_flag,
             "function": self.run_dng_to_jpg,
             "out_path": str(out_path / "converted_images"),
             "key": CONVERT_JPG_PATH
             },
            {"name": "normalize_histogram",
             "flag": normalize_histogram_flag,
             "function": self.run_normalize_histogram,
             "out_path": str(out_path / "normalized_images"),
             "key": NORMALIZED_JPG_PATH
             },
            {"name": "resize",
             "flag": resize_flag,
             "function": self.run_resize,
             "out_path": str(out_path / "resized_bw_images"),
             "key": SCALED_JPG_PATH
             }
        ]

        list_style = '\n\t-> '

        steps_name = list_style.join([step['name'] for step in pipeline])
        print(
            f"[INFO] Running preprocess() with mode = \"{mode}\" and steps: {list_style}{steps_name}")

        print("flags: ",normalize_histogram_flag)

        if mode.lower() == "hard_overwrite":
            out_path = Path("data") / "processed" / "transformed_images"
            for step in pipeline:
                if step["flag"]:
                    jpg_path_list = step["function"](file_list, str(out_path))
                    file_list = jpg_path_list

        elif mode.lower() == "soft_overwrite":
            out_path = self.session_out_path / "processed"
            for step in pipeline:
                if step["flag"]:
                    jpg_path_list = step["function"](file_list, str(out_path))
                    file_list = jpg_path_list

        elif mode.lower() == "no_overwrite":
            # default value in pipeline is for no_overwrite.

            for step in pipeline:
                if step["flag"]:
                    jpg_path_list = step["function"](
                        file_list, step['out_path'])
                    file_list = jpg_path_list
                    self.images_dataframe[step["key"]] = jpg_path_list


        else:
            print(
                f"[ERROR] Wrong Choice Selected for preprocess(mode = '${mode}').\n${mode} is not a choice.")
            exit(0)

        print("images_dataframe columns: ", self.images_dataframe.columns)

        # # Running all functions inside the pipeline
        # for step in pipeline:
        #     if step["flag"]:
        #         jpg_path_list = step["function"](file_list, step['out_path'])
        #         file_list = jpg_path_list

        self.images_dataframe[JPG_PATH] = jpg_path_list

        save_csv(self.images_dataframe, "images_dataset.csv",self.session_out_path)

    @create_checkpoint("dng_to_jpg_convert")
    def run_dng_to_jpg(self, file_list, out_path):

        dng_path_list = []

        dng_path_index = []
        for i, img_path in enumerate(file_list):
            if img_path.lower().endswith(".dng"):
                dng_path_list.append(img_path)
                dng_path_index.append(i)

        if len(dng_path_list) > 0:

            out_path = Path(out_path)
            out_path.mkdir(exist_ok=True, parents=True)

            print(f"[INFO] Running dng_to_jpg for session: {self.session}\n")

            # with mp.Pool(mp.cpu_count()) as p:
            #     dng_converted_path_list = p.starmap(dng_to_jpg.convert, zip(
            #         dng_path_list, repeat(str(out_path))))

            dng_converted_path_list = []
            for i,j in zip(dng_path_list,repeat(out_path)):
                dng_converted_path_list.append(dng_to_jpg.convert(i,j))


            for i, index in enumerate(dng_path_index):
                file_list[index] = dng_converted_path_list[i]

            jpg_path_list = file_list
            print(jpg_path_list)
            # jpg_path_list = []
            # for i, file_path in enumerate(file_list):
            #     jpg_path_temp = dng_to_jpg.convert(
            #         file_path, str(out_path))  # dng path
            #     jpg_path_list.append(jpg_path_temp)
            #     paths_util.printProgressBar(i+1, self.total)

            return jpg_path_list
        else:
            print(
                f"[INFO] NO .dng files found in directory for Session: {self.session}")
            print("[INFO] Skipping dng_to_jpg_convert")
            return file_list

    @create_checkpoint("normalize_histogram")
    def run_normalize_histogram(self, jpg_path_list, out_path):
        out_path = Path(out_path)
        out_path.mkdir(exist_ok=True, parents=True)
        print(
            f"[INFO] Running normalize_histogram for session: {self.session}\n")
        new_jpg_list = []

        with mp.Pool(mp.cpu_count()) as p:
            new_jpg_list = p.starmap(normalize_histogram.normalize_image, zip(
                jpg_path_list, repeat(str(out_path))))

        # for i, jpg_path in enumerate(jpg_path_list):
        #     new_jpg_path = normalize_histogram.normalize_image(
        #         jpg_path, str(out_path))
        #     new_jpg_list.append(new_jpg_path)
        #     paths_util.printProgressBar(i+1, self.total)

        return new_jpg_list

    @create_checkpoint("resize_images")
    def run_resize(self, jpg_path_list, out_path):

        out_path = Path(out_path)
        out_path.mkdir(exist_ok=True, parents=True)

        print(f"[INFO] Running resize_images for session: {self.session}\n")

        new_jpg_path_list = []

        with mp.Pool(mp.cpu_count()) as p:
            new_jpg_path_list = p.starmap(resize.resize_image, zip(
                jpg_path_list, repeat(str(out_path)), repeat(self.resize_max_length)))

        # for i, jpg_path in enumerate(jpg_path_list):
        #     new_jpg_path = resize.resize_image(jpg_path, str(
        #         out_path), self.resize_max_length)  # jpg path
        #     new_jpg_path_list.append(new_jpg_path)
        #     paths_util.printProgressBar(i+1, self.total)

        return new_jpg_path_list

    global save_csv
    def save_csv(dataframe, csv_file_name, session_out_path,index=False):
        """
        Saves a CSV inside a session, taking in a DataFrame, and CSV file Name
        """
        # out_path = Path(self.session_out_path) / "reporting"
        out_path = Path(session_out_path)
        out_path.mkdir(exist_ok=True, parents=True)
        dataframe.to_csv(out_path / csv_file_name, index=index)

    def print_end(self):
        print("\n")
        print('*******************************************')
        print('*************** PROGRAM END ***************')
        print('*******************************************')

    def print_start(self):
        print('*********************************************')
        print('*************** PROGRAM START ***************')
        print('*********************************************')
        print("\n")



if __name__ == "__main__":
    if sys.platform.startswith('win'):
        # On Windows calling this function is necessary.
        mp.freeze_support()
    #pyinstaller with windows & Multipreprocessing get stuck during build
    #above code is used to avoid it

    parser = argparse.ArgumentParser()

    parser.add_argument("-dp","--dir_path",
                        help="Path to a directory containing images.", default=os.path.join(os.getcwd(), "Smart_Preview"))
    parser.add_argument("-op","--output_path",
                        help="Output Path to a directory containing images.", default=os.path.join(os.getcwd(), "results"))
    parser.add_argument("-nh", "--normalizehistogram", action="store_true",
                        help="If flag is given Normalization of Images Histogram is performed.")
    parser.add_argument("-rs", "--resize", action="store_true",
                        help="If flag is given Images are resized as given by default size or different value by using --resizelength arg.")
    parser.add_argument("-st", "--similaritythreshold", default=0.93, type=float,
                        help="Similarity threshold for considering image to be a duplicate. [default: 0.93]")
    parser.add_argument("-rl", "--resizelength", default=1800, type=int,
                        help="Length for the image to be resized to. [default: 1800]")
    parser.add_argument("-wm", "--writemode", default="no_overwrite",
                        help="Writing mode for intermediate files created during processing:\n no_overwrite [default], soft_overwrite, hard_overwrite")
    
    args = parser.parse_args()

    if args.output_path:
        if not Path("results").exists():
            os.mkdir("results")
        unique = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time())[:-7].replace(':', '-')
        newname=unique+ '_SP'
        destination_path=os.path.join(args.output_path, "results",newname)
    else:
        if not Path("results").exists():
            os.mkdir("results")

        if (len(os.listdir("results")) == 0):
            newname = '1' + '_Smart_Preview'
        else:
            p_count = max([int(i.split('_')[0]) for i in os.listdir('results')])
            newname = str(p_count + 1) + '_Smart_Preview'
        destination_path = os.path.join(os.getcwd(),"results",newname) 

    shutil.copytree(args.dir_path, destination_path, dirs_exist_ok=True)
    
    print("flags: ",args.normalizehistogram)

    input_dir = Path(args.dir_path)
    input_list = list(input_dir.glob("**/*"))
    input_list = [item for item in input_list if not item.stem.startswith('.') and item.is_file()]
    print("input_list")
    print(input_list)
    new_session = create_session(destination_path,resize_max_length=args.resizelength, similarity_threshold=args.similaritythreshold)

    new_session.preprocess(mode=args.writemode, normalize_histogram_flag=args.normalizehistogram,
                           resize_flag=args.resize)
    tf.saved_model.LoadOptions(experimental_io_device=None)

    new_model = tf.keras.models.load_model('./saved_model/Densenet121')
    # Check its architecture
    new_model.summary()
    processed_path = Path("Data")
    print("Processed_path:", processed_path)
    images_dataframe = pd.read_csv(processed_path/"images_dataset.csv")

    #image_path = [os.path.join(os.getcwd(), args.dir_path, x) for x in os.listdir(args.dir_path)]

    image_path = images_dataframe[JPG_PATH]

    image_path = list(image_path)

    #from glob import glob
    #import tensorflow as tf
    #image_path_list = glob('data/train/*/*.jpg')
    data = tf.data.Dataset.list_files(image_path)

    def load_images(path):
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image)
        
        return image
    
    data = data.map(load_images)

    def preprocess(image):
        image = tf.image.resize(image, (160, 160))
        image = tf.expand_dims(image, axis=-1)
        #image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.reshape(image, (-1, 160, 160, 3))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)    
        image /= 255.
        image -= 0.5    
        
        return image
    
    data = data.map(preprocess)



    predictions = new_model.predict(data).flatten()
    print(predictions)
    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    df_final = pd.DataFrame()
    #print(len(input_list))
    #print(len(predictions))
    #print(predictions)
    df_final['file_path'] = input_list
    df_final['predictions'] = predictions


    df_final.to_csv(os.path.join(destination_path, "images_dataset_final.csv"), index=False)

    #print(predictions)
    print("DONE")

