# blood_noblood_classfication
Using different deep learning model to detect image contain blood or no


**Note: This is the initial phase of the Repo, designed for Classification stage of Blood-No_Blood-classification.
# Directory structure
```bash
├── src
│   ├── Data // Folder contains image_dataset.csv which have all the paths of converted and intial images    
│   ├── results
│   │        └──results
│   │                   └── UNIQUE_TIMESTAMP_FOLDER for EACH RUN
│   ├── saved_model // put the DenseNet121 model from thh drive link
│   ├── app.py // main program to run 
│   ├── dng_to_jpg // preprocessing image to jpg
│   ├── normalize_histogram.py 
│   ├── preprocessing_image.py
│   ├── path_utils.py 
│   ├── time_utils.py
│   ├── resize.py
│   └── requirement.txt // to create virtual  environment
│
│
├── training
│      └── training_and_evaluating.ipynb
├── model
│      └──model_download_link.txt // text file which contain download drive link for all the models
│
├── README.md
└── .gitignore
```

# Training 
Used Google colab to train the different model. Here are the following step to create the model
    
    1. Importing library
    2. Data preprocessing
        2.1 Creating test set
        2.2 Configure the dataset for performance
        2.3 Use data augmentation
        2.4 Rescale pixel values
    3. Create the base model from the pre-trained convnets
        3.1 Compile the model
        3.2 Train the model
        3.3 Learning curves
    4. Fine tuning
        4.1 Un-freeze the top layers of the model
        4.2 Compile the model
    5. Evaluation and prediction 

## Different model accuracy on test data
    1. DenseNet121 --> 93.75 (which i fairly good and that's why we used it)
        1/1 [==============================] - 1s 769ms/step - loss: 0.2913 - accuracy: 0.9375
        Test accuracy : 0.9375
        
    2. MobileNetV2 --> 84.23 
        1/1 [==============================] - 1s 709ms/step - loss: 0.3213 - accuracy: 0.8423
        Test accuracy : 0.8423
    
    3. ResNet50 --> 72.55 
        1/1 [==============================] - 1s 869ms/step - loss: 0.4513 - accuracy: 0.7255
        Test accuracy : 0.7255
    
    4. EfficentNetB2 --> 61.43 
        1/1 [==============================] - 1s 888ms/step - loss: 0.5913 - accuracy: 0.6143
        Test accuracy : 0.6143


# Instruction To Run Test Code

1. Download DenseNet model can be downloaded from here: https://drive.google.com/drive/folders/102B4J_V8TFO4ihs31WcLBzCgq8B3Yc65?usp=sharing

3. Save model file to location: src > saved_model > Densenet121 


4. Create a virtual enviornment to install libraries related to **Blink_detection**
    - Run `python3 -m venv <name of virtual env>`
    - linux: Run source `<name of virtual env>/bin/activate`
    - windows: Run `<name of virtual env>\env\Scripts\activate.bat`
5. Change working directory to 'src'
5. Run `pip install -r requirements.txt`
6. put images in the smart_preview folder
7. Run `python3 app.py`


On successful execution of above command, Output will be found in results/reslult/UNIQUE_TIMESTAMP_FOLDER

# CLI

## Approach 1

1. Put the test images in the smart_preview folder
2. Run python3 app.py
3. Results will be in result/results/ UNIQUE TIME STAMP FOLDER FOR EACH RUN

## Approach 2 for custom input output directory

`parser.add_argument("-dp","--dir_path",
                        help="Path to a directory containing images.", default=os.path.join(os.getcwd(), "Smart_Preview"))`
                        
`parser.add_argument("-op","--output_path",
                        help="Output Path to a directory containing images.", default=os.path.join(os.getcwd(), "results"))`
                        
1. Run `python3 app.py -dp <input_path_directory> -op <output_path_directory>
2. Result will be in output_path_directory


## Thank you
