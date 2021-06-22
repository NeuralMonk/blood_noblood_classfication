# blood_noblood_classfication
Using different deep learning model to detect image contain blood or no


**Note: This is the initial phase of the Repo, designed for Classification stage of Blood-No_Blood-classification.

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

1. Download DenseNet model can be downloaded from here: https://drive.google.com/file/d/1KHRZhqNlXN9xFpZr8y8L6nExc-d7H9S8/view?usp=sharing

3. Save model file to location: src > saved_model > Densenet121 


4. Create a virtual enviornment to install libraries related to **Blink_detection**
    - Run `python3 -m venv <name of virtual env>`
    - linux: Run source `<name of virtual env>/bin/activate`
    - windows: Run `<name of virtual env>\env\Scripts\activate.bat`
5. Change working directory to 'src'
5. Run `pip install -r requirements.txt`
6. Run `python3 app.py`


On successful execution of above command, Output files will be created in output directory

