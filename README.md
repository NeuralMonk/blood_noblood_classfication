# blood_noblood_classfication
Using different deep learning model to detect image contain blood or no


**Note: This is the initial phase of the Repo, designed for Classification stage of Blood-No_Blood-classification.


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

