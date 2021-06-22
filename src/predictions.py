def prediction(path):
    import keras
    from keras.preprocessing.image import load_img, img_to_array 
    from keras.models import load_model
    import PIL
    import numpy as np
    from PIL import Image
    img = load_img(path)
    img = img.resize((160, 160))
    img = img_to_array(img) 

    img = img.reshape( -1,160, 160,3)

    #model = load_model('model1.h5')
    pred =  new_model.predict(img)

    return pred