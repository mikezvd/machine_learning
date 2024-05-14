# This program recognises handwritten digits 
# For this you will need to run on your cmd
# pip install numpy
# pip install opencv-python
# pip install matplotlib
# pip install tensorflow

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt  
import tensorflow as tf

# load the dataset
mnist = tf.keras.datasets.mnist

# split data between training data and testing data
# x being the pixel data and y being the classification (the digit)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# next we normalize the data (ie. scalling the data between 0 and 1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# neuro-network model 
model = tf.keras.models.Sequential()  # basic sequential neuro-network

# add layers to the model

# First layer
# turn grid into a long line (called flattening)
# a 28x28 pixel image will be a long string with 784 pixels
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  

# Second layer
# add a dense layer (this creates a layer where every node 
# will connect to all the ones from layer before)
model.add(tf.keras.layers.Dense(128, activation='relu'))  # using rectifying linear unit

# Third Layer
# this layer is the same as last one to help evaluate the data
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Fourth layer
# this will be our output layer
# is set to 10 units to represent the 10 digits
model.add(tf.keras.layers.Dense(10, activation='softmax')) # softmax makes sure that all outputs (all 10 neurons) add to 1
                                                           # giving the probability (how likely it is) to be a certain number

# compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# traning (fitting) the model
model.fit(x_train, y_train, epochs=6)  # raising the number of epochs will make the model more accurate

# saving the model to possibly be used later without having to retrain it
model.save('handwritten_model.keras')

# To reload the model without having to run the code above we use : 
# model = tf.keras.models.load_model('handwritten_model.keras')

# check it's acccuracy
loss, accuracy = model.evaluate(x_test, y_test)

print(f'The loss is : {loss}')
print(f'It\'s accuracy is : {accuracy}')



# iterate through the images to be tested
# images need to be saved on a folder called "digits" where this current file is saved
# images can be created on Paint with 28x28 pixels as size and saved in png format
# the name format should be "digitX.png" where X is a number starting at 1

image_number = 1
my_path = f"digits/digit{image_number}.png"
while os.path.isfile(my_path):
    
    try:
        img = cv2.imread(my_path)[:,:,0]
        img = np.invert(np.array([img]))
        
        prediction = model.predict(img)     # predict the number on image
        
        print(f"This digit is probably a {np.argmax(prediction)}")  # show prediction
        
        plt.imshow(img[0], cmap=plt.cm.binary)  # show image of the number being predicted
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1
        my_path = f"digits/digit{image_number}.png"


