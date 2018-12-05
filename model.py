import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

# Most of the code is from snippets provided by Udacity material.
samples = []
# Load CSV file and ignore the header.
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:     
        samples.append(line)

# Split Dataset. Validation gets 20%.               
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# create adjusted steering measurements for the side camera images
correction = 0.2 # this is a parameter to tune
path = './data/IMG/'
def generator(samples, train_mode, batch_size=32):
    
    image_shape = (160,320,3) # original image shape
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                #Get name of image stored under IMG directory
               name_center = path + batch_sample[0].split('/')[-1]
               # Read Image
               center_image = cv2.imread(name_center)
               #print(batch_sample[3])
               
               try:
                    center_angle = float(batch_sample[3])
               except:
                    center_angle = 0
               #print()     
               if train_mode == True:
                    # if training set then add left and right camera images.
                    name_left = path + batch_sample[1].split('/')[-1]
                    name_right = path + batch_sample[2].split('/')[-1]
                    
                    left_image = cv2.imread(name_left)
                    right_image = cv2.imread(name_right)
                   
                    left_angle = center_angle + correction
                    right_angle = center_angle - correction

                    images.extend([center_image, left_image, right_image])
                    angles.extend([center_angle, left_angle, right_angle])

                    #Flip images
                    #image_flipped = np.fliplr(image)
                    flipped_center_image = cv2.flip(center_image, 1)
                    flipped_left_image = cv2.flip(left_image, 1)
                    flipped_right_image = cv2.flip(right_image, 1)
                    #Flip angles
                    flipped_center_angle = -center_angle
                    flipped_left_angle =  -left_angle
                    flipped_right_angle = -right_angle 

                    images.extend([flipped_center_image, flipped_left_image, flipped_right_image])
                    angles.extend([flipped_center_angle, flipped_left_angle, flipped_right_angle])
               else:
                    images.append(center_image)
                    angles.append(center_angle)
                
          
           
            X_train = np.array(images)
            y_train = np.array(angles)
            X_train = X_train.flatten().reshape([-1, image_shape[0], image_shape[1], image_shape[2]])
            y_train = y_train.flatten().reshape([-1, 1])
        
            #print(" X_train Shape ", X_train.shape)
            #print(" y_train Shape ", y_train.shape)
            yield shuffle(X_train, y_train)




# compile and train the model using the generator function
train_generator = generator(train_samples, train_mode=True,  batch_size=32)
validation_generator = generator(validation_samples, train_mode=False, batch_size=32)

# Keras forum had a suggestion on resizing
def resize_and_normalize(image):
    import cv2
    from keras.backend import tf as ktf   
    resized = ktf.image.resize_images(image, (66, 200))
    #normalize 0-1
    resized = resized/255.0 - 0.5

    return resized
from keras.layers import Cropping2D, Conv2D, Flatten, Dense, Lambda
from keras.models import Model, Sequential
import cv2
# NVIDIA's autonomous vehicle project architecture is used below
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
#model.add(Lambda(lambda x: x/127.5 - 1.,
        #input_shape=(ch, row, col),
        #output_shape=(ch, row, col)))

# set up cropping2D layer
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
#model.add(... finish defining the rest of your model architecture here ...)
#model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,input_shape=(3, nrows,ncols)))
# trim image to only see section with road
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
#Resize image to fit the architecture
model.add(Lambda(resize_and_normalize, input_shape=(80, 320, 3), output_shape=(66, 200, 3)))

model.add(Conv2D(24, (5,5) , padding='valid', activation='relu', strides=(2,2)))

model.add(Conv2D(36, (5,5), padding='valid', activation='relu', strides=(2,2)))

model.add(Conv2D(48, (5,5), padding='valid', activation='relu', strides=(2,2)))

model.add(Conv2D(64, (3,3), padding='valid', activation='relu', strides=(1,1)))

model.add(Conv2D(64, (3,3),padding='valid', activation='relu', strides=(1,1)))

model.add(Flatten())

model.add(Dense(1164, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(1, activation='tanh'))

model.summary()

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch= 
            len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=3)
# Save model to JSON
#with open('model.json', 'w') as outfile:
    #outfile.write(json.dumps(json.loads(model.to_json()), indent=2))
# Save model to h5 file    
model.save('model.h5')

import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
