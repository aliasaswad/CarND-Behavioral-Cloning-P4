import os
import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D




def combine_images(c_images, l_images, r_images, measure):
    
    image_paths = []
    image_paths.extend(c_images)
    image_paths.extend(l_images)
    image_paths.extend(r_images)
    measurements = []
    measurements.extend(measure)
    correct = 0.2
    measurements.extend([x + correct for x in measure])
    measurements.extend([x - correct for x in measure])
    return (image_paths, measurements)

def get_lines(data_path):
    
    lines = []
    with open(data_path + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        next(reader, None)
        for line in reader:
            lines.append(line)
    return lines

def get_images(data_path):
    
    folders = [x[0] for x in os.walk(data_path)]
    data_folders = list(filter(lambda fold: os.path.isfile(fold + '/driving_log.csv'), folders)
    center_images = []; left_images = []; right_images = []; measurement_total = []
    for folder in data_folders:
        lines = get_lines(folder)
        center = []; left = []; right = []; measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            center.append(folder + '/' + line[0].strip())
            left.append(  folder + '/' + line[1].strip())
            right.append( folder + '/' + line[2].strip())
        center_images.extend(center)
        left_images.extend(left)
        right_images.extend(right)
        measurement_total.extend(measurements)
    return (center_images, left_images, right_images, measurement_total)   
            
def create_new(samples, batch_size=32):
    
    number_samples = len(samples)
    while 1:
        shf_samples = sklearn.utils.shuffle(samples)
        for offset in range(0, number_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []; measurements = []
            for image_path, measurement in batch_samples:
                origin_image = cv2.imread(image_path)
                image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
                images.append(image);             measurements.append(measurement)
                images.append(cv2.flip(image,1)); measurements.append(measurement*-1.0)
            inputs = np.array(images); outputs = np.array(measurements) # Trim the image
            yield sklearn.utils.shuffle(inputs, outputs)
                       
                 
def preprocessing_layer():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model
                       

from sklearn.model_selection import train_test_split

center_images, left_images, right_images, measurements = get_images('data')
image_paths, measurements = combine_images(center_images, left_images, right_images, measurements)
samples = list(zip(image_paths, measurements))
smpl_train, smpl_validation = train_test_split(samples, test_size=0.2)

print('Number of Images      = {}'.format(len(image_paths)))
print('Number of train smpls = {}'.format(len(smpl_train)))
print('Number of Valid. smpls= {}'.format(len(smpl_validation)))
                            
crt_train  = create_new(smpl_train)
crt_validation = create_new(smpl_validation)

print('Star nVidia model\n')               
model = nVidia()
print('Star compile\n')
model.compile(loss='mse', optimizer='adam')
# obj_histo = model.fit_generator(crt_train, samples_per_epoch=len(smpl_train), validation_data=crt_validation, nb_val_samples=len(smpl_validation), nb_epoch=3, verbose=1)
obj_histo = model.fit_generator(crt_train, steps_per_epoch=len(smpl_train), validation_data=crt_validation, validation_steps=len(smpl_validation), nb_epoch=3, verbose=1)
print('Saving the model ....\n')
model.save('model.h5')
print('Model saved')
print(obj_histo.history.keys())
print(obj_histo.history['loss']) #Loss
print(obj_histo.history['val_loss']) #Validation_loss


# Epoch 1/3
# [==============================] - loss: 0.0323 - val_loss: 0.0247
# Epoch 2/3
# [==============================] - loss: 0.0231 - val_loss: 0.0194
# Epoch 3/3
# [==============================] - loss: 0.0210 - val_loss: 0.0185
# Saving the model ....

# # Model saved
# dict_keys(['val_loss', 'loss'])
# [0.032302197645465083, 0.023199605755640847, 0.0210451334593958193] #Loss
# [0.024738030490593414, 0.01940256927326544, 0.01856277178770334] #Validation_loss

