import cv2
import csv
import numpy as np

def get_line_from_csv(data_path):
    
    first_row = True
    lines = []
    with open(data_path + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        if first_row:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines

def get_image_and_measurement(data_path, image_path, measurement, images, measurements):

#     print(image_path.strip())
#     exit()
#     all_images = cv2.imread(data_path + '/' + image_path.strip())
    all_images = cv2.imread(image_path.strip())
    rgb_images = cv2.cvtColor(all_images, cv2.COLOR_BGR2RGB)
    images.append(rgb_images)
    measurements.append(measurement)
    images.append(cv2.flip(rgb_images,1))
    measurements.append(measurement*-1.0)

def get_images_and_measurements(data_path):
    
    lines = get_line_from_csv(data_path)
    images = []
    measurements = []
    for line in lines:
        measurement = float(line[3])
        #Central Camera
        get_image_and_measurement(data_path, line[0], measurement, images, measurements)
        #Left Camera
        get_image_and_measurement(data_path, line[1], measurement, images, measurements)
        #Right Camera
        get_image_and_measurement(data_path, line[2], measurement, images, measurements)
    return (np.array(images), np.array(measurements))


from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D

def leNet_model():

    model = preprocessing_layer()
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def nVidia_model():
   
    model = preprocessing_layer()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def preprocessing_layer():
    
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

def simple_model():
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    return model

def train_and_save(model, xtrain, ytrain, model_name, epochs=2):
    model.compile(loss='mse', optimizer='adam')
    model.fit(xtrain, ytrain, validation_split=0.2, shuffle=True, nb_epoch=epochs)
    model.save(model_name)
    print("Model trained and saved!")    
    
print('Start read data . . .')
X_train, y_train = get_images_and_measurements('data/')
print('finish reading data')
print('Star simple model')
model = simple_model()
print('Start train and save model . . .')
train_and_save(model, X_train, y_train, 'models/model.h5')
print('\n')
print('Star lenet model')
model = leNet_model()
print('Start training . . .')
train_and_save(model, X_train, y_train, 'models/lenNet_model.h5')
print('\n')
print('Star nvidia model')
model = nVidia_model
print('Start training . . .')
train_and_save(model, X_train, y_train, 'models/nVidia_model.h5')






