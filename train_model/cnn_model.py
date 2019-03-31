__author__ = 'Victor Ruiz, ruizv1@email.chop.edu'
import pandas as pd
from os.path import join
from os import listdir
import tensorflow
from keras.backend import tf as ktf
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, MaxPooling2D
from keras import Sequential
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras import losses, optimizers
import multiprocessing as mp
import argparse
import pickle
import sys

def read_image(f_path, image_size):
    image_array = np.asarray(Image.open(f_path).resize((image_size, image_size)))
    return {'file': [f_path.split('/')[-1]], 'image': [image_array]}

def get_images_df(data_dir, label, image_size):
    pool = mp.Pool(processes=mp.cpu_count() - 1)
    f_list = []
    for filename in tqdm(listdir(data_dir), desc="loading {}".format(label)):
        f_path = join(data_dir, filename)
        f_list.append(pool.apply_async(read_image, args=(f_path, image_size)))
    results = []
    for i in tqdm(range(len(f_list)), desc='gathering {}'.format(label)):
        current_result = f_list[i].get()
        if current_result['image'][0].shape[-1] != 3:
            continue
        results += [current_result]
    out_df = pd.concat([pd.DataFrame(data=var) for var in results], axis=0, sort=True, ignore_index=True)
    out_df.loc[:, 'label'] = label
    return out_df

def add_channel_means(images_df):
    rgb_details = images_df.apply(lambda x: pd.Series(
        {'{}_mean'.format(k): v for k, v in zip(['Red', 'Green', 'Blue'], np.mean(x['image'], (0, 1)))}
    ), axis=1)
    grey_scale_vector = rgb_details.mean(axis=1)
    for col in rgb_details.columns:
        rgb_details.loc[:, col] = rgb_details.loc[:, col] / grey_scale_vector
    rgb_details.loc[:, 'Grey_mean'] = grey_scale_vector

    for col in rgb_details.columns:
        images_df[col] = rgb_details[col]
    return images_df

def normalize_images(a):
    mean = np.mean(a)
    std = np.std(a)
    normalized = (a - mean) / std
    return normalized, mean, std

def declare_net(image_size, class_cardinality):
    input_shape = (image_size, image_size, 3)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_cardinality, activation='softmax'))

    return model

def declare_net2(image_size, class_cardinality):
    cnn = Sequential()
    cnn.add(Conv2D(filters=32,
                   kernel_size=(2, 2),
                   strides=(1, 1),
                   padding='same',
                   input_shape=(image_size, image_size, 3),
                   data_format='channels_last'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2),
                         strides=2))
    cnn.add(Conv2D(filters=64,
                   kernel_size=(2, 2),
                   strides=(1, 1),
                   padding='valid'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2),
                         strides=2))
    cnn.add(Flatten())
    cnn.add(Dense(64))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.25))
    cnn.add(Dense(class_cardinality))
    cnn.add(Activation('softmax'))
    return cnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help="Directory containing images with sufolders case and control")
    parser.add_argument('--out_dir', required=True, help="directory where model will be saved")
    options = parser.parse_args()

    data_dir = options.data_dir
    out_dir = options.out_dir


    ### load dataset and convert to arrays
    image_size = 224
    case_df = get_images_df(join(data_dir, 'case'), 'malignant', image_size)
    control_df = get_images_df(join(data_dir, 'control'), 'benign', image_size)

    dataset = pd.concat([case_df, control_df], axis=0, ignore_index=True, sort=False)
    dataset = add_channel_means(dataset)

    ### train/test split
    class_labels = pd.get_dummies(dataset.loc[:, ['label']]).values
    X_train, X_test, y_train, y_test = train_test_split(dataset.image.tolist(), class_labels, test_size=0.25)

    ### normalize

    X_hat_train, mean, std = normalize_images(X_train)
    X_hat_test, _, _ = normalize_images(X_test)

    with open(join(out_dir, 'mean.pkl'), 'wb') as out_f:  # it's just a float, but will be useful later
        pickle.dump(mean, out_f)
    with open(join(out_dir, 'std.pkl'), 'wb') as out_f:
        pickle.dump(std, out_f)

    ### get model
    model = declare_net(image_size, 2)
    model.compile(loss=losses.binary_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])

    batch_size = 100
    epochs = 100

    history = model.fit(X_hat_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.2,
                        shuffle=True)
    predictions = model.predict(np.asarray(X_test))
    with open(join(out_dir, 'predictions.pkl'), 'wb') as out_f:
        pickle.dump(predictions, out_f)
    score = model.evaluate(X_hat_test, y_test, verbose=0)
    model.save(join(out_dir, 'model.hd5'))
    with open(join(out_dir, 'results.txt'), 'w') as f:
        f.write('Test loss: {}\n'.format(str(score[0])))
        f.write('Test accuracy: {}\n'.format(str(score[1])))


if __name__ == '__main__':
    main() 