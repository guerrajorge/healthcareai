__author__ = 'Victor Ruiz, ruizv1@email.chop.edu'
import keras
import numpy as np
import argparse
from PIL import Image

def load_input_image(file_path, mean, std, image_size=224):
    image = read_image(file_path, image_size)
    return (image - mean) / std

def read_image(f_path, image_size):
    image_array = np.asarray(Image.open(f_path).resize((image_size, image_size)))
    return image_array

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path')
    parser.add_argument('--model_path')
    parser.add_argument('--out_path')
    options = parser.parse_args()


    mean = 153.33241497580914  # from training dataset
    std = 46.89837331344314

    image = load_input_image(options.image_path, mean, std)

    model = keras.models.load_model(options.model_path)

    prediction = model.predict(np.array([image])).reshape(-1)

    prediction = 'benign' if np.argmax(prediction) == 1 else 'malignant'

    with open(options.out_path, 'w') as out_f:
        out_f.write(prediction)


if __name__ == '__main__':
    main()