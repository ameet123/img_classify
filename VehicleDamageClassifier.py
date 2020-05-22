import json
import os

import numpy as np
from PIL import Image
from keras import backend as K, Model
from keras import optimizers, losses, activations
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, model_from_yaml, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


class Cnn(Model):
    epochs = 30
    batch_size = 8
    IMG_SZ = 300

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(">>Initializing variables")
        self.model_name = "vehicle_cnn"
        self.model = None
        top_dir = "data"
        self.train_dir = top_dir + "/training"
        self.val_dir = top_dir + "/validation"
        self.test_dir = top_dir + "/test/mix"
        self.model_dir = top_dir + "/model"
        self.model_yaml_file = os.path.join(self.model_dir, self.model_name + ".yml")
        self.model_wt_file = os.path.join(self.model_dir, self.model_name + "_wt.h5")
        self.model_arch_wt_file = os.path.join(self.model_dir, self.model_name + "_arch_wt.h5")
        # Mapping of labels to class
        self.label_class_map = {"damage":0, "whole":1}
        self.class_label_file = os.path.join(self.model_dir, self.model_name + "_class_label_map.json")

    def build_train_model(self, img_size=300, train_dir=None, val_dir=None, model_dir=None):
        print(">>Building a model")

        if train_dir is not None:
            self.train_dir = train_dir
        if val_dir is not None:
            self.val_dir = val_dir
        if model_dir is not None:
            self.model_dir = model_dir
        if img_size is not None:
            Cnn.IMG_SZ = img_size

        if K.image_data_format() == 'channels_first':
            input_shape = (3, Cnn.IMG_SZ, Cnn.IMG_SZ)
        else:
            input_shape = (Cnn.IMG_SZ, Cnn.IMG_SZ, 3)

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        self.model.add(Activation(activations.relu))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation(activations.relu))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation(activations.relu))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation(activations.relu))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation(activations.sigmoid))

        self.model.compile(loss=losses.binary_crossentropy,
                           optimizer=optimizers.rmsprop(),
                           metrics=['accuracy'])
        model_yaml = self.model.to_yaml()
        with open(self.model_yaml_file, "w") as yaml_file:
            yaml_file.write(model_yaml)

        print(">>Training model with train and validation data")
        # this is the augmentation configuration we will use for training
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(Cnn.IMG_SZ, Cnn.IMG_SZ),
            batch_size=Cnn.batch_size,
            class_mode='binary')
        self.label_class_map = train_generator.class_indices
        with open(self.class_label_file, 'w') as fp:
            json.dump(self.label_class_map, fp)
        print(">>Training class label Map={}".format(train_generator.class_indices))

        validation_generator = test_datagen.flow_from_directory(
            self.val_dir,
            target_size=(Cnn.IMG_SZ, Cnn.IMG_SZ),
            batch_size=Cnn.batch_size,
            class_mode='binary')

        nb_train_samples = self.file_count(self.train_dir)
        nb_validation_samples = self.file_count(self.val_dir)
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // Cnn.batch_size,
            epochs=Cnn.epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // Cnn.batch_size)

        self.model.save(self.model_arch_wt_file)
        self.model.save_weights(self.model_wt_file)

    def file_count(self, dir_name):
        x = 0
        for root, dirs, files in os.walk(dir_name):
            x = x + len(files)
        print("total files={}".format(x))
        return x

    def fetch_model(self, model_file=None):
        if model_file is None:
            model_file = self.model_arch_wt_file
        print(">>Loading model from file:{}".format(model_file))
        self.model = load_model(model_file)
        return self

    def from_yaml(self):
        yaml_file = open(self.model_yaml_file, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(loaded_model_yaml)
        model.summary()
        return model

    def predict_singles(self, *images):
        # predicting images
        for img_file in images:
            img = image.load_img(img_file, target_size=(Cnn.IMG_SZ, Cnn.IMG_SZ))
            x = image.img_to_array(img)
            x = x / 255.
            print("img array shape={}".format(x.shape))
            x = np.expand_dims(x, axis=0)
            print("img expand dim shape={}".format(x.shape))
            images = np.vstack([x])
            print("img vstack shape={}".format(images.shape))
            prob = self.model.predict(images)
            classes = self.model.predict_classes(images, batch_size=10)
            predicted_label = sorted(self.label_class_map.keys())[classes[0][0]]
            print("\t>>prob={} class={} label={}".format(prob, classes, predicted_label))

    def predict_batch(self, test_dir):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(Cnn.IMG_SZ, Cnn.IMG_SZ),
            batch_size=Cnn.batch_size,
            class_mode=None)
        probabilities = self.model.predict_generator(test_generator)
        print(probabilities)


def get_size_statistics(image_path):
    for root, d_names, f_names in os.walk(image_path):
        if len(f_names) == 0 or ".jp" not in f_names[0].lower():
            continue
        print("{}:[{}]\n{}\n".format(root, len(f_names), "=" * 50))

        heights = []
        widths = []
        for img in os.listdir(root):
            path = os.path.join(root, img)
            data = np.array(Image.open(path))
            heights.append(data.shape[0])
            widths.append(data.shape[1])
        avg_height = sum(heights) / len(heights)
        avg_width = sum(widths) / len(widths)
        print("\tHeight: avg = {} max = {} min = {}\n\tWidth:  avg = {} max = {} min = {}\n".
              format(int(avg_height), max(heights), min(heights), int(avg_width), max(widths), min(widths)))


if __name__ == '__main__':
    # get_size_statistics(TOP_DIR)
    cnn = Cnn()
    cnn.fetch_model()
    # cnn.build_train_model(TRAIN_DIR, VAL_DIR, MODEL_DIR, 300)

    cnn.predict_singles("data/test/mix/damage/0001-dam.JPEG")
    # predict_batch(model)
