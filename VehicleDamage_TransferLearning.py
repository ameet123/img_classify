import collections
import json
import os
from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sn
from keras import applications, Model
from keras import backend as K
from keras import optimizers, losses, activations
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report


class CnnTransLearn:
    bottleneck_train_filename = "data/model/vgg/vgg16_btlnk__train.npy"
    bottleneck_val_filename = "data/model/vgg/vgg16_btlnk__val.npy"
    top_model_arch_weights_path = "data/model/vgg/vgg16_bottleneck_fc_model_arch_wt.h5"
    class_label_file = "data/model/vgg/vgg16_bottleneck_class_label_map.json"
    conf_matrix_heatmap_file = "data/model/vgg/conf_matrix_hm.png"
    classf_report_file = "data/model/vgg/classf_report.rpt"
    epochs = 30
    batch_size = 8  # 16

    def __init__(self, img_wd, img_ht, train_data_dir='data/training', validation_data_dir='data/validation'):
        print(">>Initializing variables")
        self.image_width = img_wd
        self.image_height = img_ht
        self.validation_data_dir = validation_data_dir
        self.train_data_dir = train_data_dir
        self.nb_train_samples: int = self.__file_count(self.train_data_dir)
        self.nb_validation_samples: int = self.__file_count(self.validation_data_dir)
        print(">>Train samples:{} Validation samples:{}".format(self.nb_train_samples, self.nb_validation_samples))
        self.model = None
        self.label_class_map = {"damage": 0, "whole": 1}
        if K.image_data_format() == 'channels_first':
            self.input_shape = (3, self.image_width, self.image_height)
        else:
            self.input_shape = (self.image_width, self.image_height, 3)
        print(">>K dataformat:{} input shape:{}".format(K.image_data_format(), self.input_shape))

    def __file_count(self, dir_name: str) -> int:
        x = 0
        for root, dirs, files in os.walk(dir_name):
            x = x + len(files)
        print("[{}]=> total files={}".format(os.path.basename(dir_name), x))
        return x

    def __count_by_class(self, dir_name: str):
        class_count_map: Dict[str, int] = {}
        for root, dirs, files in os.walk(dir_name):
            if len(files) == 0:
                continue
            class_name = str(os.path.basename(root))
            # print(">>{} = {}".format(class_name, len(files)))
            class_count_map[class_name] = len(files)
        return collections.OrderedDict(sorted(class_count_map.items()))

    def __make_train_val_labels(self, dir_name):
        class_count_map = self.__count_by_class(dir_name)
        labels = []
        for i, k in enumerate(class_count_map):
            labels = labels + [i] * class_count_map[k]
        return labels

    def __save_bottlebeck_features(self):
        print(">>Instantiating VGG16 models w/ data and training it with bottleneck")
        datagen = ImageDataGenerator(rescale=1. / 255)

        # build the VGG16 network

        model = applications.VGG16(include_top=False, weights='imagenet')

        generator = datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.image_width, self.image_height),
            batch_size=CnnTransLearn.batch_size,
            class_mode=None,
            shuffle=False)
        self.label_class_map = generator.class_indices
        with open(self.class_label_file, 'w') as fp:
            json.dump(generator.class_indices, fp)
        print(">>Training class label Map={}".format(generator.class_indices))
        bottleneck_features_train = model.predict_generator(
            generator, self.nb_train_samples // CnnTransLearn.batch_size)
        print(">>Bottleneck train shape:{}".format(bottleneck_features_train.shape))
        np.save(CnnTransLearn.bottleneck_train_filename, bottleneck_features_train)

        val_generator = datagen.flow_from_directory(
            self.validation_data_dir,
            target_size=(self.image_width, self.image_height),
            batch_size=CnnTransLearn.batch_size,
            class_mode=None,
            shuffle=False)
        bottleneck_features_validation = model.predict_generator(
            val_generator, self.nb_validation_samples // CnnTransLearn.batch_size)
        print(">>Bottleneck val shape:{}".format(bottleneck_features_validation.shape))
        np.save(CnnTransLearn.bottleneck_val_filename, bottleneck_features_validation)

    def __train_top_model(self):
        print(">>Train the VGG16 model with top Fully-Connected layer")
        train_data = np.load(self.bottleneck_train_filename)
        train_labels = np.array(self.__make_train_val_labels(self.train_data_dir))

        validation_data = np.load(self.bottleneck_val_filename)
        validation_labels = np.array(self.__make_train_val_labels(self.validation_data_dir))

        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation=activations.relu))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation=activations.sigmoid))

        model.compile(loss=losses.binary_crossentropy,
                      optimizer=optimizers.rmsprop(), metrics=['accuracy'])

        print(">>train: data={} labels={} Val: data={} labels={}".format(train_data.shape, train_labels.shape,
                                                                         validation_data.shape,
                                                                         validation_labels.shape))
        model.fit(train_data, train_labels,
                  epochs=CnnTransLearn.epochs,
                  batch_size=CnnTransLearn.batch_size,
                  validation_data=(validation_data, validation_labels))
        model.save(CnnTransLearn.top_model_arch_weights_path)
        self.model = model

    def train_bottleneck_top_model(self):
        self.__save_bottlebeck_features()
        self.__train_top_model()

    def fine_tuned_model(self):
        self.label_class_map = json.load(open(self.class_label_file, "r"))
        model = applications.VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        print('Model loaded.')

        # build a classifier model to put on top of the convolutional model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(1, activation='sigmoid'))

        # note that it is necessary to start with a fully-trained
        # classifier, including the top classifier,
        # in order to successfully do fine-tuning
        top_model.load_weights(CnnTransLearn.top_model_arch_weights_path)
        # add the model on top of the convolutional base
        model = Model(inputs=model.input, outputs=top_model(model.output))

        # set the first 25 layers (up to the last conv block)
        # to non-trainable (weights will not be updated)
        for layer in model.layers[:25]:
            layer.trainable = False

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])
        self.model = model

    def predict_singles(self, *images):
        # predicting images
        for i, img_file in enumerate(images):
            img = image.load_img(img_file, target_size=(self.image_width, self.image_height))
            x = image.img_to_array(img)
            x = x / 255.
            print("img array shape={}".format(x.shape))
            images = np.expand_dims(x, axis=0)
            # print("img expand dim shape={}".format(x.shape))
            # images = np.vstack([x])
            # print("img vstack shape={}".format(images.shape))
            # Since we are using Model class rather than Sequential, we can't do predict_classes
            probs = self.model.predict(images)
            pred_class = int(probs[0][0] >= .5)
            predicted_label = sorted(self.label_class_map.keys())[pred_class]
            print("\t>>[{}]: prob={} class={} class_name={}".format(i, probs, pred_class, predicted_label))

    def predict_batch(self, test_dir):
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.image_width, self.image_height),
            batch_size=CnnTransLearn.batch_size,
            class_mode=None,
            shuffle=False)  # This is necessary otherwise, it won't match with test_generator.classes labels.

        probabilities = self.model.predict_generator(test_generator)
        print(">>Prob:{}".format(probabilities))
        predictions = (probabilities >= 0.5).astype(int)
        print(">>Test classes:{}".format(test_generator.classes))

        print('Confusion Matrix')
        # The truth is picked from `classes` for each file picked from the directory matching class name
        cm = confusion_matrix(test_generator.classes, predictions)
        print(cm)
        print('Classification Report')
        labels = sorted(self.label_class_map.keys())
        cf_rpt = classification_report(test_generator.classes, predictions, target_names=labels)
        print(cf_rpt)
        with open(self.classf_report_file, "w") as cf_file:
            cf_file.write(cf_rpt)

        df_cm = pd.DataFrame(cm, labels, labels)
        hm = sn.heatmap(df_cm, annot=True)
        hm.get_figure().savefig(CnnTransLearn.conf_matrix_heatmap_file)


if __name__ == '__main__':
    cnn = CnnTransLearn(300, 300)

    cnn.fine_tuned_model()
    # cnn.predict_singles("data/validation/whole/0232.jpg", "data/validation/damage/0232.JPEG")
    cnn.predict_batch("data/test")
