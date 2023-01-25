import json
import os.path
import tensorflow as tf


class PascalLoader:

    def __init__(self, input_shape):
        # self.label_color = self.__create_labels_color_dict()  # BGR format
        self.label_color = json.load(open('../data/label_color.json'))
        self.input_shape = eval(input_shape)
        self.colors = tf.cast([list(reversed(eval(color))) for color in list(self.label_color.values())], dtype='uint8')

    def __create_labels_color_dict(self):
        label_color, new_label_color = json.load(open('../data/label_color.json')), {}
        labels_to_keep = ["background", "void", "person", "bicycle", "car", "dog", "horse", "boat"]

        for label in labels_to_keep:
            new_label_color[label] = label_color[label]
        with open("./new_label_color.json", 'w') as file_output:
            json.dump(new_label_color, file_output)
        return new_label_color

    def __get_img_paths(self, path_dataset, split):
        path = os.path.join(path_dataset, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', split + '.txt')
        img_names = open(path, 'r').readlines()

        path = os.path.join(path_dataset, 'VOCdevkit', 'VOC2012')
        img_paths = [os.path.join(path, 'JPEGImages', str(img_name + '.jpg').replace('\n', '')) for img_name in
                     img_names]
        target_paths = [os.path.join(path, 'SegmentationClass', str(img_name + '.png').replace('\n', '')) for img_name
                        in img_names]

        return img_paths, target_paths

    @tf.function
    def __load_image(self, img_path, target_img_path):
        # load + resize 'img_path'
        x = tf.io.read_file(img_path)
        x = tf.io.decode_jpeg(x, channels=3)
        x = tf.image.resize(x, size=tf.cast(self.input_shape[:2], dtype='int32'))
        x = tf.cast(x, dtype='float32') / 255.0

        # load + resize + one-hot encoding pixel-wise
        y = tf.io.read_file(target_img_path)
        y = tf.io.decode_png(y, channels=3, dtype='uint8')
        y = tf.image.resize(y, size=self.input_shape[:2], method='nearest')
        y = tf.reduce_all(tf.expand_dims(y, axis=-1) == tf.transpose(self.colors), axis=-2)
        y = tf.cast(tf.where(y, 1, 0), dtype='float32')

        return x, y

    def __load_tf_dataset(self, dataset, batch_size, train):
        tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)

        if train:
            tf_dataset = tf_dataset.map(self.__load_image, num_parallel_calls=tf.data.AUTOTUNE)  # (train+val) datasets
        else:
            pass  # test dataset
        tf_dataset = tf_dataset \
            .shuffle(3000) \
            .cache() \
            .batch(batch_size) \
            .prefetch(tf.data.AUTOTUNE)
        return tf_dataset

    def load_dataset(self, path_dataset, batch_size, train):
        train_img_paths, train_target_paths = self.__get_img_paths(path_dataset=path_dataset, split='train')
        val_img_paths, val_target_paths = self.__get_img_paths(path_dataset=path_dataset, split='val')

        tf_train_dataset = self.__load_tf_dataset(dataset=(train_img_paths, train_target_paths),
                                                  batch_size=batch_size,
                                                  train=train)
        tf_val_dataset = self.__load_tf_dataset(dataset=(val_img_paths, val_target_paths),
                                                batch_size=batch_size,
                                                train=train)

        return tf_train_dataset, tf_val_dataset
