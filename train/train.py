import json
import cv2
import tensorflow as tf
import os.path
import argparse
import shutil
from model.unet import Unet
from data.pascal_loader import PascalLoader
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from datetime import datetime
from utils import read_yaml_file, reload_sample_to_img


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str)
    parser.add_argument('--path_config', type=str)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--path_save_model', type=str, default=None)
    return parser.parse_args()


def load_dataset(path_dataset, config):
    print('[INFO]: LOADING TRAINING AND VALIDATION DATASETS...')
    train_info = config['train_info']

    if config['dataset'] == 'pascal':
        loader = PascalLoader(input_shape=train_info['input_shape'])
        train_dataset, val_dataset = loader.load_dataset(path_dataset=path_dataset,
                                                         batch_size=train_info['batch_size'],
                                                         train=True)
    batch_size = train_info['batch_size']
    print(f'[INFO] TRAINING DATASET SIZE = {tf.data.experimental.cardinality(train_dataset).numpy() * batch_size}')
    print(f'[INFO] VALIDATION DATASET SIZE = {tf.data.experimental.cardinality(val_dataset).numpy() * batch_size}')

    return train_dataset, val_dataset


def create_model(config):
    model_type = config['model']
    print(f'[INFO] CREATE NEW MODEL: {model_type}')
    if model_type == 'unet':
        return Unet(model_info=config['unet_info']).model
    return None


def create_callbacks(config, path_save_model):
    callbacks = [EarlyStopping(monitor='val_loss', patience=15)]

    if path_save_model is not None:
        callbacks.append(
            ModelCheckpoint(filepath=os.path.join(path_save_model, 'saved_model'),
                            save_best_only=True,
                            monitor='val_loss'))
    if config['tensorboard'] and path_save_model is not None:
        callbacks.append(
            TensorBoard(log_dir=os.path.join(path_save_model, 'logs'),
                        histogram_freq=1))
    if config['reduce_on_plateau']:
        callbacks.append(
            ReduceLROnPlateau(monitor='val_loss',
                              min_lr=0.00001,
                              factor=0.1,
                              patience=10,
                              mode='min'))
    return callbacks


def loss_fn(loss):
    if loss == 'categorical_crossentropy':
        return CategoricalCrossentropy()


def optimizer_fn(optimizer, lr):
    if optimizer == 'adam':
        return Adam(learning_rate=lr)
    if optimizer == 'rmsprop':
        return RMSprop(learning_rate=lr)
    if optimizer == 'sgd':
        return SGD(learning_rate=lr)


def compile_model(model, train_info):
    print('[INFO] COMPILING THE MODEL...')
    model.compile(optimizer=optimizer_fn(train_info['optimizer'], lr=train_info['learning_rate']),
                  loss=loss_fn(train_info['loss_fn']),
                  metrics=['accuracy'])
    return model


def preview_dataset(dataset):
    print('[INFO] SHOW DATASET SAMPLES...')
    label_color = json.load(open('../data/label_color.json'))
    for i, sample in enumerate(dataset):
        if i == 5:
            break
        # reload image
        img_x = reload_sample_to_img(sample=sample[0][0], label_color=None, target=False)
        cv2.imshow('x', img_x)
        cv2.waitKey(0)
        # reload target image
        img_y = reload_sample_to_img(sample=sample[1][0], label_color=label_color, target=True)
        cv2.imshow('y', img_y)
        cv2.waitKey(0)


def main():
    # read args
    args = read_args()
    # read config file
    config = read_yaml_file(args.path_config)
    # load datasets
    train_dataset, val_dataset = load_dataset(path_dataset=args.path_dataset,
                                              config=config)

    if args.show:  # show samples to check if the preprocessing is done correctly in the pipeline
        preview_dataset(train_dataset)

    # create model
    model = create_model(config=config)
    # compile model
    model = compile_model(model=model, train_info=config['train_info'])

    if args.path_save_model:
        # create folder where we save the model
        dir_path = os.path.join(args.path_save_model, datetime.now().strftime('%m_%d_%Y__%H_%M_%S'))
        os.mkdir(path=dir_path)
        # save config file
        shutil.copy(args.path_config, dir_path)
    else:
        dir_path = None

    # training
    print('[INFO] TRAINING STARTED...')
    model.fit(train_dataset,
              validation_data=val_dataset,
              epochs=config['train_info']['epochs'],
              verbose=2,
              callbacks=create_callbacks(config=config['train_info']['callbacks'], path_save_model=dir_path))
    print('[INFO] TRAINING OVER.')


if __name__ == '__main__':
    main()
