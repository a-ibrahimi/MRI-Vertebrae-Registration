"""
Script: train_unsupervised_affine_2d.py

Description:
This script trains a unsupervised 2D image segmentation model using Voxelmorph and affine transformations.
It utilizes command line arguments and configurations from the 'config.ini' file.

Dependencies:
- argparse
- datetime
- voxelmorph
- tensorflow
- numpy
- sklearn
- keras
- configparser
- sys
- os

Usage:
Run this script to train a unsupervised 2D image segmentation model using affine transformations with specified
parameters. The model is based on Voxelmorph and uses TensorFlow as the backend.
Configuration details are loaded from the 'config.ini' file.
"""

import argparse
import datetime

import voxelmorph as vxm
from voxelmorph.tf.networks import VxmDense

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

import configparser

import sys
import os

sys.path.append(os.getcwd())

import vxlmorph.generators as generators

if __name__ == '__main__':

    config_path = 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)

    training_params = config['Semi2DTrainingParameters']

    # Parsing command line arguments
    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('--slice_number', type=int, default=training_params['slice_number'], help='Number for generate_2d_slices function')
    parser.add_argument('--batch_size', type=int, default=training_params['batch_size'], help='Batch size for data generators')
    parser.add_argument('--test_size_1', type=float, default=training_params['test_size_1'], help='First test size for splitting data')
    parser.add_argument('--test_size_2', type=float, default=training_params['test_size_2'], help='Second test size for splitting data')
    parser.add_argument('--int_steps', type=int, default=training_params['int_steps'], help='Integration steps for VxmDense model')
    parser.add_argument('--lambda_param', type=float, default=training_params['lambda_param'], help='Lambda parameter for loss weights')
    parser.add_argument('--steps_per_epoch', type=int, default=training_params['steps_per_epoch'], help='Steps per epoch during training')
    parser.add_argument('--nb_epochs', type=int, default=training_params['nb_epochs'], help='Number of epochs for training')
    parser.add_argument('--verbose', type=int, default=training_params['verbose'], help='Verbose mode')
    parser.add_argument('--loss', type=str, default=training_params['loss'], help='Type of loss function')
    parser.add_argument('--grad_norm_type', type=str, choices=['l1', 'l2'], default=training_params['grad_norm_type'], help='Type of norm for Grad loss (l1 or l2)')
    parser.add_argument('--batch_number', type=int, default=training_params['batch_number'], help='')
    parser.add_argument('--gamma_param', type=float, default=training_params['gamma_param'], help='weight of dice loss (gamma) (default: 0.02)')
    parser.add_argument('--learning_rate', type=float, default=training_params['learning_rate'], help='Learning rate (default: 0.0001)')
    parser.add_argument('--images_path', type=str, default=training_params['images_path'], help='Path to npy file containing the MRI scans as numpy array')
    parser.add_argument('--segmentations_path', type=str, default=training_params['segmentations_path'], help='Path to npy file containing the segmentation masks as numpy array')
    parser.add_argument('--weights_path', type=str, default=training_params['weights_path'], help='Path to save model weights')

    parser.add_argument('--patience', type=int, default=training_params['patience'], help='Number of epochs with no improvement in validation performance after which training will be stopped.')


    args = parser.parse_args()

    # Specify GPU device and allow memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f" {len(physical_devices)} GPU(s) is/are available")
    else:
        print("No GPU detected")

    all_images = np.load(args.images_path)
    all_seg = np.load(args.segmentations_path)

    images = all_images[:args.slice_number]
    seg = all_seg[:args.slice_number]

    images = (images - np.min(images)) / (np.max(images) - np.min(images))

    x_train, x_other = train_test_split(images, test_size=args.test_size_1, random_state=42)
    x_test, x_val = train_test_split(x_other, test_size=args.test_size_2, random_state=42)

    segmentations_train, segmentation_other = train_test_split(seg, test_size=args.test_size_1, random_state=42)
    segmentations_test, segmentations_val = train_test_split(segmentation_other, test_size=args.test_size_2, random_state=42)

    labels = np.load(args.labels)

    x_train, x_other, seg_train, seg_other = train_test_split(images, seg, test_size=0.2, random_state=42)
    x_val, x_test = train_test_split(x_other, test_size=0.1 / 0.2, random_state=42)
    seg_val, seg_test = train_test_split(seg_other, test_size=0.1 / 0.2, random_state=42)

    labels = np.load(args.labels)

    train_gen = generators.vxm_data_generator_affine(x_train, seg_train, batch_size=args.batch_size)
    val_gen = generators.vxm_data_generator_affine(x_val, seg_val, batch_size=args.batch_size)
    test_gen = generators.vxm_data_generator_affine(x_test, seg_test, batch_size=args.batch_size)

    nf_enc=[14, 28, 144, 320]
    nf_dec=[1152, 1152, 320, 144, 28, 14, 14]

    inshape = next(train_gen)[0][0].shape[1:-1]
    vxm_model = VxmDense(inshape=inshape, nb_unet_features=[nf_enc, nf_dec], int_steps=args.int_steps)
    # define lossls
    if args.loss == 'MSE':
        loss_func = vxm.losses.MSE().loss
    elif args.loss == 'NCC':
        loss_func = vxm.losses.NCC().loss
    elif args.loss == 'MI':
        loss_func = vxm.losses.MutualInformation().loss
    elif args.loss == 'TukeyBiweight':
        loss_func = vxm.losses.TukeyBiweight().loss
    else:
        loss_func = vxm.losses.MSE().loss
    # define grad
    if args.grad_norm_type == 'l1':
        grad_norm = 'l1'
    elif args.grad_norm_type == 'l2':
        grad_norm = 'l2'
    else:
        grad_norm = 'l2'
    # Add more elif blocks here for other loss types if needed
    losses = [loss_func, vxm.losses.Grad(grad_norm).loss, vxm.losses.Dice().loss]
    loss_weights = [1, args.lambda_param, args.gamma_param]

    # compile model
    print('Compiling model...')
    with tf.device('/GPU:0'):
        log_dir = "vxlmorph/tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience = 60, restore_best_weights=True)
        vxm_model.compile(tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=losses, loss_weights=loss_weights)
        # train and validate model
        print(f'Training model with hyperparams: Loss: {args.loss}, Lambda: {args.lambda_param}, Gamma: {args.gamma_param}, Learning rate: {args.learning_rate}')
        vxm_model.fit(train_gen, steps_per_epoch=args.steps_per_epoch, epochs=args.nb_epochs, validation_data=val_gen, validation_steps=args.steps_per_epoch, verbose=args.verbose, callbacks=[tensorboard_callback, early_stopping])
        # save model
        print('Saving model...')
        vxm_model.save_weights(args.weights_path)
        # evaluate or test model
        print('Evaluating model...')
        vxm_model.evaluate(test_gen, steps=args.steps_per_epoch, verbose=args.verbose)
        # predict model and calculate the dice score between the predicted and ground truth images
        print('Predicting model...')
        dice_scores = []
        for i in range(args.batch_number):
            test_input, _ = next(test_gen)

            #Save input as npy
            np.save(f'vxlmorph/tensorboard/Semisupervised/Evaluation_input/Moving/hyper1{args.loss}_{args.gamma_param}_{args.lambda_param}_{args.learning_rate}_{i}.npy', test_input[0])
            np.save(f'vxlmorph/tensorboard/Semisupervised/Evaluation_input/Fixed/hyper1{args.loss}_{args.gamma_param}_{args.lambda_param}_{args.learning_rate}_{i}.npy', test_input[1])
            test_pred = vxm_model.predict(test_input, verbose=args.verbose)
            
            #Save output as npy
            np.save(f'vxlmorph/tensorboard/Semisupervised/Evaluation_output/Moved/hyper1{args.loss}_{args.gamma_param}_{args.lambda_param}_{args.learning_rate}_{i}.npy', test_pred[0])
            np.save(f'vxlmorph/tensorboard/Semisupervised/Evaluation_output/Field/hyper1{args.loss}_{args.gamma_param}_{args.lambda_param}_{args.learning_rate}_{i}.npy', test_pred[1])
            
            test_input = tf.convert_to_tensor(test_input[1], dtype=tf.float32)
            test_pred = tf.convert_to_tensor(test_pred[0], dtype=tf.float32)
            dice = vxm.losses.Dice().loss(test_input, test_pred)
            dice_scores.append(dice)
        average_dice_score = np.mean(dice_scores)
        print('Average dice score: ', average_dice_score)

        print(f'Model hyperparams: Loss: {args.loss}, Lambda: {args.lambda_param}, Gamma: {args.gamma_param}, Learning rate: {args.learning_rate} - Average dice score: {average_dice_score}')
        np.save(f'vxlmorph/tensorboard/Semisupervised/Metrics/Dice_hyper1{args.loss}_{args.gamma_param}_{args.lambda_param}_{args.learning_rate}.npy', np.array(dice_scores))
        
        print('\n---------------------------------------------------------------------------------------------------------\n')


