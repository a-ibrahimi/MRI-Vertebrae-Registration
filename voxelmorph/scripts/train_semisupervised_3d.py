import argparse
import datetime

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from generators import *

from voxelmorph.tf.networks import VxmDense, VxmDenseSemiSupervisedSeg
import voxelmorph as vxm

import configparser

if __name__ == '__main__':
    
    config_path = 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
    
    training_params = config['Semi3DTrainingParameters']
    
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('--slice_number', type=int, default=int(training_params['slice_number']), help='Number for generate_2d_slices function')
    parser.add_argument('--batch_size', type=int, default=int(training_params['batch_size']), help='Batch size for data generators')
    parser.add_argument('--test_size_1', type=float, default=float(training_params['test_size_1']), help='First test size for splitting data')
    parser.add_argument('--test_size_2', type=float, default=float(training_params['test_size_2']), help='Second test size for splitting data')
    parser.add_argument('--int_steps', type=int, default=int(training_params['int_steps']), help='Integration steps for VxmDense model')
    parser.add_argument('--lambda_param', type=float, default=float(training_params['lambda_param']), help='Lambda parameter for loss weights')
    parser.add_argument('--steps_per_epoch', type=int, default=int(training_params['steps_per_epoch']), help='Steps per epoch during training')
    parser.add_argument('--nb_epochs', type=int, default=int(training_params['nb_epochs']), help='Number of epochs for training')
    parser.add_argument('--verbose', type=int, default=int(training_params['verbose']), help='Verbose mode')
    parser.add_argument('--weights_path', type=str, default=training_params['weights_path'], help='Path to save model weights')
    parser.add_argument('--loss', type=str, default=training_params['loss'], help='Type of loss function')
    parser.add_argument('--grad_norm_type', type=str, choices=['l1', 'l2'], default=training_params['grad_norm_type'], help='Type of norm for Grad loss (l1 or l2)')
    parser.add_argument('--batch_number', type=int, default=int(training_params['batch_number']), help='')
    parser.add_argument('--gamma_param', type=float, default=float(training_params['gamma_param']), help='weight of dice loss (gamma) (default: 0.01)')
    parser.add_argument('--learning_rate', type=float, default=float(training_params['learning_rate']), help='Learning rate (default: 0.0001)')
    parser.add_argument('--images_path', type=str, default=training_params['images_path'], help='Path to npy file containing the MRI scans as numpy array')
    parser.add_argument('--segmentations_path', type=str, default=training_params['segmentations_path'], help='Path to npy file containing the segmentation masks as numpy array')
    parser.add_argument('--labels', type=str, default=training_params['labels'], help='label list (npy format) to use in dice loss')
    parser.add_argument('--patience', type=int, default=int(training_params['patience']), help='Number of epochs with no improvement in validation performance after which training will be stopped.')


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
    
    # split data
    x_train, x_other = train_test_split(images, test_size=args.test_size_1, random_state=42)
    x_test, x_val = train_test_split(x_other, test_size=args.test_size_2, random_state=42)
    
    segmentations_train, segmentation_other = train_test_split(seg, test_size=args.test_size_1, random_state=42)
    segmentations_test, segmentations_val = train_test_split(segmentation_other, test_size=args.test_size_2, random_state=42)

    labels = np.load(args.labels)

    train_gen = semisupervised(x_train, segmentations_train, labels=labels, batch_size=args.batch_size)
    val_gen = semisupervised(x_val, segmentations_val, labels=labels,batch_size=args.batch_size)
    test_gen = semisupervised(x_test, segmentations_test, labels=labels,batch_size=args.batch_size)
    
    # create model
    nf_enc=[16, 32, 64, 128]
    nf_dec=[128, 64, 64, 32, 32, 16, 16]
    
    inshape = next(train_gen)[0][0].shape[1:-1]
    
    vxm_model = VxmDenseSemiSupervisedSeg(inshape=inshape, nb_labels=len(labels), seg_resolution=1, int_steps=args.int_steps, nb_unet_features=[nf_enc, nf_dec])
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
        log_dir = "voxelmorph/tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience = args.patience, restore_best_weights=True)
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
