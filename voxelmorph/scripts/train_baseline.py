import os
import argparse

import nibabel as nib
import neurite as ne
import voxelmorph as vxm
from voxelmorph.tf.networks import VxmDense

import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import BIDS
from BIDS import *

import configparser

def get_orientation(img):
    orientation = nib.aff2axcodes(img.affine)
    return orientation

def generate_2d_slices(num):
    slices = []
    data_dir = "/local_ssd/practical_wise24/vertebra_labeling/data/"
    i=0
    for subject_dir in os.listdir(os.path.join(data_dir, 'spider_raw')):
        subject_path = os.path.join(data_dir, 'spider_raw', subject_dir, 'T1w')
        
        if os.path.exists(subject_path):
            nii_files = [file for file in os.listdir(subject_path) if file.endswith('.nii.gz')]
            
            if len(nii_files) > 0:

                img = nib.load(os.path.join(subject_path, nii_files[0]))
                nii_img = BIDS.NII(img)
                nii_img.rescale_((1,1,1), verbose=False) 
                data = nii_img.get_array()

                orientation = get_orientation(img)
                        
                if orientation == ('L', 'P', 'S'):
                    central_axis = 0  
                elif orientation == ('P', 'I', 'R'):
                    central_axis = 2 

                central_slice_index = data.shape[central_axis] // 2 

                if central_axis == 0:
                    central_slice = data[central_slice_index, :, :]
                else:
                    central_slice = data[:, :, central_slice_index]
                
                slices.append(central_slice)
            i+=1
        if i==num:
            break
    return slices

def resample(slice_shape, slices):

    final_shape = (len(slices), slice_shape[0], slice_shape[1])
    final_array = np.empty(final_shape)

    for i, array in enumerate(slices):
        normalized_array = (array - array.min()) / (array.max() - array.min())

        row_factor = final_shape[1] / normalized_array.shape[0]
        col_factor = final_shape[2] / normalized_array.shape[1]

        resampled_array = zoom(normalized_array, (row_factor, col_factor), order=1)

        final_array[i] = resampled_array[:final_shape[1], :final_shape[2]]

    return final_array

def vxm_data_generator(x_data, batch_size=16):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """
    vol_shape = x_data.shape[1:]
    ndims = len(vol_shape)
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        outputs = [fixed_images, zero_phi]
        
        yield (inputs, outputs)

def plot_history(hist, loss_name=['loss', 'val_loss']):
    """
    Simple function to plot training history
    """
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name[0]], label='Training Loss', marker='o', linestyle='-')
    plt.plot(hist.epoch, hist.history[loss_name[1]], label='Validation Loss', marker='o', linestyle='-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    plt.savefig('/u/home/qug/practice/plot/loss.png')

if __name__ == '__main__':

    config_path = 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)

    training_params = config['2DBaselineTrainingParameters']

    parser = argparse.ArgumentParser(description='Process model parameters.')
    parser.add_argument('--slice_number', type=int, default=training_params['slice_number'], help='Number for generate_2d_slices function')
    parser.add_argument('--batch_size', type=int, default=training_params['batch_size'], help='Batch size for data generators')
    parser.add_argument('--test_size_1', type=float, default=training_params['test_size_1'], help='First test size for splitting data')
    parser.add_argument('--test_size_2', type=float, default=training_params['test_size_2'], help='Second test size for splitting data')
    parser.add_argument('--int_steps', type=int, default=training_params['int_steps'], help='Integration steps for VxmDense model')
    parser.add_argument('--scaled_up', type=bool, default=training_params['scaled_up'], help='Scaled up model')
    parser.add_argument('--lambda_param', type=float, default=training_params['lambda_param'], help='Lambda parameter for loss weights')
    parser.add_argument('--steps_per_epoch', type=int, default=training_params['steps_per_epoch'], help='Steps per epoch during training')
    parser.add_argument('--nb_epochs', type=int, default=training_params['nb_epochs'], help='Number of epochs for training')
    parser.add_argument('--verbose', type=int, default=training_params['verbose'], help='Verbose mode')
    parser.add_argument('--weights_path', type=str, default=training_params['weights_path'], help='Path to save model weights')
    parser.add_argument('--loss', type=str, default=training_params['loss'], help='Type of loss function')
    parser.add_argument('--grad_norm_type', type=str, choices=training_params['grad_norm_type'], default='l2', help='Type of norm for Grad loss (l1 or l2)')
    parser.add_argument('--batch_number', type=int, default=training_params['batch_number'], help='Example number for prediction to calculate dice score')
    parser.add_argument('--demo_path', type=str, default=training_params['demo_path'], help='Path to save demo images')
    
    args = parser.parse_args()

    slices = generate_2d_slices(args.slice_number)
    resampled_slices = resample((256, 256), slices)

    x_train, x_other = train_test_split(resampled_slices, test_size=args.test_size_1, random_state=42)
    x_test, x_val = train_test_split(x_other, test_size=args.test_size_2, random_state=42)

    train_gen = vxm_data_generator(x_train, batch_size=args.batch_size)
    val_gen = vxm_data_generator(x_val, batch_size=args.batch_size)
    test_gen = vxm_data_generator(x_test, batch_size=args.batch_size)

    if args.scaled_up == False:
        nf_enc=[16,32,32,32]
        nf_dec=[32,32,32,32,32,16,16,3]
    else:
        nf_enc=[14, 28, 144, 320]
        nf_dec=[1152, 1152, 320, 144, 28, 14, 14]
    nb_features = [nf_enc, nf_dec]
    inshape = resampled_slices.shape[1:]
    vxm_model = VxmDense(inshape, nb_features, int_steps=args.int_steps)

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

    if args.grad_norm_type == 'l1':
        grad_norm = 'l1'
    elif args.grad_norm_type == 'l2':
        grad_norm = 'l2'
    else:
        grad_norm = 'l2'

    losses = [loss_func, vxm.losses.Grad(grad_norm).loss]
    loss_weights = [1, args.lambda_param]

    print('Compiling model...')
    vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

    print('Training model...')
    vxm_model.fit(train_gen, steps_per_epoch=args.steps_per_epoch, epochs=args.nb_epochs, validation_data=val_gen, validation_steps=args.steps_per_epoch, verbose=args.verbose)

    print('Plotting history...')
    plot_history(vxm_model.history)

    print('Saving model...')
    vxm_model.save_weights(args.weights_path)

    print('Evaluating model...')
    vxm_model.evaluate(test_gen, steps=args.steps_per_epoch, verbose=args.verbose)

    print('Predicting model...')
    dice_scores = []
    for i in range(args.batch_number):
        test_input, _ = next(test_gen)
        test_pred = vxm_model.predict(test_input, verbose=args.verbose)
        test_input = tf.convert_to_tensor(test_input[1], dtype=tf.float32)
        test_pred = tf.convert_to_tensor(test_pred[0], dtype=tf.float32)
        dice = vxm.losses.Dice().loss(test_input, test_pred)
        dice_scores.append(dice)
    average_dice_score = np.mean(dice_scores)
    print('Average dice score: ', average_dice_score)

    print('Demonstrating on a sample image...')
    for i in range(10):
        test_input, _ = next(test_gen)
        test_pred = vxm_model.predict(test_input, verbose=args.verbose)
        images = [img[0, :, :, 0] for img in test_input + test_pred] 
        titles = ['moving', 'fixed', 'moved', 'flow']
        ne.plot.slices(images, titles=titles, cmaps=['gray'], do_colorbars=True)
        plt.savefig(args.demo_path +str(i)+'.png')