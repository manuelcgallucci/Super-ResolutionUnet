import cv2
from skimage import io
import matplotlib.pyplot as plt
from utility import *
import numpy as np
import os
from torchvision.transforms.functional import resize
import torchvision.transforms as T
import torch


from PIL import Image
import glob
import re 

# Histogram paramters
def save_histogram(x, out_path, title, n_bins=20):
    plt.figure()
    counts, bins = np.histogram(x, bins=n_bins)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.title(title)
    plt.savefig(out_path)
    plt.close()

def save_double_histograms(x, y, labelx, labely, out_path, title, n_bins=20):

    max_ = max([np.amax(x), np.amax(y)])
    min_ = min([np.amin(x), np.amin(y)])
    range_ = tuple((min_, max_))#np.linspace(, n_bins)
    plt.figure()
    counts, bins = np.histogram(x, bins=n_bins, range=range_)
    counts = counts / np.sum(counts)
    plt.hist(bins[:-1], bins, weights=counts, alpha=0.5, label=labelx)

    counts, bins = np.histogram(y, bins=n_bins, range=range_)
    counts = counts / np.sum(counts)
    plt.hist(bins[:-1], bins, weights=counts, alpha=0.5, label=labely)

    #plt.hist(y.flatten(), bins, alpha=0.5, label=labely)
    plt.legend(loc='best')
    plt.title(title)
    plt.savefig(out_path)
    plt.close()

def plot_save(x, title, path):
    plt.imshow(x)
    plt.colorbar()
    plt.title(title)
    plt.savefig(path)
    plt.close()

def save_losses(metrics_path, out_path, alpha=None, beta=None, epochs=None, first_n=0):
    if alpha is None or epochs is None:
        return 
    
    data = np.load(metrics_path)
    # Values 
    # mge_train_loss, mse_train_loss,train_loss,train_psnr,train_ssim,mge_val_loss,mse_val_loss,val_loss,val_psnr,val_ssim
    mge_train_loss = data[0,:] * alpha
    mse_train_loss = data[1,:] * beta
    train_loss = data[2,:]

    mge_val_loss = data[5,:] * alpha
    mse_val_loss = data[6,:] * beta
    val_loss = data[7,:]

    
    plt.plot(epochs[first_n:], mge_train_loss[first_n:], label="MGE Loss")
    plt.plot(epochs[first_n:], mse_train_loss[first_n:], label="MSE Loss")
    plt.plot(epochs[first_n:], train_loss[first_n:], label="Total Loss")
    plt.xlabel("Epochs")
    plt.title("Train losses \n alpha:{:4.3f} beta:{:4.3f}".format(alpha, beta))
    plt.legend(loc="best")
    plt.savefig(out_path + "loss_train.png")
    plt.close()

    plt.plot(epochs, mge_val_loss, label="MGE Loss")
    plt.plot(epochs, mse_val_loss, label="MSE Loss")
    plt.plot(epochs, val_loss, label="Total Loss")
    plt.xlabel("Epochs")
    plt.title("Validation losses \n alpha:{:4.3f}".format(alpha))
    plt.legend(loc="best")
    plt.savefig(out_path + "loss_val.png")
    plt.close()

    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epochs")
    plt.title("Loss comparison \n alpha:{:4.3f}".format(alpha))
    plt.legend(loc="best")
    plt.savefig(out_path + "loss_both.png")
    plt.close()


def main(MAX_CONSTANT, alpha=0.0001, beta=0.9999, epochs=150):
    
    out_epoch = 5
    model_name = "newBeta0-9999"


    edge = 16
    n_imgs = 5

    samples_dir = './output/'+ model_name +'/samples/'
    training_data_dir = './output/'+ model_name +'/training_data/'
    metrics_dir = './output/'+ model_name +'/metrics/'
    #samples_dir = './samples/'
    #training_data_dir = './'
    #metrics_dir = './Metrics/'

    # save_losses(metrics_dir + 'metrics.npy', metrics_dir, alpha=alpha, beta=beta, epochs=[i for i in range(1,epochs+1)])
     
    out_epoch_file = training_data_dir + 'output_ep_{:d}.npz'.format(out_epoch)
    original_imgs_file = training_data_dir + 'original_images.npz'
    data = np.load(out_epoch_file)
    originals = np.load(original_imgs_file)

    outputs = data['outputs'] # model outputs
    lst = originals['lst'] # lst outputs
    ndvi = originals['ndvi'] # ndvi gradients
    original_lst = originals['lst_original'] # original lst
    original_ndvi = originals['ndvi_original'] # original ndvi

    # print(np.shape(outputs))
    # print(np.shape(lst))
    # print(np.shape(ndvi))
    # print(np.shape(original_lst))
    # print(np.shape(original_ndvi))

    for i in range(n_imgs):
        output_image = outputs[i,0,:,:]
        output_image = output_image[edge:256-edge,edge:256-edge] * MAX_CONSTANT
        # save_histogram(output_image, './samples/hist_output_'+str(i)+'.png', "Histogram \n from: Output LST 256x256 250m", n_bins=20)
        # save_histogram(original_lst[i,:,:]* MAX_CONSTANT, './samples/hist_original_lst_'+str(i)+'.png', "Histogram \n from: Original LST 64x64 250m", n_bins=20)

        save_double_histograms(output_image, original_lst[i,:,:]* MAX_CONSTANT, "Output LST 256x256 1km", "Real LST 64x64 250m", samples_dir + 'hist_comp_'+str(i)+'.png', "Histogram comparison")

        plot_save(output_image, "Model Output LST \n 256x256 250m edge:{:d}".format(edge), samples_dir + 'output_'+str(i)+'.png')
        plot_save(lst[i,:,:], "Interpolated Bicubic LST \n 256x256 1km", samples_dir + 'lst_'+str(i)+'.png')
        plot_save(ndvi[i,:,:], "Normalized NVDI Gradient \n (from: 256x256 250m)", samples_dir + 'ndvi_'+str(i)+'.png')
        plot_save(original_lst[i,:,:]* MAX_CONSTANT, "Orignal LST \n 64x64 1km", samples_dir + 'original_lst_'+str(i)+'.png')
        plot_save(original_ndvi[i,:,:], "Orignal NDVI \n 256x256 250m", samples_dir + 'original_ndvi_'+str(i)+'.png')

        image = original_lst[i,:,:]
        up = cv2.resize(image, (256, 256), cv2.INTER_CUBIC)
        up = torch.reshape(torch.Tensor(up),(1,256,256))
        down = resize(up,(64,64),T.InterpolationMode.BICUBIC)
        down = down.detach().numpy()
        print("Norm between original LST and upscale then downscale version:", np.linalg.norm(original_lst-down))

if __name__ == "__main__":
    MAX_CONSTANT = 333.32000732421875
    main(MAX_CONSTANT)
