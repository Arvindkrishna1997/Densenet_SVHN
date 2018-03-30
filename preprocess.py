#%matplotlib inline
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from scipy.io import loadmat

train_data = loadmat('train_32x32.mat', variable_names='X').get('X')
train_labels = loadmat('train_32x32.mat', variable_names='y').get('y')

train_data, train_labels = train_data.transpose((3,0,1,2)), train_labels[:,0]

print(train_data.shape, train_labels.shape)

def plot_images(img, labels, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat):
        print("i", i ," ax ", ax)
        print(img[i].shape)
        im = img[i]
        if img[i].shape == (32, 32, 3):
            plt.subplot(ax)
            plt.imshow(img[i])
        else:
            plt.subplot(ax)
            plt.imshow(img[i,:,:,0])
        plt.subplots_adjust(wspace=0.2, hspace=0)
        ax.set_title(labels[i])
        ax.set_xticks([]);ax.set_yticks([])
    plt.show()

num_images = train_data.shape[0]
#plot_images(train_data, train_labels, 2, 8)
print("total no of images is ", num_images)

train_labels[train_labels== 10] = 0

def balanced_subsampling(y, s):
    sample = []
    for label in np.unique(y):
        images = np.where(y==label)[0]
        random_sample = np.random.choice(images, size=s, replace=False)
        sample += random_sample.tolist()
    return sample

train_samples = balanced_subsampling(train_labels, 400)

x_val, y_val = np.copy(train_data[train_samples]), np.copy(train_labels[train_samples])

perm = np.arange(4000)
np.random.shuffle(perm)
x_val = x_val[perm]
y_val = y_val[perm]

train_data = np.delete(train_data, train_samples, axis=0)
train_labels = np.delete(train_labels, train_samples, axis=0)


#print("Training", x_val.shape, y_val.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

fig.suptitle('Class Distribution', fontsize=14, fontweight='bold', y=1.05)

ax1.hist(train_labels, bins=10)
ax1.set_title("Training set")
ax1.set_xlim(0, 9)

ax2.hist(y_val, color='r', bins=10)
ax2.set_title("Validation set")

fig.tight_layout()
#plt.show()

def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2989, 0.5870, 0.1140]), axis=3)

train_greyscale = rgb2gray(train_data).astype(np.float32)
valid_greyscale = rgb2gray(x_val).astype(np.float32)
plot_images(valid_greyscale, y_val, 2, 8)

#plot_images(x_val  , y_val, 2, 10)
print("dimensions")
print("Training set", train_data.shape, train_greyscale.shape)
print("Validation set", x_val.shape, valid_greyscale.shape)

import h5py

h5f = h5py.File('./SVHN_greyscale.h5', 'w')

h5f.create_dataset('x_train', data=train_greyscale)
h5f.create_dataset('y_train', data=train_labels)
h5f.create_dataset('x_val', data=valid_greyscale)
h5f.create_dataset('y_val', data=y_val)

h5f.close()
