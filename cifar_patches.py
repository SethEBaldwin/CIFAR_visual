
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#CHANGE THESE HYPERPARAMETERS TO GENERATE DIFFERENT IMAGE PATCHES:	
LAYER = 5 #which convolutional layer of the network to use, from 0 to 5
ROWS = 4 #number of rows of image patches to generate for each filter
COLUMNS = 4	#number of columns of image patches to generate for each filter
FILTER_IDXS = range(3) #list of filter indexes to print image patches for
						#Valid range of filter indices:
						#layers 0,1: indices can range from 0 to 31
						#layers 2,3: indices can range from 0 to 63
						#layers 4,5: indices can range from 0 to 123					
						
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model = keras.models.load_model('cifar_model')
model.summary()

#shows images on a rows x columns grid
def show_img_grid(imgs,rows,columns):
	plt.figure(figsize=(rows,columns))
	for i in range(rows*columns):
		plt.subplot(rows,columns,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(imgs[i], cmap=plt.cm.binary)
	plt.show()

#returns indices of n largest elements of array (unsorted)
def argmaxs(array,n):
	idxs = np.argpartition(array.flatten(), -n)[-n:]
	return [ np.unravel_index(idx, array.shape) for idx in idxs ]

#pads and crops an image
def process_img(img,padding,x0,x1,y0,y1):
	padded = np.pad(img,((padding,padding),(padding,padding),(0,0)),'constant')
	cropped = padded[x0:x1, y0:y1, :]
	return cropped
	
#gets the n image patches that maximize the activations (output) of the given filter (filter_idx)
#padding, patch_size, scale determine how to pad and crop the image patches
def get_img_patches(input_imgs,output,filter_idx,n,padding,patch_size,scale):
	idxs = argmaxs(output[:,:,:,filter_idx],n)
	img_patches = [ 
		process_img(
			input_imgs[idx[0]], padding, scale*idx[1], scale*idx[1] + patch_size, 
			scale*idx[2], scale*idx[2] + patch_size ) 
		for idx in idxs ]
	return img_patches

depths = [0,2,6,8,12,14] #the depths of the conv layers
cumulative_paddings = [1,2,4,6,10,14] #amount to pad image patches for each conv layer
patch_sizes = [3,5,10,14,24,32] #image patch sizes for each conv layer
scales = [1,1,2,2,4,4] #amount to scale indices in each image patch (due to max pooling)

conv_layer = keras.Model(inputs = model.input, outputs = model.get_layer(index = depths[LAYER]).output)
conv_output = conv_layer.predict(x_test)

#for each chosen filter, print rows*columns highest activation image patches in a rows x columns grid.
for filter_idx in FILTER_IDXS:
	imgs = get_img_patches(
		x_test, conv_output, filter_idx, ROWS*COLUMNS,
		cumulative_paddings[LAYER], patch_sizes[LAYER], scales[LAYER] )
	show_img_grid(imgs, ROWS, COLUMNS)
