from __future__ import print_function

from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np

class GAN():
	def __init__(self):
		self.rows = 32
		self.cols = 32
		self.channels = 3
		self.img_shape = (self.rows, self.cols, self.channels)
		self.num_classes = 2
		optimizer = Adam(0.0002, 0.5)
		 # Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss=['binary_crossentropy'], 
		    loss_weights=[0.5],
		    optimizer=optimizer,
		    metrics=['accuracy'])

		# Build and compile the generator
		self.generator = self.build_generator()
		self.generator.compile(loss=['binary_crossentropy'], 
		    optimizer=optimizer)

		# The generator takes noise as input and generates imgs
		z = Input(shape=self.img_shape)
		gen_img = self.generator(z)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False
		# The valid takes generated images as input and determines validity
		valid = self.discriminator(gen_img)

		# The combined model  (stacked generator and discriminator) takes
		# masked_img as input => generates images => determines validity 
		self.combined = Model(z, valid)
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


	def build_generator(self):


		model = Sequential()

		# try to change strides to 1 
		# try kernel_size of 3 
		# imagenet
		# use maxpooling
		# use only cat or dogs
		# use max pooling
		# use padding as valid in encoder part
		# try to use sigmoid in place of tanh in output layer

		# Encoder
		model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=self.img_shape, padding="same"))
		model.add(Activation('relu'))
		model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
		model.add(Activation('relu'))
		model.add(Conv2D(512, kernel_size=4, strides=2, padding="same"))
		model.add(Activation('relu'))

		# Decoder
		model.add(UpSampling2D())
		model.add(Conv2D(256, kernel_size=4, padding="same"))
		model.add(Activation('relu'))
		model.add(UpSampling2D())
		model.add(Conv2D(128, kernel_size=4, padding="same"))
		model.add(Activation('relu'))
		model.add(UpSampling2D())
		model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
		model.add(Activation('tanh'))

		model.summary()

		noise = Input(shape=self.img_shape)
		img = model(noise)

		return Model(noise, img)

	def build_discriminator(self):

		model = Sequential()

		model.add(Conv2D(32, kernel_size=3, input_shape=self.img_shape, padding="same"))
		model.add(Activation('relu'))

		model.add(MaxPooling2D())

		model.add(Conv2D(64, kernel_size=3, padding="same"))
		model.add(Activation('relu'))

		model.add(MaxPooling2D())

		model.add(Conv2D(128, kernel_size=3, padding="same"))
		model.add(Activation('relu'))
		model.add(Conv2D(128, kernel_size=3, padding="same"))
		model.add(Activation('relu'))

		model.add(MaxPooling2D())

		model.add(Conv2D(256, kernel_size=3, padding="same"))
		model.add(Activation('relu'))
		model.add(Conv2D(256, kernel_size=3, padding="same"))
		model.add(Activation('relu'))

		model.add(MaxPooling2D())

		model.add(Flatten())

		model.summary()

		img = Input(shape=self.img_shape)
		features = model(img)

		valid = Dense(1, activation="sigmoid")(features)
		label = Dense(self.num_classes+1, activation="softmax")(features)

		return Model(img, valid)



	def train(self, epochs, batch_size=128, save_interval=50):

		# Load the dataset
		(X_train, _), (_, _) = cifar10.load_data()

		# Rescale -1 to 1
		X_train = X_train / 255
		X_train = 2 * X_train - 1

		half_batch = int(batch_size / 2)

		count = 1
		for epoch in range(epochs):
			print("count : ", count)
			count +=1

			# training the discriminator
			# Select a random half batch of images
			idx = np.random.randint(0, X_train.shape[0], half_batch)
			imgs = X_train[idx]
			noise = np.random.normal(0, 1, (half_batch, 32 , 32 , 3))

			# Generate a half batch of new images
			gen_imgs = self.generator.predict(noise)
			# print("imgs shape is : ", imgs.shape)

			valid = np.ones((half_batch, 1))
			fake = np.zeros((half_batch, 1))
			# Train the discriminator
			d_loss_real = self.discriminator.train_on_batch(imgs, valid)
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# training the generator

			noise = np.random.normal(0, 1, (batch_size, 32, 32, 3))
			valid = np.array([1] * batch_size)
            
			# Train the generator
			g_loss = self.combined.train_on_batch(noise,  valid)
			print ("%f [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

			if epoch % save_interval == 0:
			    self.save_imgs(epoch)


	def save_imgs(self, epoch):
		r, c = 3,6
		noise = np.random.normal(0, 1, (r * c, 32, 32, 3))
		gen_imgs = self.generator.predict(noise)

		# Rescale images 0 - 1
		gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(r, c)
		cnt=0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt, :,:,0])
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig("gan/images2/cifar_%d.png" % epoch)
		plt.close()



if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=200000, batch_size=32, save_interval=200)






