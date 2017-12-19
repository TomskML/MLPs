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
from keras.callbacks import TensorBoard

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
		self.generator = self.build_generator()
		self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
		# The generator takes noise as input and generated imgs
		z = Input(shape=(100,))
		img = self.generator(z)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The valid takes generated images as input and determines validity
		valid = self.discriminator(img)

		# The combined model  (stacked generator and discriminator) takes
		# noise as input => generates images => determines validity 
		self.combined = Model(z, valid)
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

	def build_generator(self):

		noise_shape = (100,)

		model = Sequential()

		model.add(Dense(1024 * 4 * 4, activation="relu", input_shape=noise_shape))
		model.add(Reshape((4, 4, 1024)))
		model.add(BatchNormalization(momentum=0.8))
		model.add(UpSampling2D())
		model.add(Conv2D(512, kernel_size=3, strides=1 , padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(momentum=0.8)) 
		model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
		model.add(Activation("relu"))
		model.add(UpSampling2D())
		model.add(BatchNormalization(momentum=0.8))
		model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(UpSampling2D())
		model.add(Conv2D(3, kernel_size=3, padding="same"))
		model.add(Activation("tanh"))

		model.summary()

		noise = Input(shape=noise_shape)
		img = model(noise)

		return Model(noise, img)


	def build_discriminator(self):

		model = Sequential()

		model.add(Conv2D(64, kernel_size=3, strides=1,input_shape=self.img_shape,  padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D())
		model.add(Conv2D(128, kernel_size=3, strides=1,input_shape=self.img_shape,  padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D())
		model.add(Conv2D(256, kernel_size=3, strides=1,input_shape=self.img_shape,  padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D())
		model.add(Conv2D(512, kernel_size=3, strides=1,input_shape=self.img_shape,  padding="same"))
		model.add(Activation("relu"))
		model.add(Flatten())
		img = Input(shape=self.img_shape)
		features = model(img)

		valid = Dense(1, activation="sigmoid")(features)

		model.summary()
		return Model(img,valid)


	def train(self , epochs , batch_size = 128 , save_interval = 200): 

		(x_train , _ ), (_ , _) = cifar10.load_data()

		x_train = x_train / 255.0

		half_batch = int(batch_size / 2)

		for epoch in range(epochs):


			# Select a random half batch of images
			idx = np.random.randint(0, x_train.shape[0], half_batch)
			imgs = x_train[idx]

			# Sample noise and generate a half batch of new images
			noise = np.random.normal(0, 1, (half_batch, 100))
			gen_imgs = self.generator.predict(noise)

			# Train the discriminator (real classified as ones and generated as zeros)
			d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# ---------------------
			#  Train Generator
			# ---------------------

			noise = np.random.normal(0, 1, (batch_size, 100))

			# Train the generator (wants discriminator to mistake images as real)
			g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

			# Plot the progress
			print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

			logs = {'d_acc': 100*d_loss[1], 'd_loss_real': d_loss_real[0],'d_loss_fake': d_loss_fake[0] , 'd_loss_overall': d_loss[0],  'g_loss': g_loss}
			TC.on_epoch_end(epoch, logs)

			# If at save interval => save generated image samples
			if epoch % save_interval == 0:
			    self.save_imgs(epoch)

	def save_imgs(self, epoch):
	    r, c = 5, 5
	    noise = np.random.normal(0, 1, (r * c, 100))
	    gen_imgs = self.generator.predict(noise)

	    # Rescale images 0 - 1
	    gen_imgs = 0.5 * gen_imgs + 0.5

	    fig, axs = plt.subplots(r, c)
	    #fig.suptitle("DCGAN: Generated digits", fontsize=12)
	    cnt = 0
	    for i in range(r):
	        for j in range(c):
	            axs[i,j].imshow(gen_imgs[cnt, :,:,0])
	            axs[i,j].axis('off')
	            cnt += 1
	    fig.savefig("gan/images2/cifar10_%d.png" % epoch)
	    plt.close()





if __name__ == '__main__':
	gan = GAN()
	TC = TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)
	TC.set_model(gan.combined)
	gan.train(epochs=30000, batch_size=32, save_interval=50)
	TC.on_train_end()

	# gen.build_generator

