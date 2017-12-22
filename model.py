from __future__ import print_function

from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.initializers import Initializer
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

import numpy as np




class GAN():
	def __init__(self):
		self.rows = 32
		self.cols = 32
		self.channels = 3
		self.img_shape = (self.rows, self.cols, self.channels)
		self.num_classes = 2
		optimizer = Adam(0.0001, 0.5)
		self.generator = self.build_generator()
		self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
		# The generator takes noise as input and generated imgs
		z = Input(shape=(500,))
		img = self.generator(z)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The valid takes generated images as input and determines validity
		valid = self.discriminator(img)

		# The combined model  (stacked generator and discriminator) takes
		# noise as input => generates images => determines validity 
		self.combined = Model(z, valid)
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

		self.predict_model = self.cnn_predict()

		self.predict_model.compile(loss='categorical_crossentropy', # Better loss function for neural networks
              optimizer=Adam(lr=1.0e-4), # Adam optimizer with 1.0e-4 learning rate
              metrics = ['accuracy']) # Metrics to be evaluated by the model

	def build_generator(self):

		noise_shape = (500,)

		model = Sequential()

		model.add(Dense(2048 * 2 * 2, activation="relu", input_shape=noise_shape, kernel_initializer='glorot_normal'))
		model.add(Reshape((2, 2, 2048)))
		model.add(BatchNormalization(momentum=0.8))
		model.add(UpSampling2D())
		model.add(Conv2D(512, kernel_size=3, strides=1 , padding="same"))
		model.add(Activation("relu"))
		model.add(UpSampling2D())
		model.add(BatchNormalization(momentum=0.8)) 
		model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
		model.add(Activation("relu"))
		model.add(UpSampling2D())
		model.add(BatchNormalization(momentum=0.8))
		model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(momentum=0.8))
		model.add(UpSampling2D())
		model.add(Conv2D(3, kernel_size=3, padding="same"))
		model.add(Activation("sigmoid"))

		model.summary()

		noise = Input(shape=noise_shape)
		img = model(noise)

		return Model(noise, img)


	def build_discriminator(self):

		model = Sequential()

		model.add(Conv2D(64, kernel_size=3, strides=1,input_shape=self.img_shape,  padding="same"))
		model.add(LeakyReLU())
		model.add(MaxPooling2D())
		model.add(Conv2D(128, kernel_size=3, strides=1,input_shape=self.img_shape,  padding="same"))
		model.add(LeakyReLU())
		model.add(MaxPooling2D())
		model.add(Conv2D(256, kernel_size=3, strides=1,input_shape=self.img_shape,  padding="same"))
		model.add(LeakyReLU())
		model.add(MaxPooling2D())
		model.add(Conv2D(512, kernel_size=3, strides=1,input_shape=self.img_shape,  padding="same"))
		model.add(Activation("relu"))
		model.add(Flatten())
		img = Input(shape=self.img_shape)
		features = model(img)

		valid = Dense(1, activation="sigmoid")(features)

		model.summary()
		return Model(img,valid)


	def cnn_predict(self):
	    
	    model = Sequential()
	    # self.no_classes = 10
	    
	    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', input_shape=self.img_shape))    
	    model.add(Dropout(0.2))
	    
	    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same'))  
	    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2))    
	    model.add(Dropout(0.5))
	    
	    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))    
	    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))
	    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2))    
	    model.add(Dropout(0.5))    
	    
	    model.add(Conv2D(192, (3, 3), padding = 'same'))
	    model.add(Activation('relu'))
	    model.add(Conv2D(192, (1, 1),padding='valid'))
	    model.add(Activation('relu'))
	    model.add(Conv2D(10, (1, 1), padding='valid'))

	    model.add(GlobalAveragePooling2D())
	    
	    model.add(Activation('softmax'))

	    model.summary()

	    return model
	def train_generate(self , epochs , batch_size = 128 , save_interval = 200): 

		(x_train , _ ), (_ , _) = cifar10.load_data()

		x_train = x_train / 255.0

		half_batch = int(batch_size / 2)

		for epoch in range(epochs):


			# Select a random half batch of images
			idx = np.random.randint(0, x_train.shape[0], half_batch)
			imgs = x_train[idx]

			# Sample noise and generate a half batch of new images
			noise = np.random.normal(0, 1, (half_batch, 500))
			

			if(epoch %2 == 0):
				gen_imgs = self.generator.predict(noise)
				# Train the discriminator (real classified as ones and generated as zeros)
				d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
				d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# ---------------------
			#  Train Generator
			# ---------------------

			noise = np.random.normal(0, 1, (batch_size,500))

			# Train the generator (wants discriminator to mistake images as real)
			g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

			# Plot the progress
			if epoch%2 == 0 :
				print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

			# logs = {'d_acc': 100*d_loss[1], 'd_loss_real': d_loss_real[0],'d_loss_fake': d_loss_fake[0] , 'd_loss_overall': d_loss[0],  'g_loss': g_loss}
			# TC.on_epoch_end(epoch, logs)

			# If at save interval => save generated image samples
			if epoch % save_interval == 0:
			    self.save_imgs(epoch)


	def data_preprocessing_cifar(self):
		(X_train, y_train), (X_test, y_test) = cifar10.load_data() # fetch CIFAR-10 data

		num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10 
		num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
		num_classes = np.unique(y_train).shape[0] # there are 10 image classes

		X_train = X_train.astype('float32') 
		X_test = X_test.astype('float32')
		X_train /= np.max(X_train) # Normalise data to [0, 1] range
		X_test /= np.max(X_test) # Normalise data to [0, 1] range

		Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
		Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

		return (X_train, Y_train),(X_test, Y_test)


	def train_predict_labels(self, epochs, batch_size=128, data_augmentation="None"):


		(X_train, Y_train), (X_test, Y_test) = self.data_preprocessing_cifar()

		
		if data_augmentation=="None":                          
			checkpoint = ModelCheckpoint('model/normal/best_model.h5',  # model filename
	                         monitor='val_loss', # quantity to monitor
	                         verbose=0, # verbosity - 0 or 1
	                         save_best_only= True, # The latest best model will not be overwritten
	                         mode='auto') # The decision to overwrite model is made 
	                                      # automatically depending on the quantity to monitor

			tbCallBack = TensorBoard(log_dir='./Graph/normal', histogram_freq=0, write_graph=True, write_images=True)
                              
			model_details = self.predict_model.fit(X_train, Y_train,batch_size = batch_size, epochs=epochs,  validation_data= (X_test, Y_test),callbacks=[checkpoint,tbCallBack],verbose=1)
		
		elif data_augmentation=="With_Keras_Augmentation":
			checkpoint = ModelCheckpoint('model/with_keras_aug/best_model_k.h5',  # model filename
                 monitor='val_loss', # quantity to monitor
                 verbose=0, # verbosity - 0 or 1
                 save_best_only= True, # The latest best model will not be overwritten
                 mode='auto') # The decision to overwrite model is made 
                              # automatically depending on the quantity to monitor

			tbCallBack = TensorBoard(log_dir='./Graph/with_keras_aug', histogram_freq=0, write_graph=True, write_images=True)

			# datagen = ImageDataGenerator(zoom_range=0.2, 
   #                           horizontal_flip=True)
			

			datagen = ImageDataGenerator(
			    featurewise_center=False,  # set input mean to 0 over the dataset
			    samplewise_center=False,  # set each sample mean to 0
			    featurewise_std_normalization=False,  # divide inputs by std of the dataset
			    samplewise_std_normalization=False,  # divide each input by its std
			    zca_whitening=False,  # apply ZCA whitening
			    rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
			    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
			    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
			    horizontal_flip=True,  # randomly flip images
			    vertical_flip=False)  # randomly flip images
			model_info = self.predict_model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size),
                                 samples_per_epoch = X_train.shape[0], nb_epoch = epochs, 
                                 validation_data = (X_test, Y_test), callbacks=[checkpoint,tbCallBack], verbose=1)

		elif data_augmentation=="With_Gan_Augmentation":

			checkpoint = ModelCheckpoint('model/best_model_simple_with_gan_augmentation.h5',  # model filename
                 monitor='val_loss', # quantity to monitor
                 verbose=0, # verbosity - 0 or 1
                 save_best_only= True, # The latest best model will not be overwritten
                 mode='auto') # The decision to overwrite model is made 
                              # automatically depending on the quantity to monitor


	
	def predict_labels(self, path="model/best_model_simple.h5",batch_size=128):

		(X_train, Y_train), (X_test, Y_test) = self.data_preprocessing_cifar()
		model = load_model(path)

		loss , acc = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose=1)


		print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


	def save_imgs(self, epoch):
		r, c = 4 , 4
		noise = np.random.normal(0, 1, (r * c, 500))
		data = self.generator.predict(noise)
		cnt = 0

		for i in range(r*c):
			plt.subplot(r,c,i+1)
			plt.imshow(data[i])
			# plt.show()
		plt.savefig("gan/image2/cifar10_%d.png" % epoch)

