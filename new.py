from model import *


if __name__ == '__main__':
	gan = GAN()
	TC = TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)
	TC.set_model(gan.combined)
	gan.train_predict_labels(epochs=100,batch_size=128,data_augmentation="With_Keras_Augmentation")
	# gan.train_generate(epochs=30000, batch_size=32, save_interval=200)
	# gan.predict_labels()

	# TC.on_train_end()

	# gen.build_generator

