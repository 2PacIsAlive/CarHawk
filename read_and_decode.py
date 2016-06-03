def read_and_decode(filename_queue):
	'''
	reader = tensorflow.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tensorflow.parse_single_example(serialized_example,
                                     dense_keys=['image_raw', 'label'],
                                     # Defaults are not specified since both keys are required.
                                     dense_types=[tensorflow.string, tensorflow.int64])

	# Convert from a scalar string tensor to a uint8 tensor
	image = tensorflow.decode_raw(features['image_raw'], tf.uint8)

	#image = tensorflow.reshape(image, [my_cifar.n_input])
	#image.set_shape([my_cifar.n_input])

	# Convert label from a scalar uint8 tensor to an int32 scalar.
	label = tensorflow.cast(features['label'], tensorflow.int32)

	return image, label
	'''
	images, labels = imageflow.inputs(filename=os.path.join(DATA_DIRECTORY, 'training.tfrecords'), batch_size=128,
                                      num_epochs=10000,
                                      num_threads=5, imshape=[640, 480, 3])
	val_images, val_labels = imageflow.inputs(filename=os.path.join(DATA_DIRECTORY, 'validation.tfrecords'), batch_size=128,
                                    num_epochs=10000,
                                    num_threads=5, imshape=[640, 480, 3])

	print images 
	print labels
	print val_images
	print val_labels
