#!/usr/bin/env python2

import tensorflow as tf
import os
import re

BATCH_SIZE = 10 # TODO make FLAGS
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 504
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 504
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def _variable_on_cpu(name, shape, initializer):
  	"""Helper to create a Variable stored on CPU memory.

  	Args:
    	name: name of the variable
    	shape: list of ints
    	initializer: initializer for Variable

  	Returns:
    	Variable Tensor
  	"""
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=initializer)
  	return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  	"""Helper to create an initialized Variable with weight decay.

  	Note that the Variable is initialized with a truncated normal distribution.
  	A weight decay is added only if one is specified.

  	Args:
    	name: name of the variable
    	shape: list of ints
    	stddev: standard deviation of a truncated Gaussian
    	wd: add L2Loss weight decay multiplied by this float. If None, weight
        	decay is not added for this Variable.

  	Returns:
    	Variable Tensor
  	"""
  	var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    	tf.add_to_collection('losses', weight_decay)
  	return var

def _activation_summary(x):
  	"""Helper to create summaries for activations.

  	Creates a summary that provides a histogram of activations.
  	Creates a summary that measure the sparsity of activations.

  	Args:
  		x: Tensor
  	Returns:
    	nothing
  	"""
  	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  	# session. This helps the clarity of presentation on tensorboard.
  	tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
  	tf.histogram_summary(tensor_name + '/activations', x)
  	tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _add_loss_summaries(total_loss):
  	"""Add summaries for losses in CIFAR-10 model.

  	Generates moving average for all losses and associated summaries for
  	visualizing the performance of the network.

  	Args:
    	total_loss: Total loss from loss().
  	Returns:
    	loss_averages_op: op for generating moving averages of losses.
  	"""
  	# Compute the moving average of all individual losses and the total loss.
  	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  	losses = tf.get_collection('losses')
  	loss_averages_op = loss_averages.apply(losses + [total_loss])

  	# Attach a scalar summary to all individual losses and the total loss; do the
  	# same for the averaged version of the losses.
  	for l in losses + [total_loss]:
		# Name each loss as '(raw)' and name the moving average version of the loss
		# as the original loss name.
		tf.scalar_summary(l.op.name +' (raw)', l)
		tf.scalar_summary(l.op.name, loss_averages.average(l))

  	return loss_averages_op

def _generate_image_and_label_for_test(image, label):
	images, label_batch = tf.train.batch(
		[image, label],
		batch_size=1, # number of test images
		num_threads=16,
		capacity=1)

  	# Display the training images in the visualizer.
  	tf.image_summary('images', images)

  	return images, tf.reshape(label_batch, [1])

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  	"""Construct a queued batch of images and labels.

  	Args:
    	image: 3-D Tensor of [height, width, 3] of type.float32.
    	label: 1-D Tensor of type.int32
    	min_queue_examples: int32, minimum number of samples to retain
      		in the queue that provides of batches of examples.
    	batch_size: Number of images per batch.
    	shuffle: boolean indicating whether to use a shuffling queue.

  	Returns:
    	images: Images. 4D tensor of [batch_size, height, width, 3] size.
    	labels: Labels. 1D tensor of [batch_size] size.
  	"""
  	# Create a queue that shuffles the examples, and then
  	# read 'batch_size' images + labels from the example queue.
  	num_preprocess_threads = 16
  	if shuffle:
  		images, label_batch = tf.train.shuffle_batch(
        	[image, label],
        	batch_size=batch_size,
        	num_threads=num_preprocess_threads,
        	capacity=min_queue_examples + 3 * batch_size,
        	min_after_dequeue=min_queue_examples)
	else:
		images, label_batch = tf.train.batch(
        	[image, label],
        	batch_size=batch_size,
        	num_threads=num_preprocess_threads,
        	capacity=min_queue_examples + 3 * batch_size)

  	# Display the training images in the visualizer.
  	tf.image_summary('images', images)

  	return images, tf.reshape(label_batch, [batch_size])


def inference(images):
	"""Build the HawkNet model.

	Args:
		images: Images returned from distorted_inputs() or inputs().

	Returns:
		Logits.
	"""
	# We instantiate all variables using tf.get_variable() instead of
	# tf.Variable() in order to share variables across multiple GPU training runs.
	# If we only ran this model on a single GPU, we could simplify this function
	# by replacing all instances of tf.get_variable() with tf.Variable().
	#
	# conv1
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
										 stddev=1e-4, wd=0.0)
		conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		bias = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(bias, name=scope.name)
		_activation_summary(conv1)

	# pool1
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
						 padding='SAME', name='pool1')
	# norm1
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					name='norm1')

	# conv2
	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
										 stddev=1e-4, wd=0.0)
		conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
		bias = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(bias, name=scope.name)
		_activation_summary(conv2)

	# norm2
	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
					name='norm2')
	# pool2
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
						 strides=[1, 2, 2, 1], padding='SAME', name='pool2')

	# local3
	with tf.variable_scope('local3') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
		reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
		dim = reshape.get_shape()[1].value
		weights = _variable_with_weight_decay('weights', shape=[dim, 384],
										  stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
		_activation_summary(local3)

	# local4
	with tf.variable_scope('local4') as scope:
		weights = _variable_with_weight_decay('weights', shape=[384, 192],
										  stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
		_activation_summary(local4)

	# softmax, i.e. softmax(WX + b)
	with tf.variable_scope('softmax_linear') as scope:
		weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
										  stddev=1/192.0, wd=0.0)
		biases = _variable_on_cpu('biases', [NUM_CLASSES],
							  tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
		_activation_summary(softmax_linear)

	return softmax_linear

def loss(logits, labels):
	"""Add L2Loss to all the trainable variables.

	Add summary for "Loss" and "Loss/avg".
	Args:
		logits: Logits from inference().
		labels: Labels from distorted_inputs or inputs(). 1-D tensor
			of shape [batch_size]

	Returns:
		Loss tensor of type float.
	"""
	# Calculate the average cross entropy loss across the batch.
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits, labels, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def read_and_decode(filename_queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		serialized_example,
		# Defaults are not specified since both keys are required.
		features={
			'image_raw': tf.FixedLenFeature([], tf.string),
			'label':     tf.FixedLenFeature([], tf.int64),
		})
	
	# Convert from a scalar string tensor (whose single string has
	# length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
	# [mnist.IMAGE_PIXELS].
	image = tf.decode_raw(features['image_raw'], tf.uint8)
	image.set_shape([57600])

	# OPTIONAL: Could reshape into a 28x28 image and apply distortions
	# here.  Since we are not applying any distortions in this
	# example, and the next step expects the image to be flattened
	# into a vector, we don't bother.

	# Convert from [0, 255] -> [-0.5, 0.5] floats.
	image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

	# Convert label from a scalar uint8 tensor to an int32 scalar.
	#label = tf.cast(features['label'], tf.int32) <-- placeholder instead

	return tf.reshape(image, [160, 120, 3]), tf.placeholder(tf.int32) # TODO doublecheck this

def validation_input(image_path):
	if not tf.gfile.Exists(image_path):
		raise ValueError('Failed to find file: ' + image_path)

  	# Create a queue that produces the filename to read.
  	with tf.name_scope('input'):
		filename_queue = tf.train.string_input_producer(image_path)

  	# Read example from file in the filename queue.
  	float_image, label = read_and_decode(filename_queue)
	
	return _generate_image_and_label_for_test(float_image, label)

def inputs(validate, data_dir, batch_size):
  	"""Construct input using the Reader ops.

  	Args:
    	eval_data: bool, indicating if one should use the train or eval data set.
    	data_dir: Path to the data directory.
    	batch_size: Number of images per batch.

  	Returns:
    	images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    	labels: Labels. 1D tensor of [batch_size] size.
  	"""
	if not validate:
		filenames = [os.path.join(data_dir, 'train_c%d.tfrecords' % i)
                 for i in xrange(0, 10)] 
		num_examples_per_epoch = 504
	else:
		filenames = [os.path.join(data_dir, 'validation.tfrecords')]
    	num_examples_per_epoch = 504

	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

  	# Create a queue that produces the filenames to read.
  	with tf.name_scope('input'):
		filename_queue = tf.train.string_input_producer(filenames)

  	# Read examples from files in the filename queue.
  	float_image, label = read_and_decode(filename_queue)
	#image = tf.cast(image, tf.float32)
	#reshaped_image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

	#reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  	height = 32
  	width = 32

  	# Image processing for evaluation.
  	# Crop the central [height, width] of the image.
  	#resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
    #                                                     width, height)

  	# Subtract off the mean and divide by the variance of the pixels.
  	#float_image = tf.image.per_image_whitening(image)

  	# Ensure that the random shuffling has good mixing properties.
  	min_fraction_of_examples_in_queue = 0.4
  	min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

	if not validate:
	  	# Generate a batch of images and labels by building up a queue of examples.
	  	return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
	else: 
		return _generate_image_and_label_for_test(float_image, label)

def train(total_loss, global_step):
  	"""Train HawkNet model.

  	Create an optimizer and apply to all trainable variables. Add moving
  	average for all trainable variables.

  	Args:
    	total_loss: Total loss from loss().
    	global_step: Integer Variable counting the number of training steps
      		processed.
  	Returns:
    	train_op: op for training.
  	"""
  	# Variables that affect learning rate.
  	num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
  	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  	# Decay the learning rate exponentially based on the number of steps.
  	lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  	tf.scalar_summary('learning_rate', lr)

  	# Generate moving averages of all losses and associated summaries.
  	loss_averages_op = _add_loss_summaries(total_loss)

  	# Compute gradients.
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
    	grads = opt.compute_gradients(total_loss)

  	# Apply gradients.
  	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  	# Add histograms for trainable variables.
	for var in tf.trainable_variables():
		tf.histogram_summary(var.op.name, var)

  	# Add histograms for gradients.
	for grad, var in grads:
		if grad is not None:
			tf.histogram_summary(var.op.name + '/gradients', grad)

  	# Track the moving averages of all trainable variables.
  	variable_averages = tf.train.ExponentialMovingAverage(
    	MOVING_AVERAGE_DECAY, global_step)
  	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

  	return train_op
