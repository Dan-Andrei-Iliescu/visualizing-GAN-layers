import tensorflow as tf
import numpy as np
from classifier import Classifier

tf.enable_eager_execution()

BATCH_SIZE = 256
EPOCHS = 10

# Import MNIST
(train_images, train_labels), (test_images, test_labels) =\
    tf.keras.datasets.mnist.load_data()

# Shape images
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)\
    .astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)\
    .astype('float32')

# Pad images to make 32x32
train_images = np.pad(
    array=train_images, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)),
    mode='constant', constant_values=0.0
)
test_images = np.pad(
    array=test_images, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)),
    mode='constant', constant_values=0.0
)

# Normalize
train_images = (train_images - 127.5) / 127.5
test_images = (test_images - 127.5) / 127.5

# Build datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    train_images, train_labels.astype('int64')
)).shuffle(60000).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((
    test_images, test_labels.astype('int64')
)).shuffle(10000).batch(BATCH_SIZE)

# Instantiate a classifier
classifier = Classifier()

# Train classifier
classifier.train(train_dataset, test_dataset, EPOCHS)
