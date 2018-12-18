import tensorflow as tf
import numpy as np
from gan import Gan

tf.enable_eager_execution()

BATCH_SIZE = 256
BUFFER_SIZE = 70000
EPOCHS = 40

# Import MNIST
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

# Shape images
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)\
    .astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)\
    .astype('float32')

# Concatenate
images = np.concatenate((train_images, test_images), axis=0)

# Pad images to make 32x32
images = np.pad(
    array=images, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)),
    mode='constant', constant_values=0.0
)

# Normalize
images = (images - 127.5) / 127.5

# Build datasets
dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(BUFFER_SIZE).\
    batch(BATCH_SIZE)

# Instantiate a gan
gan = Gan()

# Train gan
gan.train(dataset, EPOCHS)
