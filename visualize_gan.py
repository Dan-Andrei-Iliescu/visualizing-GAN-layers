import tensorflow as tf
import numpy as np
from gan import Gan
from classifier import Classifier
from visualize import visualize_mnist

tf.enable_eager_execution()
NUMBER = 4096

# Import MNIST
(_, _), (test_images, test_labels) =\
    tf.keras.datasets.mnist.load_data()

# Shape images
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)\
    .astype('float32')
test_labels = test_labels.astype('int64')

# Pad images to make 32x32
test_images = np.pad(
    array=test_images, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)),
    mode='constant', constant_values=0.0
)

# Normalize
test_images = (test_images - 127.5) / 127.5

# Random subsample of test images and labels
np.random.seed(128)
real_images = np.random.permutation(test_images)[:NUMBER]
np.random.seed(128)
real_labels = np.random.permutation(test_labels)[:NUMBER]

# Instantiate a gan
gan = Gan()

# Train gan
dr1, dr2, dr3, g1, g2, g3, g4, dg1, dg2, dg3 = gan.layers(real_images)

# Visualize discriminator for real images
visualize_mnist(real_images, real_labels, "Real_MNIST")
visualize_mnist(dr1, real_labels, "Real_MNIST-discriminator_layer_1")
visualize_mnist(dr2, real_labels, "Real_MNIST-discriminator_layer_2")
visualize_mnist(dr3, real_labels, "Real_MNIST-discriminator_layer_3")

# Instantiate a classifier
classifier = Classifier()

# Train classifier
gen_labels = classifier.predict(g4)

# Visualize discriminator for real images
visualize_mnist(g1, gen_labels, "GAN_MNIST-generator_layer_1")
visualize_mnist(g2, gen_labels, "GAN_MNIST-generator_layer_2")
visualize_mnist(g3, gen_labels, "GAN_MNIST-generator_layer_3")
visualize_mnist(g4, gen_labels, "GAN_MNIST")
visualize_mnist(dg1, gen_labels, "GAN_MNIST-discriminator_layer_1")
visualize_mnist(dg2, gen_labels, "GAN_MNIST-discriminator_layer_2")
visualize_mnist(dg3, gen_labels, "GAN_MNIST-discriminator_layer_3")
