import tensorflow as tf
import time
import os
import numpy as np
import matplotlib.pyplot as plt


class Gan(object):
    def __init__(self):
        super(Gan, self).__init__()

        # Data constants
        self.size = 32
        self.channels = 1
        self.latent_size = 128
        self.depth = 32
        self.train_dir = os.path.join('train', 'gan')

        # Optimizer vars
        self.learning_rate = 0.0001
        self.beta1 = 0.5

        # Sampling generated images
        self.num_samples = 8
        self.z_sample = tf.random_normal(
            [self.num_samples**2, self.latent_size]
        )

        # Choose optimizers
        self.generator_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, beta1=self.beta1
        )
        self.discriminator_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, beta1=self.beta1
        )

        # Make model
        self.generator = self.Generator(
            self.depth, self.size, self.channels, self.latent_size
        )
        self.discriminator = self.Discriminator(self.depth, self.size)

        # Setup checkpoint
        self.checkpoint_dir = os.path.join('checkpoint', 'gan')
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator, discriminator=self.discriminator
        )

        # Speed up training
        self.train_step = tf.contrib.eager.defun(self.train_step)

    class Generator(tf.keras.Model):
        def __init__(self, depth, size, channels, latent_size):
            super(Gan.Generator, self).__init__()

            # Architecture params
            self.depth = depth
            self.size = size
            self.channels = channels
            self.latent_size = latent_size

            # Weighted layers
            self.fc1 = tf.keras.layers.Dense(
                int(4*self.depth*int(self.size/8)*int(self.size/8)),
                use_bias=False, input_shape=(self.latent_size,)
            )
            self.batchnorm1 = tf.keras.layers.BatchNormalization()

            self.conv2 = tf.keras.layers.Conv2DTranspose(
                2*self.depth, (4, 4), strides=(2, 2),
                padding='same', use_bias=False
            )
            self.batchnorm2 = tf.keras.layers.BatchNormalization()

            self.conv3 = tf.keras.layers.Conv2DTranspose(
                self.depth, (4, 4), strides=(2, 2),
                padding='same', use_bias=False
            )
            self.batchnorm3 = tf.keras.layers.BatchNormalization()

            self.conv4 = tf.keras.layers.Conv2DTranspose(
                self.channels, (4, 4), strides=(2, 2), padding='same',
                use_bias=False, activation='tanh'
            )

        def call(self, x, training=True):
            # Layer 1: LATENT_SIZE -> SIZE*SIZE*DEPTH/16
            x = self.fc1(x)
            x = self.batchnorm1(x, training=training)
            x = tf.keras.layers.LeakyReLU()(x)

            # Reshape: SIZE*SIZE*DEPTH/16 -> SIZE/8 x SIZE/8 x 4*DEPTH
            x = tf.keras.layers.Reshape(
                (int(self.size/8), int(self.size/8), 4*self.depth)
            )(x)
            self.g1 = x

            # Layer 2: SIZE/8 x SIZE/8 x 4*DEPTH -> SIZE/4 x SIZE/4 x 2*DEPTH
            x = self.conv2(x)
            x = self.batchnorm2(x, training=training)
            x = tf.keras.layers.LeakyReLU()(x)
            self.g2 = x

            # Layer 3: SIZE/4 x SIZE/4 x 2*DEPTH -> SIZE/2 x SIZE/2 x DEPTH
            x = self.conv3(x)
            x = self.batchnorm3(x, training=training)
            x = tf.keras.layers.LeakyReLU()(x)
            self.g3 = x

            # Layer 4: SIZE/2 x SIZE/2 x DEPTH -> SIZE x SIZE x CHANNELS
            x = self.conv4(x)

            return x

    class Discriminator(tf.keras.Model):
        def __init__(self, depth, size):
            super(Gan.Discriminator, self).__init__()

            # Architecture params
            self.depth = depth
            self.size = size

            # Weighted layers
            self.conv1 = tf.keras.layers.Conv2D(
                self.depth, (4, 4), strides=(2, 2), padding='same'
            )
            self.batchnorm1 = tf.keras.layers.BatchNormalization()

            self.conv2 = tf.keras.layers.Conv2D(
                2*self.depth, (4, 4), strides=(2, 2), padding='same'
            )
            self.batchnorm2 = tf.keras.layers.BatchNormalization()

            self.conv3 = tf.keras.layers.Conv2D(
                4*self.depth, (4, 4), strides=(2, 2), padding='same'
            )
            self.batchnorm3 = tf.keras.layers.BatchNormalization()

            self.fc4 = tf.keras.layers.Dense(1)

        def call(self, x, training=True):
            # Layer 1: SIZE x SIZE x CHANNELS -> SIZE/2 x SIZE/2 x DEPTH
            x = self.conv1(x)
            x = self.batchnorm1(x, training=training)
            x = tf.keras.layers.LeakyReLU()(x)
            self.d1 = x

            # Layer 2: SIZE/2 x SIZE/2 x DEPTH -> SIZE/4 x SIZE/4 x 2*DEPTH
            x = self.conv2(x)
            x = self.batchnorm2(x, training=training)
            x = tf.keras.layers.LeakyReLU()(x)
            self.d2 = x

            # Layer 3: SIZE/4 x SIZE/4 x 2*DEPTH -> SIZE/8 x SIZE/8 x 4*DEPTH
            x = self.conv3(x)
            x = self.batchnorm3(x, training=training)
            x = tf.keras.layers.LeakyReLU()(x)
            self.d3 = x

            # Reshape: SIZE/8 x SIZE/8 x 4*DEPTH -> SIZE*SIZE*DEPTH/16
            x = tf.keras.layers.Flatten()(x)

            # Layer 4: SIZE*SIZE*DEPTH/16 -> 1
            x = self.fc4(x)

            return x

    # Cross entropy
    def generator_loss(self, generated_output):
        return tf.losses.sigmoid_cross_entropy(
            tf.ones_like(generated_output), generated_output
        )

    def discriminator_loss(self, real_output, generated_output):
        # [1,1,...,1] with real output since it is true
        real_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.ones_like(real_output), logits=real_output
        )

        # [0,0,...,0] with generated images since they are fake
        generated_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.zeros_like(generated_output),
            logits=generated_output
        )

        total_loss = real_loss + generated_loss

        return total_loss

    # Update weights
    def train_step(self, images):
        # generating latent code from a normal distribution
        latent_code = tf.random_normal(
            [tf.shape(images)[0], self.latent_size]
        )

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate images
            generated_images = self.generator(latent_code, training=True)

            # Dicriminate real from false
            real_output = self.discriminator(images, training=True)
            generated_output = self.discriminator(
                generated_images, training=True
            )

            # Compute loss
            gen_loss = self.generator_loss(generated_output)
            disc_loss = self.discriminator_loss(real_output, generated_output)

            # Compute gradients
            gradients_of_generator = gen_tape.gradient(
                gen_loss, self.generator.variables
            )
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, self.discriminator.variables
            )

        # Update weights
        self.generator_optimizer.apply_gradients(zip(
            gradients_of_generator, self.generator.variables
        ))
        self.discriminator_optimizer.apply_gradients(zip(
            gradients_of_discriminator, self.discriminator.variables
        ))

    # Save generated images
    def generate_and_save_images(self, epoch, test_input):
        predictions = self.generator(test_input, training=False)
        images = predictions * 127.5 + 127.5

        image = np.zeros((self.size*8, self.size*8))
        for i in range(self.num_samples**2):
            image[
                int(i / self.num_samples)*self.size:
                int(i / self.num_samples + 1)*self.size,
                (i % self.num_samples)*self.size:
                (i % self.num_samples + 1)*self.size
            ] = images[i, :, :, 0]

        path = os.path.join(
            self.train_dir, 'image_at_epoch_{:04d}.png'.format(epoch)
        )
        plt.imsave(fname=path, arr=image, cmap='gray')

    # Main loop
    def train(self, dataset, epochs):
        # Restore the latest checkpoint
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir)
        )

        for epoch in range(epochs):
            start = time.time()

            # Update weights
            for images in dataset:
                self.train_step(images)

            # Save images for checking
            self.generate_and_save_images(epoch + 1, self.z_sample)

            # saving (checkpoint) the model every epoch
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            # Print time
            print('Time taken for training epoch {} is {} sec'.format(
                epoch + 1, time.time()-start
            ))

        # Save images for checking
        self.generate_and_save_images(epoch + 1, self.z_sample)

    def layers(self, images):
        number = images.shape[0]

        # Restore the latest checkpoint
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir)
        )

        # Discriminator layers for real images
        images = tf.convert_to_tensor(images, dtype=tf.float32)
        _ = self.discriminator(images, training=False)
        dr1 = self.discriminator.d1.numpy()
        dr2 = self.discriminator.d2.numpy()
        dr3 = self.discriminator.d3.numpy()

        # Random sample from latent space
        latent_code = tf.random_normal(
            [number, self.latent_size]
        )

        # Generator leayers
        g4 = self.generator(latent_code, training=False)
        g1 = self.generator.g1.numpy()
        g2 = self.generator.g2.numpy()
        g3 = self.generator.g3.numpy()

        # Discriminator layers for fake images
        _ = self.discriminator(g4, training=False)
        dg1 = self.discriminator.d1.numpy()
        dg2 = self.discriminator.d2.numpy()
        dg3 = self.discriminator.d3.numpy()

        return dr1, dr2, dr3, g1, g2, g3, g4.numpy(), dg1, dg2, dg3
