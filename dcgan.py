import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import time

tf.enable_eager_execution()

SIZE = 32
DEPTH = 32
CHANNELS = 1

BUFFER_SIZE = 70000
BATCH_SIZE = 256
EPOCHS = 50
LATENT_SIZE = 100
NUM_SAMPLES = 8
LEARNING_RATE = 0.0001
BETA1 = 0.5

CHECKPOINT_DIR = 'checkpoints/dcgan'
TRAIN_DIR = 'train/dcgan'

# We'll re-use this random vector used to seed the generator so
# it will be easier to see the improvement over time.
Z_SAMPLE = tf.random_normal([NUM_SAMPLES**2, LATENT_SIZE])


def make_generator_model():
    model = tf.keras.Sequential()

    # Layer 1
    model.add(tf.keras.layers.Dense(
        int(4*DEPTH*int(SIZE/8)*int(SIZE/8)), use_bias=False,
        input_shape=(LATENT_SIZE,)
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((int(SIZE/8), int(SIZE/8), 4*DEPTH)))

    # Layer 2
    model.add(tf.keras.layers.Conv2DTranspose(
        2*DEPTH, (4, 4), strides=(2, 2), padding='same', use_bias=False
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # Layer 3
    model.add(tf.keras.layers.Conv2DTranspose(
        DEPTH, (4, 4), strides=(2, 2), padding='same', use_bias=False
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # Layer 4
    model.add(tf.keras.layers.Conv2DTranspose(
        1, (4, 4), strides=(2, 2), padding='same', use_bias=False,
        activation='tanh'
    ))

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()

    # Layer 1
    model.add(tf.keras.layers.Conv2D(
        DEPTH, (4, 4), strides=(2, 2), padding='same'
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # Layer 2
    model.add(tf.keras.layers.Conv2D(
        2*DEPTH, (4, 4), strides=(2, 2), padding='same'
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # Layer 3
    model.add(tf.keras.layers.Conv2D(
        4*DEPTH, (4, 4), strides=(2, 2), padding='same'
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Flatten())

    # Layer 4
    model.add(tf.keras.layers.Dense(1))

    return model


def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(
        tf.ones_like(generated_output), generated_output
    )


def discriminator_loss(real_output, generated_output):
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


def train_step(images):
    # generating latent code from a normal distribution
    latent_code = tf.random_normal([BATCH_SIZE, LATENT_SIZE])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(latent_code, training=True)

        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.variables
        )

    generator_optimizer.apply_gradients(zip(
        gradients_of_generator, generator.variables
    ))
    discriminator_optimizer.apply_gradients(zip(
        gradients_of_discriminator, discriminator.variables
    ))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for images in dataset:
            train_step(images)

        generate_and_save_images(
            generator, epoch + 1, Z_SAMPLE
        )

        # saving (checkpoint) the model every epoch
        path = os.path.join(CHECKPOINT_DIR, "ckpt")
        checkpoint.save(file_prefix=path)

        print('Time taken for epoch {} is {} sec'.format(
            epoch + 1, time.time()-start
        ))

    # generating after the final epoch
    generate_and_save_images(generator, epochs, Z_SAMPLE)


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    images = predictions * 127.5 + 127.5

    image = np.zeros((SIZE*8, SIZE*8))
    for i in range(NUM_SAMPLES**2):
        image[
            int(i / NUM_SAMPLES)*SIZE:int(i / NUM_SAMPLES + 1)*SIZE,
            (i % NUM_SAMPLES)*SIZE:(i % NUM_SAMPLES + 1)*SIZE
        ] = images[i, :, :, 0]

    path = os.path.join(TRAIN_DIR, 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.imsave(fname=path, arr=image, cmap='gray')


# Process input
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)\
    .astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)\
    .astype('float32')
images = np.zeros((
    train_images.shape[0]+test_images.shape[0], SIZE, SIZE, CHANNELS
)).astype('float32')
images[:train_images.shape[0], 2:30, 2:30] = train_images
images[train_images.shape[0]:, 2:30, 2:30] = test_images
images = (images - 127.5) / 127.5
train_dataset = tf.data.Dataset.from_tensor_slices(images)\
    .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.train.AdamOptimizer(
    learning_rate=LEARNING_RATE, beta1=BETA1
)
discriminator_optimizer = tf.train.AdamOptimizer(
    learning_rate=LEARNING_RATE, beta1=BETA1
)


checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer, generator=generator,
    discriminator=discriminator
)

# restoring the latest checkpoint in CHECKPOINT_DIR
checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
train_step = tf.contrib.eager.defun(train_step)
train(train_dataset, EPOCHS)
