import tensorflow as tf
import time
import os


class Classifier(object):
    def __init__(self):
        # Global constants
        self.classes = 10
        self.depth = 32
        self.checkpoint_dir = os.path.join('checkpoint', 'classifier')
        self.learning_rate = 0.0001
        self.beta1 = 0.5

        # Choose optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, beta1=self.beta1
        )

        # Make model
        self.classify = self.make_classifier_model(self.depth)

        # Setup checkpoint
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(model=self.classify)

        # Speed up training
        self.train_step = tf.contrib.eager.defun(self.train_step)

    # Model architecture
    def make_classifier_model(self, depth):
        model = tf.keras.Sequential()

        # Layer 1: SIZE x SIZE x CHANNELS -> SIZE/2 x SIZE/2 x DEPTH
        model.add(tf.keras.layers.Conv2D(
            self.depth, (4, 4), strides=(2, 2), padding='same'
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        # Layer 2: SIZE/2 x SIZE/2 x DEPTH -> SIZE/4 x SIZE/4 x 2*DEPTH
        model.add(tf.keras.layers.Conv2D(
            2*self.depth, (4, 4), strides=(2, 2), padding='same'
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        # Layer 3: SIZE/4 x SIZE/4 x 2*DEPTH -> SIZE/8 x SIZE/8 x 4*DEPTH
        model.add(tf.keras.layers.Conv2D(
            4*self.depth, (4, 4), strides=(2, 2), padding='same'
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())

        # Reshape: SIZE/8 x SIZE/8 x 4*DEPTH -> SIZE*SIZE*DEPTH/16
        model.add(tf.keras.layers.Flatten())

        # Layer 4: SIZE*SIZE*DEPTH/16 -> CLASSES
        model.add(tf.keras.layers.Dense(self.classes))

        return model

    # Cross entropy
    def classifier_loss(self, real_labels, predictions):
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=tf.one_hot(indices=real_labels, depth=self.classes),
            logits=predictions
        )

        return loss

    # Update weights
    def train_step(self, images, labels):
        with tf.GradientTape() as class_tape:
            # Predict labels
            predictions = self.classify(images, training=True)

            # Compute loss
            loss = self.classifier_loss(labels, predictions)

            # Compute gradients
            gradients = class_tape.gradient(
                loss, self.classify.variables
            )

        # Update weights
        self.optimizer.apply_gradients(zip(
            gradients, self.classify.variables
        ))

    # Test accuracy
    def test_step(self, images, labels):
        predictions = self.classify(images, training=False)
        gen_labels = tf.argmax(input=predictions, axis=1)
        accuracy = tf.contrib.metrics.accuracy(
            labels=labels, predictions=gen_labels
        )

        return accuracy, tf.size(labels, out_type=tf.float32)

    # Main loop
    def train(self, train_dataset, test_dataset, epochs):
        # Restore the latest checkpoint
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir)
        )

        for epoch in range(epochs):
            start = time.time()

            # Update weights
            for images, labels in train_dataset:
                self.train_step(images, labels)

            # saving (checkpoint) the model every epoch
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            # Print time
            print('Time taken for training epoch {} is {} sec'.format(
                epoch + 1, time.time()-start
            ))

            # Test accuracy
            accuracy = 0
            total = 0
            for images, labels in test_dataset:
                current, weight = self.test_step(images, labels)
                accuracy += weight * current
                total += weight

            accuracy /= total

            # Print accuracy
            print('Accuracy at training epoch {} is {}'.format(
                epoch+1, accuracy.numpy()
            ))

    # External predictions
    def predict(self, images):
        # Restore the latest checkpoint
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir)
        )

        # Predict classes
        predictions = self.classify(images, training=False)
        gen_labels = tf.argmax(input=predictions, axis=1)

        return gen_labels.numpy()
