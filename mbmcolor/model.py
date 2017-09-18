import os

import numpy as np
from keras.layers import (Input, Conv2D, Conv2DTranspose, Dense,
                          Activation, BatchNormalization, Concatenate, Flatten)
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, TerminateOnNaN

from mbmcolor.losses import build_mbm_log_likelihood


class MBMColorNet(object):
    # TODO: Write me a documentation

    def __init__(self,
                 input_shape, n_components,
                 nb_filters_per_layer, kernel_size, padding, batch_normalization,
                 optimizer='adam', es_patience=10,
                 model_path='/tmp/', weights_name_format='weights.{epoch:02d}-{val_loss:.6f}.hdf5',
                 histogram_freq=0):

        self.input_shape = input_shape
        self.n_components = n_components

        self.nb_filters_per_layer = nb_filters_per_layer
        self.kernel_size = kernel_size
        self.padding = padding
        self.batch_normalization = batch_normalization

        self.n_dense_prior = 128

        self.model_path = model_path
        self.weights_name_format = weights_name_format
        self.optimizer = optimizer
        self.es_patience = es_patience
        self.histogram_freq = histogram_freq

        self._loss = build_mbm_log_likelihood(input_shape, n_components)

        self._input_layer = None
        self._output_layer = None
        self._model = None

    def _build_callbacks(self):
        """Builds callbacks for training model.
        """
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        checkpointer = ModelCheckpoint(filepath=os.path.join(self.model_path, self.weights_name_format),
                                       monitor='val_loss', save_best_only=True, save_weights_only=True)

        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.es_patience)

        epoch_logger = CSVLogger(os.path.join(self.model_path, 'epoch_log.csv'))

        tensorboard_path = os.path.join(self.model_path, 'tensorboard')
        tensorboard = TensorBoard(log_dir=tensorboard_path,
                                  histogram_freq=self.histogram_freq, write_grads=(self.histogram_freq > 0))

        terminator = TerminateOnNaN()

        return [checkpointer, early_stopper, epoch_logger, tensorboard, terminator]

    def _custom_conv2d(self, layer, nb_filters, strides):
        # TODO: Write me a documentation

        layer = Conv2D(nb_filters, kernel_size=self.kernel_size, strides=strides, padding=self.padding)(layer)

        if self.batch_normalization:
            layer = BatchNormalization(mode=0, axis=1)(layer)

        layer = Activation('relu')(layer)

        return layer

    def _custom_conv2dtranspose(self, layer, nb_filters, strides):

        layer = Conv2DTranspose(nb_filters, kernel_size=self.kernel_size, strides=strides, padding=self.padding)(layer)

        if self.batch_normalization:
            layer = BatchNormalization(mode=0, axis=1)(layer)

        layer = Activation('relu')(layer)

        return layer

    def _build_layers(self):
        """Builds all layers
        """
        height, width = self.input_shape
        self._input_layer = Input(shape=(height, width, 1))

        # ------ Define convolutional layers
        layer = self._input_layer

        for i, nb_filters in enumerate(self.nb_filters_per_layer):
            layer = self._custom_conv2d(layer, nb_filters, (1, 1))
            layer = self._custom_conv2d(layer, nb_filters, (1, 1))
            layer = self._custom_conv2d(layer, nb_filters, (2, 2))

        bottleneck = layer

        for i, nb_filters in enumerate(self.nb_filters_per_layer[::-1]):
            layer = self._custom_conv2dtranspose(layer, nb_filters, (1, 1))
            layer = self._custom_conv2dtranspose(layer, nb_filters, (2, 2))
            layer = self._custom_conv2dtranspose(layer, nb_filters, (1, 1))

        # ------ Output layers that parametrize a Multivariate Bernoulli Mixture Density.
        # Means
        mu = Conv2D(self.n_components, kernel_size=self.kernel_size, padding='same')(layer)
        mu = Flatten()(mu)

        # Priors
        # First squeeze the filters with a convolution before flattening
        prior = Conv2D(1, kernel_size=self.kernel_size, padding='same')(bottleneck)
        prior = Flatten()(prior)
        prior = Dense(self.n_dense_prior, activation='relu')(prior)
        prior = Dense(self.n_components, activation='softmax')(prior)

        self._output_layer = Concatenate(axis=-1)([prior, mu])

    def _compile_model(self):
        self._model.compile(optimizer=self.optimizer, loss=self._loss)

    def build_model(self):
        """Build the model.
        """
        self._build_layers()
        self._model = Model(self._input_layer, self._output_layer)
        self._compile_model()

    def summary(self):
        """Print summary of model to stdout.
        """
        self._model.summary()

    def load_from_file(self, path):
        """Load the model from a file.

        Args:
            path (str): Path to the model file.
        """
        self._model = load_model(path)
        print('Loaded model from file.')

    def fit_generator(self, train_generator, steps_per_epoch, epochs,
                      validation_generator, validation_steps):
        """Train the model on data yielded batch-by-batch by a generator.

        Args:
            train_generator: A data generator that yields (x, y) tuples of training data/labels.
            steps_per_epoch: Steps (number of batches) per epoch.
            epochs: Number of epochs.
            validation_generator: A data generator that yields (x, y) tuples of validation data/labels.
            validation_steps: Validation steps (number of batches).

        Returns:
            Keras History object with history of training losses.
       """

        callbacks = self._build_callbacks()

        history = self._model.fit_generator(generator=train_generator,
                                            steps_per_epoch=steps_per_epoch,
                                            epochs=epochs,
                                            callbacks=callbacks,
                                            validation_data=validation_generator,
                                            validation_steps=validation_steps)

        return history

    @staticmethod
    def _check_input_array(x):
        if x.dtype != 'float32':
            raise ValueError('Input array must be of type float32.')
        if x.ndim != 4:
            raise ValueError('Input array must have exactly 4 dimensions: (samples, height, width, channels)')
        if x.shape[-1] != 1:
            raise ValueError('Size of last dimension of input array must be exactly 1.')

    def predict(self, array):
        """Predict from model.

        Args:
            array (numpy.ndarray): Input array of shape (samples, height, width, 1) and type float32.

        Returns:
            Array of the same shape and type as the input containing the colorized image.
            Each pixel y of the colorized image is predicted as:
                y = mu[K] where K = argmax(k)(prior[k]), k is the index of each component in the mixture.
        """
        self._check_input_array(array)

        pred = self._model.predict(array)

        m = self.n_components
        _, height, width, _ = array.shape
        splits = [m]  # numpy.split expect locations, not sizes

        # Get MBM parameters
        # Parameters are concatenated along the second axis
        prior, mu = np.split(pred, axis=1, indices_or_sections=splits)

        mu = np.reshape(mu, (-1, height, width, m))

        which = prior.argmax(axis=1)
        sample, height, width = np.indices(array.shape[:-1])

        return np.expand_dims(mu[sample, height, width, which], axis=3)
