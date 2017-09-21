import os
import sys
import time

import numpy as np
from keras.optimizers import Adam

from mbmcolor.datagen import ImagePreprocessor
from mbmcolor.model import MBMColorNet


def main():

    # Dataset parameters
    checkpoints_path = '/work/data/ilsvrc2012/analysis/checkpoints/'
    dataset_path = '/work/data/ilsvrc2012/raw/train/'

    # Model parameters
    input_shape = (224, 224)  # Typical ImageNet size
    n_components = 10
    nb_filters_per_layer = (8, 16)
    kernel_size = (3, 3)
    padding = 'same'
    batch_normalization = False
    optimizer = Adam(lr=0.001)
    es_patience = 10
    histogram_freq = 0

    # Training parameters
    mean_shift = 0.5
    norm_factor = 1.0
    alexnet_resize = True
    batch_size = 64
    steps_per_epoch = 128
    epochs = 150
    validation_steps = 32

    # Define model
    time_string = time.strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(checkpoints_path, time_string)  # dir for model files and log

    # Build model
    model = MBMColorNet(input_shape, n_components,
                        nb_filters_per_layer, kernel_size, padding, batch_normalization,
                        optimizer=optimizer, es_patience=es_patience,
                        model_path=model_path,
                        histogram_freq=histogram_freq)

    model.build_model()

    model.summary()

    # Create data generators
    preprocessor = ImagePreprocessor(img_size=input_shape,
                                     mean_shift=mean_shift, norm_factor=norm_factor, alexnet_resize=alexnet_resize)

    n_train_imgs = epochs * steps_per_epoch * batch_size
    train_gen = preprocessor.build_image_generator(dataset_path, batch_size=batch_size,
                                                   n_imgs=n_train_imgs)

    n_val_imgs = epochs * validation_steps * batch_size
    val_gen = preprocessor.build_image_generator(dataset_path, batch_size=batch_size,
                                                 n_imgs=n_val_imgs)

    # Train model
    if histogram_freq > 0:
        # The TensorBoard callback cannot make histograms if the validation data comes from a generator
        # Thus, we have to burn the generator to make a static validation set
        val_data_x = []
        val_data_y = []

        for i, (x, y) in enumerate(val_gen):
            if i == validation_steps:
                break
            val_data_x.append(x)
            val_data_y.append(y)

        val_data = (np.concatenate(val_data_x, axis=0),
                    np.concatenate(val_data_y, axis=0))
    else:
        val_data = val_gen

    model.fit_generator(train_generator=train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
                        validation_generator=val_data, validation_steps=validation_steps)

    return 0

if __name__ == '__main__':
    sys.exit(main())
