from keras import backend as K

_EPSILON = 10e-8


def build_mbm_log_likelihood(input_shape, m):
    """Build log-likelihood loss for Multivariate Bernoulli Mixture Density.

    Args:
        input_shape (tuple): Shape of input image.
        m (int): Number of components in the mixture.

    Returns:
        Loss function.
    """

    # TODO: Add check for integers in input_shape
    if len(input_shape) != 2 or any([d <= 0 for d in input_shape]):
        raise ValueError('input_shape must be 2D with positive dimensions.')
    if not (m > 0 and isinstance(m, int)):
        raise ValueError('m must be a positive integer.')

    def _mbm_log_likelihood(y_true, y_pred):
        """Log-likelihood loss for Multivariate Bernoulli Mixture Density.
        Currently only supports TensorFlow backend.

        Args:
            y_true (tensor): A tensor of shape (samples, height*width*2) with the flattened true values.
                (The factor of 2 corresponds to the ab channels)
            y_pred (tensor): Tensor of shape (samples, m*(height*width*2 + 1)), where m is the number
                of mixture components.
                The second dimension encodes the following parameters (in that order):
                1) m priors (outputs of a softmax activation layer)
                2) m*height*width*2 means (outputs of a sigmoid activation layer)

        Returns:
            Negative log-likelihood of each sample.
        """
        height, width = input_shape
        c = height * width * 2  # Number of output dimensions
        splits = [m, m * c]

        # Get MBM parameters
        # Parameters are concatenated along the second axis
        # tf.split expect sizes, not locations
        prior, mu = K.tf.split(y_pred, num_or_size_splits=splits, axis=1)

        y_true = K.tile(K.expand_dims(y_true, axis=2), [1, 1, m])
        mu = K.reshape(mu, [-1, c, m])  # -1 is for the sample dimension
        prob = - K.binary_crossentropy(y_true, mu, from_logits=False)  # Undo the negative inside binary crossentropy
        prob = K.sum(prob, axis=1)

        log_prior = K.log(K.clip(prior, _EPSILON, 1 - _EPSILON))
        exponent = log_prior + prob

        return -K.logsumexp(exponent, axis=1)

    return _mbm_log_likelihood
