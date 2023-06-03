from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Concatenate, Conv2D, \
    MaxPooling2D, UpSampling2D, Activation, Multiply, Add
import numpy as np

class AttentionUNet(Model):
    """
    2D Attention U-Net implementation with batch normalization.

    See original paper at https://arxiv.org/abs/1804.03999
    """
    def __init__(self, n_classes, img_rows=None, img_cols=None, dim=None, n_channels=1,
                 depth=4, activation="relu", kernel_size=3, padding="same",
                 complexity_factor=1, flatten_output=False, **kwargs):
        """
        n_classes (int):
            The number of classes to model, gives the number of filters in the
            final 1x1 conv layer.
        img_rows, img_cols (int, int):
            Image dimensions. Note that depending on image dims cropping may
            be necessary. To avoid this, use image dimensions DxD for which
            D * (1/2)^n is an integer, where n is the number of (2x2)
            max-pooling layers; in this implementation 4.
            For n=4, D \in {..., 192, 208, 224, 240, 256, ...} etc.
        dim (int):
            img_rows and img_cols will both be set to 'dim'
        n_channels (int):
            Number of channels in the input image.
        depth (int):
            Number of conv blocks in encoding layer (number of 2x2 max pools)
            Note: each block doubles the filter count while halving the spatial
            dimensions of the features.
        activation (string):
            Activation function for convolution layers
        kernel_size (int):
            Kernel size for convolution layers
        padding (string):
            Padding type ('same' or 'valid')
        complexity_factor (int/float):
            Use int(N * sqrt(complexity_factor)) number of filters in each
            2D convolution layer instead of default N.
        flatten_output (bool):
            Flatten the output to array of shape [batch_size, -1, n_classes]
        """
        super(AttentionUNet, self).__init__()
        if not ((img_rows and img_cols) or dim):
            raise ValueError("Must specify either img_rows and img_cols or dim")
        if dim:
            img_rows, img_cols = dim, dim

        self.img_shape = (img_rows, img_cols, n_channels)
        self.n_classes = n_classes
        self.cf = np.sqrt(complexity_factor)
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.depth = depth
        self.flatten_output = flatten_output

        # Build model and initialize base Keras Model class
        super().__init__(*self.init_model())

    def _create_encoder(self, inputs, filters):
        skip_connections = []
        x = inputs
        for i in range(self.depth):
            conv1 = Conv2D(int(filters * self.cf), self.kernel_size,
                           activation=self.activation, padding=self.padding)(x)
            conv2 = Conv2D(int(filters * self.cf), self.kernel_size,
                           activation=self.activation, padding=self.padding)(conv1)
            x = BatchNormalization()(conv2)
            skip_connections.append(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            filters *= 2
        return x, skip_connections

    def _create_bottom(self, x, filters):
        conv1 = Conv2D(int(filters * self.cf), self.kernel_size,
                       activation=self.activation, padding=self.padding)(x)
        conv2 = Conv2D(int(filters * self.cf), self.kernel_size,
                       activation=self.activation, padding=self.padding)(conv1)
        x = BatchNormalization()(conv2)
        return x

    def _create_decoder(self, x, skip_connections, filters):
        for i in reversed(range(self.depth)):
            conv1 = Conv2D(int(filters * self.cf), self.kernel_size,
                           activation=self.activation, padding=self.padding)(x)
            conv2 = Conv2D(int(filters * self.cf), self.kernel_size,
                           activation=self.activation, padding=self.padding)(conv1)
            x = BatchNormalization()(conv2)
            x = UpSampling2D(size=(2, 2))(x)
            x = Concatenate(axis=-1)([x, skip_connections[i]])

            attention = self._create_attention_block(x, filters)

            x = Multiply()([attention, x])
            x = Conv2D(int(filters * self.cf), self.kernel_size,
                       activation=self.activation, padding=self.padding)(x)
            x = Conv2D(int(filters * self.cf), self.kernel_size,
                       activation=self.activation, padding=self.padding)(x)
            x = BatchNormalization()(x)
            filters //= 2
        return x

    def _create_attention_block(self, x, filters):
        g = Conv2D(filters, kernel_size=1, padding=self.padding)(x)
        g = BatchNormalization()(g)
        x = Conv2D(filters, kernel_size=1, padding=self.padding)(x)
        x = BatchNormalization()(x)
        phi = Activation('relu')(Add()([g, x]))

        f = Conv2D(1, kernel_size=1, padding=self.padding)(phi)
        f = Activation('sigmoid')(f)

        attention = Multiply()([x, f])
        return attention

    def init_model(self):
        inputs = Input(shape=self.img_shape)

        filters = 64
        encoding, skip_connections = self._create_encoder(inputs, filters)
        bottom = self._create_bottom(encoding, filters)
        decoding = self._create_decoder(bottom, skip_connections, filters)

        outputs = Conv2D(self.n_classes, kernel_size=1, activation='softmax')(decoding)
        if self.flatten_output:
            outputs = Reshape([-1, self.n_classes])(outputs)

        return [inputs], [outputs]
