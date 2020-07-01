import keras
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class CAE:
    def create_model(self, embedding_dim, input_shape=(60, 60, 1), filters=[16, 64, 128]):
        if input_shape[0] % 8 == 0:
            pad3 = "same"
        else:
            pad3 = "valid"

        # encoder
        inputs = Input(shape=input_shape)
        # layer1
        encoded = Conv2D(filters[0], 5, strides=2, padding="same", activation="relu")(inputs)
        # layer2
        encoded = Conv2D(filters[1], 5, strides=2, padding="same", activation="relu")(encoded)
        # layer3
        encoded = Conv2D(filters[2], 3, strides=2, padding=pad3, activation="relu")(encoded)

        encoded = Flatten()(encoded)
        encoded = Dense(embedding_dim)(encoded)

        self.encoder = Model(inputs, encoded)

        # decoder
        decoded = Dense(filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation="relu")(encoded)
        decoded = Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2]))(decoded)
        # layer1
        decoded = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation="relu")(decoded)
        # layer2
        decoded = Conv2DTranspose(filters[0], 5, strides=2, padding="same", activation="relu")(decoded)
        # layer3
        decoded = Conv2DTranspose(input_shape[2], 5, strides=2, padding="same", activation="sigmoid")(decoded)

        self.autoencoder = Model(inputs, decoded)
        self.autoencoder.compile(optimizer=Adam(), loss="binary_crossentropy")

    def train(self, x_train, epochs, batch_size):
        self.autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True)

    def save_autoencoder(self, path):
        self.autoencoder.save(path)

    def save_encoder(self, path):
        self.encoder.save(path)
