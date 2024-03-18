import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, Rescaling


def make_model(input_shape, num_note_names: int, num_lengths: int):
    inputs = tf.keras.Input(shape=input_shape)
    x = Rescaling(1.0 / 255)(inputs)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(48, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((4, 4))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = GlobalMaxPooling2D()(x)
    x = Dense(64, activation='relu')(x)
    note_names = Dense(num_note_names, name='note', activation='softmax')(x)
    lengths = Dense(num_lengths, name='length', activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs={'note': note_names, 'length': lengths})


if __name__ == '__main__':
    model = make_model((140, 80, 1), 8)
    model.summary()
