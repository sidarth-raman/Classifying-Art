import pandas as pd
import numpy as np
import random
import tensorflow as tf

from "./preprocess.py" import get_paintings


def data_augmentation(): 


    Conv2D = conv_ns.Conv2D
    BatchNormalization = norm_ns.BatchNormalization
    Dropout = drop_ns.Dropout
    Conv2D_manual = man_conv_ns.Conv2D

    input_prep_fn = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(scale=1 / 255),
            tf.keras.layers.Resizing(32, 32),
        ]
    )
    output_prep_fn = tf.keras.layers.CategoryEncoding(
        num_tokens=10, output_mode="one_hot"
    )



    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size
    print("Total number of batches =", STEP_SIZE_TRAIN, "and", STEP_SIZE_VALID)


def cnn(): 
    # base_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_input_shape)

    for layer in base_model.layers:
        layer.trainable = True

    model = CustomSequential(
        [Conv2D_manual(16, 3, strides=(2, 2), padding='valid'), tf.keras.layers.ReLU(), BatchNormalization(),
        Conv2D(32, 3, activation="leaky_relu"), BatchNormalization(),
        tf.keras.layers.Flatten(), tf.keras.layers.Dense(150, activation='leaky_relu'), tf.keras.layers.Dense(75, activation='leaky_relu'), tf.keras.layers.Dense(10, activation='softmax')],
        input_prep_fn=input_prep_fn,
        output_prep_fn=output_prep_fn,
        augment_fn=augment_fn
        ## Take a look at the constructor for CustomSequential to see if you
        ## might need to pass in the necessary preparation functions...
    )

    output = Dense(n_classes, activation='softmax')(X)

    model = Model(inputs=base_model.input, outputs=output)
    optimizer = Adam(lr=0.0001)
    
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, 
              metrics=['accuracy'])


if __name__ == '__main__':
    get_paintings()
