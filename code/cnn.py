import pandas as pd
import numpy as np
import random
import tensorflow as tf

from "./preprocess.py" import get_paintings


def data_augmentation():
    # Choose data augmentation we want
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(scale=1 / 255),
            tf.keras.layers.Resizing(32, 32),
        ]
    )
    
    return None

def CNN_Resnet():
    train_input_shape = (224, 224, 3) #Used from online model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_input_shape)

    # Make each layer trainable in ResNet50
    for layer in base_model.layers:
        layer.trainable = True  

    # Choose a set of Dense Layers
    output = # Dense layers go here


    model = Model(inputs=base_model.input, outputs=output)

    optimizer = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, 
              metrics=['accuracy'])



    train = model.fit_generator(generator=train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                              validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                              epochs=50,
                              shuffle=True,
                              verbose=1,
                              callbacks=[reduce_lr],
                              use_multiprocessing=True,
                              workers=16,
                              class_weight=class_weights
                             )



def CNN_model():
    Conv2D = tf.keras.layers.Conv2D
    BatchNormalization = tf.keras.layers.BatchNormalization
    Dropout = tf.keras.layers.Dropout

    input_prep_fn = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(scale=1 / 255),
            tf.keras.layers.Resizing(32, 32),
        ]
    )
    output_prep_fn = tf.keras.layers.CategoryEncoding(
        num_tokens=10, output_mode="one_hot"
    )
    augment_fn = tf.keras.Sequential([tf.keras.layers.RandomRotation(factor=(-0.1, 0.1))])


    model = CustomSequential(
        [Conv2D(16, 3, strides=(2, 2), padding='valid'), tf.keras.layers.ReLU(), BatchNormalization(),
        Conv2D(32, 3, activation="leaky_relu"), BatchNormalization(),
        tf.keras.layers.Flatten(), tf.keras.layers.Dense(150, activation='leaky_relu'), tf.keras.layers.Dense(75, activation='leaky_relu'), tf.keras.layers.Dense(10, activation='softmax')],
        
        input_prep_fn=input_prep_fn,
        output_prep_fn=output_prep_fn,
        augment_fn=augment_fn
    )

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),  
        loss="categorical_crossentropy",  
        metrics=["categorical_accuracy"],
    )

    return SimpleNamespace(model=model, epochs=20, batch_size=100)

if __name__ == '__main__':
    get_paintings()
