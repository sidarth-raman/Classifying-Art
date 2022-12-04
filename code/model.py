import numpy as np
import tensorflow as tf
from preprocess import get_paintings

def run_task(epochs=None, batch_size=None):
    import cnn
    # This will be passed in from preprocess
    data, label = get_paintings()
    # X0, Y0, X1, Y1 = data
    
    args = cnn.CNN_model()

    print(len(data))
    print(len(label))
    print(args)


    # Data passed in
    X0_sub = np.asarray(data[:1000], dtype=np.float32)
    Y0_sub = np.zeros(1000)
    X1_sub = np.asarray(data[5669:], dtype=np.float32)
    Y1_sub = np.zeros(5669)



    # Training model
    print("Model Training")
    history = args.model.fit(
        X0_sub, Y0_sub,
        epochs          = epochs,
        batch_size      = batch_size,
        validation_data = (X1_sub, Y1_sub),
    )

    return args.model


if __name__ == "__main__":
    run_task(epochs=5, batch_size=64)