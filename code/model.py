import numpy as np
import tensorflow as tf

def run_task(data, task, subtask="all", epochs=None, batch_size=None):
    import cnn     

    # This will be passed in from preprocess
    X0, Y0, X1, Y1 = data
    
    args = cnn.CNN_model

    # Data passed in
    X0_sub, Y0_sub = X0, Y0
    X1_sub, Y1_sub = X1, Y1

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
    data = get_data()
    run_task(data, args.task, args.subtask)