def CNN(X_train, y_train, X_test, y_test, epoc=10, output=10, filter_count=10):
    import tensorflow as tf

    from keras import Sequential
    from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    import numpy as np
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from keras.preprocessing.image import ImageDataGenerator


    gpus = tf.config.list_physical_devices("GPU")
    print(gpus)
    if gpus:
        try:
            for gpu in gpus:
                print(gpu)
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=2000)]
                )
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

            # Currently, memory growth needs to be the same across GPUs

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # DATA AUGMENTATION
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
    )

    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    model = Sequential()

    # 1st convolutional Layer
    model.add(Conv2D(128, (3, 3), input_shape=X_train.shape[1:], padding="SAME",activation = tf.keras.layers.LeakyReLU(alpha=0.01)))
    # model.add()
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # 2st convolutional Layer
    model.add(Conv2D(96, (3, 3), padding="SAME"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # 3rd convolutional Layer
    model.add(Conv2D(64, (3, 3), padding="SAME"))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))
    # model.add(MaxPooling2D(pool_size=(2, 2)))


    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(250, activation="relu", use_bias=True))
    model.add(Dropout(0.25))
    model.add(Dense(150, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation="relu", use_bias=True))
    model.add(Dropout(0.15))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(output, activation="softmax"))

    print(model.summary())
    print(len(X_train))

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.001
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )


    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # history = model.fit(
    #     X_train,
    #     y_train,
    #     epochs=epoc,
    #     shuffle="True",
    #     validation_split=0.2,
    #     callbacks=[reduce_lr, early_stopping],
    # )
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=32),
        epochs=epoc,
        shuffle="True",
        validation_data=valid_datagen.flow(X_test, y_test),
        # validation_split=0.2,
        callbacks=[reduce_lr, early_stopping],
        batch_size=5,
    )

    X_test = X_test / 255.0
    result = model.evaluate(X_test, y_test, batch_size=10)

    print("test Loss : ", result[0], " , Test Accuracy : ", result[1])

    return model

