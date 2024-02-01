# import tensorflow as tf
import tensorflow.lite as lite

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
# print(gpus)
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

model = tf.keras.models.load_model(
    "./facial_recognition_cnn_model.h5"
)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("lite_facial_recognition_model2.tflite", "wb").write(tflite_model)

