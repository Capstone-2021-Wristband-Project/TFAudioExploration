import tensorflow as tf


def representative_dataset(fit_ds: tf.data.Dataset):
    # Not sure if we can just pull in test_ds like this
    def data_set_generator():
        ds = fit_ds.batch(1)
        for data, label in ds.take(64):
            print(data)
            yield [data]
    return data_set_generator


def tflite_convert(model, fit_ds: tf.data.Dataset):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Dynamic range quantization
    # https://www.tensorflow.org/lite/performance/post_training_quantization#dynamic_range_quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Take a representative dataset and perform full integer only quantization
    # https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization
    # https://www.tensorflow.org/lite/performance/post_training_quantization#integer_only

    converter.representative_dataset = representative_dataset(fit_ds)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8

    return converter.convert()


def tflite_convert_and_save_to_file(
        model: tf.keras.models.Model,
        fit_ds: tf.data.Dataset,
        output_location: str
):
    tflite_model = tflite_convert(model, fit_ds)

    # Save the model.
    with open(output_location, 'wb') as f:
        f.write(tflite_model)

    print("On Unix systems, run the following command to convert the tflite model to a C array:")
    print("$ xxd -i model/audio_classify_aligned_test.tflite > model/audio_classify_aligned_test.cpp")
