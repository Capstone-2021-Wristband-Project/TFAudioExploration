from DataProcessing import training_prep
from TFLightConversion import tflite_convert_and_save_to_file
import tensorflow as tf

input_model_location = "model/audio_classify_aligned_test"
data_set_location = "data/speech_commands_aligned"
output_model_location = "model/audio_classify_aligned.tflite"


def main():
    datasets, commands = training_prep(data_dir=data_set_location)
    model = tf.keras.models.load_model(input_model_location)
    tflite_convert_and_save_to_file(model, datasets[0], output_model_location)


if __name__ == "__main__":
    main()
