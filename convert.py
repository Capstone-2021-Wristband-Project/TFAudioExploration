import tensorflow as tf


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("model/audio_classify_aligned_test") # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the model.
with open('model/audio_classify_aligned_test.tflite', 'wb') as f:
  f.write(tflite_model)

print("On Unix systems, run the following command to convert the tflite model to a C array:")
print("$ xxd -i model/audio_classify_aligned_test.tflite > model/audio_classify_aligned_test.cpp")