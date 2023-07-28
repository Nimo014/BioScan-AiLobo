import tensorflow as tf

# Load the saved model
loaded_model = tf.keras.models.load_model('./saved_model_h5/saved_model_openpose_ear_v1.h5',compile=False)

# Print the model summary
loaded_model.summary()
