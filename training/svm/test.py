import os
import numpy as np
import tensorflow as tf

TFLITE_PATH = input("tflite model path: ").strip()
OUTPUT_PATH = os.path.splitext(TFLITE_PATH)[0] + ".keras"

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = tuple(input_details[0]['shape'][1:])
output_shape = tuple(output_details[0]['shape'][1:])

print(f"Input shape: {input_shape}")
print(f"Output shape: {output_shape}")


class TFLiteLayer(tf.keras.layers.Layer):
    def __init__(self, tflite_path, output_shape_tuple, **kwargs):
        super().__init__(**kwargs)
        self.tflite_path = tflite_path
        self._output_shape_tuple = output_shape_tuple
        self._interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self._interpreter.allocate_tensors()
        self._in_idx = self._interpreter.get_input_details()[0]['index']
        self._out_idx = self._interpreter.get_output_details()[0]['index']

    def call(self, x):
        def run(batch):
            results = []
            for sample in batch:
                self._interpreter.set_tensor(self._in_idx, np.expand_dims(sample, axis=0))
                self._interpreter.invoke()
                results.append(self._interpreter.get_tensor(self._out_idx)[0])
            return np.array(results, dtype=np.float32)

        return tf.py_function(run, [x], tf.float32)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self._output_shape_tuple

    def get_config(self):
        config = super().get_config()
        config.update({"tflite_path": self.tflite_path, "output_shape_tuple": self._output_shape_tuple})
        return config


inputs = tf.keras.Input(shape=input_shape)
outputs = TFLiteLayer(TFLITE_PATH, output_shape)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.save(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
