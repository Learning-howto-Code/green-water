import tensorflow as tf
import argparse
import os
import tempfile

def convert(model_path, output_path, quantize=False):
    is_keras = model_path.lower().endswith((".h5", ".keras"))

    if is_keras:
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Remove/skip data augmentation layers by rebuilding model without them
        # Build a new model that skips augmentation
        new_model = _strip_augmentation(model)
        if new_model is not None:
            model = new_model
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

    # Broader op set to avoid MLIR inference crashes; enable legacy converter if present.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    if hasattr(converter, "experimental_new_converter"):
        converter.experimental_new_converter = False

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        tflite_model = converter.convert()
    except Exception as e:
        # Fallback: try building concrete function directly
        if not is_keras:
            raise
        
        if not model.inputs:
            raise
        
        shape_obj = model.inputs[0].shape
        if hasattr(shape_obj, "as_list"):
            shape_list = shape_obj.as_list()
        else:
            shape_list = list(shape_obj)
        input_shape = [d if d is not None else 1 for d in shape_list]
        input_dtype = getattr(model.inputs[0], "dtype", None) or tf.float32

        run_model = tf.function(lambda x: model(x, training=False))
        concrete_func = run_model.get_concrete_function(
            tf.TensorSpec(input_shape, input_dtype)
        )

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        if hasattr(converter, "experimental_new_converter"):
            converter.experimental_new_converter = False
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model to: {output_path}")


def _strip_augmentation(model):
    """Remove data augmentation layers from model for inference."""
    try:
        aug_indices = []
        for i, layer in enumerate(model.layers):
            if 'data_augmentation' in layer.name.lower() or \
               'random' in layer.name.lower():
                aug_indices.append(i)
        
        if not aug_indices:
            return None
        
        # Get inputs and outputs
        inputs = model.inputs[0]
        x = inputs
        
        # Skip augmentation layers
        skip_set = set(aug_indices)
        for i, layer in enumerate(model.layers):
            if i not in skip_set and i > 0:  # Skip input layer
                x = layer(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x)
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model to .tflite")
    parser.add_argument("model_path", help="Path to model (.h5/.keras or SavedModel folder)")
    parser.add_argument("output_path", help="Output .tflite path")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization")

    args = parser.parse_args()

    convert(args.model_path, args.output_path, args.quantize)
