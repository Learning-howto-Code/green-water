import tensorflow as tf
import argparse

def convert(model_path, output_path, quantize=False):
    model = tf.keras.models.load_model(model_path, compile=False)

    new_model = _strip_augmentation(model)
    if new_model is not None:
        model = new_model

    # Use concrete function to avoid MLIR inference crashes
    shape_list = [d if d is not None else 1 for d in model.inputs[0].shape]
    run_model = tf.function(lambda x: model(x, training=False))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(shape_list, tf.float32)
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

    print(f"Saved: {output_path}")


def _strip_augmentation(model):
    """Rebuild model without data augmentation layers for inference."""
    try:
        aug_indices = {
            i for i, layer in enumerate(model.layers)
            if 'augmentation' in layer.name.lower() or 'random' in layer.name.lower()
        }
        if not aug_indices:
            return None

        x = model.inputs[0]
        for i, layer in enumerate(model.layers):
            if i not in aug_indices and i > 0:
                x = layer(x)

        return tf.keras.Model(inputs=model.inputs[0], outputs=x)
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to .keras or .h5 file")
    parser.add_argument("output_path", help="Output .tflite path")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    convert(args.model_path, args.output_path, args.quantize)
