import os
import sys
import h5py
import numpy as np
import keras
from keras import layers, models

def convert():
    h5_path = os.path.join('model', 'crop_disease_model.h5')
    keras_path = os.path.join('model', 'crop_disease_model.keras')
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Source model file not found at {h5_path}")
        
    initial_h5_mtime = os.path.getmtime(h5_path)

    print(f"Reading H5 file directly from {h5_path} using h5py (NEVER calling load_model on H5)...")
    with h5py.File(h5_path, 'r') as f:
        if 'model_weights' not in f:
            raise KeyError("H5 file does not contain 'model_weights' group.")
        
        mw_group = f['model_weights']
        
        # 1. Locate MobileNetV2 weights group in H5
        mobilenet_key = None
        for key in mw_group.keys():
            if 'mobilenetv2' in key.lower():
                mobilenet_key = key
                break
                
        if mobilenet_key is None:
            raise KeyError("Could not locate MobileNetV2 weights group in H5 file.")
            
        mb_group = mw_group[mobilenet_key]
        if mobilenet_key in mb_group and isinstance(mb_group[mobilenet_key], h5py.Group):
            mb_group = mb_group[mobilenet_key]

        # 2. Deterministic Dense layer weight retrieval by exact verified paths
        dense_kernel_path = 'dense/dense/kernel'
        dense_bias_path = 'dense/dense/bias'

        if dense_kernel_path not in mw_group:
            raise KeyError(f"Exact H5 path '{dense_kernel_path}' not found in model_weights.")
        if dense_bias_path not in mw_group:
            raise KeyError(f"Exact H5 path '{dense_bias_path}' not found in model_weights.")

        dense_kernel_obj = mw_group[dense_kernel_path]
        dense_bias_obj = mw_group[dense_bias_path]

        if not isinstance(dense_kernel_obj, h5py.Dataset):
            raise KeyError(f"H5 path '{dense_kernel_path}' is not an h5py.Dataset.")
        if not isinstance(dense_bias_obj, h5py.Dataset):
            raise KeyError(f"H5 path '{dense_bias_path}' is not an h5py.Dataset.")

        dense_kernel = np.array(dense_kernel_obj)
        dense_bias = np.array(dense_bias_obj)

        print(f"Dense kernel exact H5 path: 'model_weights/{dense_kernel_path}', shape: {dense_kernel.shape}")
        print(f"Dense bias exact H5 path: 'model_weights/{dense_bias_path}', shape: {dense_bias.shape}")

        if dense_kernel.shape != (1280, 5):
            raise ValueError(f"Dense kernel shape mismatch: expected (1280, 5), got {dense_kernel.shape}.")
        if dense_bias.shape != (5,):
            raise ValueError(f"Dense bias shape mismatch: expected (5,), got {dense_bias.shape}.")

        # 3. Build fresh target architecture with weights=None
        print("\nBuilding fresh target architecture with MobileNetV2 (weights=None)...")
        inputs = layers.Input(shape=(224, 224, 3), name="input_layer")
        x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescaling_preprocessing")(inputs)
        
        base_model = keras.applications.MobileNetV2(
            input_tensor=x,
            include_top=False,
            weights=None
        )
        
        x = base_model.output
        x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
        x = layers.Dropout(0.2, name="dropout")(x)
        outputs = layers.Dense(5, activation="softmax", name="dense")(x)
        
        target_model = models.Model(inputs=inputs, outputs=outputs, name="crop_disease_model")

        # 4. Deterministic mapping of all target MobileNetV2 weights by exact H5 path
        print("\n--- Performing Deterministic MobileNetV2 Weight Mapping ---")
        mapped_h5_weights = []
        used_h5_paths = set()
        mapped_target_names = set()

        ALLOWED_ROLES = {'kernel', 'gamma', 'beta', 'moving_mean', 'moving_variance'}

        for layer in base_model.layers:
            if not layer.weights:
                continue
            
            for w in layer.weights:
                w_name = w.path
                w_shape = tuple(w.shape)
                
                role = w.name.split("/")[-1].split(":")[0]

                if role not in ALLOWED_ROLES:
                    raise ValueError(
                        f"CRITICAL ERROR: Unexpected weight role '{role}' for "
                        f"target variable '{w.name}' on layer '{layer.name}'."
                    )

                exact_path = w.path
                expected_path = f"{layer.name}/{role}"

                if exact_path != expected_path:
                    raise ValueError(
                        f"CRITICAL ERROR: Target weight path mismatch. "
                        f"Expected '{expected_path}', got '{exact_path}'."
                    )

                if exact_path not in mb_group:
                    raise KeyError(
                        f"CRITICAL ERROR: Exact H5 path '{exact_path}' for target weight '{w_name}' "
                        f"(layer '{layer.name}', role '{role}') not found in H5 MobileNetV2 group."
                    )

                ds_obj = mb_group[exact_path]
                if not isinstance(ds_obj, h5py.Dataset):
                    raise KeyError(
                        f"CRITICAL ERROR: H5 path '{exact_path}' is not an h5py.Dataset."
                    )

                arr = np.array(ds_obj)

                # Verify exact shape match
                if arr.shape != w_shape:
                    raise ValueError(
                        f"CRITICAL ERROR: Shape mismatch for target weight '{w_name}' (H5 path '{exact_path}'): "
                        f"Target shape {w_shape} != H5 shape {arr.shape}."
                    )

                if exact_path in used_h5_paths:
                    raise ValueError(f"CRITICAL ERROR: Duplicate H5 path mapping detected for '{exact_path}'.")

                if w_name in mapped_target_names:
                    raise ValueError(f"CRITICAL ERROR: Duplicate target variable mapping detected for '{w_name}'.")

                used_h5_paths.add(exact_path)
                mapped_target_names.add(w_name)
                mapped_h5_weights.append(arr)

        # 5. Strict Verification of Mapped Weight Counts
        total_target_vars = len(base_model.weights)
        print(f"Mapped {len(mapped_h5_weights)} / {total_target_vars} target weight variables.")

        if total_target_vars != 260:
            raise ValueError(f"Expected fresh MobileNetV2 to have 260 weights, found {total_target_vars}.")

        if len(mapped_h5_weights) != 260:
            raise ValueError(f"Expected exactly 260 mapped weights, got {len(mapped_h5_weights)}.")

        if len(used_h5_paths) != 260:
            raise ValueError(f"Expected exactly 260 unique H5 dataset paths, got {len(used_h5_paths)}.")

        if len(mapped_target_names) != 260:
            raise ValueError(f"Expected exactly 260 unique target names, got {len(mapped_target_names)}.")

        print("Strict deterministic mapping verification passed: Exactly 260 unique target variables mapped to 260 unique H5 dataset paths with exact shape matches!")

        # 6. Set weights to target model
        print("Setting mapped weights into target base_model and Dense layer...")
        base_model.set_weights(mapped_h5_weights)
        target_model.get_layer('dense').set_weights([dense_kernel, dense_bias])
        print("Successfully set all weights into target model.")

    # 7. Pre-Save Inspection & Safety Verification
    print("\n--- Pre-Save Weight Statistics & Safety Checks ---")
    print(f"Dense kernel min: {np.min(dense_kernel):.6f}, max: {np.max(dense_kernel):.6f}")
    print(f"Dense bias min: {np.min(dense_bias):.6f}, max: {np.max(dense_bias):.6f}")
    print(f"Total target model weights: {len(target_model.weights)}")

    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    dummy_pred = target_model.predict(dummy_input, verbose=0)
    print(f"Dummy prediction output shape: {dummy_pred.shape}")
    
    prob_sum = float(np.sum(dummy_pred))
    print(f"Sum of dummy prediction probabilities: {prob_sum:.6f}")

    if not np.isclose(prob_sum, 1.0, atol=1e-4):
        raise ValueError(f"CRITICAL ERROR: Dummy prediction probability sum is invalid: {prob_sum} (expected ~1.0).")

    print("Pre-save safety checks passed!")

    # 8. Save to modern .keras format
    print(f"\nSaving converted model to {keras_path}...")
    target_model.save(keras_path)
    print(f"Successfully saved {keras_path}.")

    # 9. Reload ONLY the new .keras file and verify architecture/shapes
    print("\n--- Reloading and Verifying .keras Model ---")
    verified_model = models.load_model(keras_path, compile=False)
    print(f"Loaded converted model from {keras_path}")

    input_shape = verified_model.input_shape
    output_shape = verified_model.output_shape
    print(f"Model Input Shape: {input_shape}")
    print(f"Model Output Shape: {output_shape}")

    if input_shape != (None, 224, 224, 3):
        raise ValueError(f"Input shape verification failed: expected (None, 224, 224, 3), got {input_shape}")

    if output_shape != (None, 5):
        raise ValueError(f"Output shape verification failed: expected (None, 5), got {output_shape}")

    reloaded_dummy_output = verified_model.predict(dummy_input, verbose=0)
    print(f"Reloaded Dummy Prediction Shape: {reloaded_dummy_output.shape}")

    if reloaded_dummy_output.shape != (1, 5):
        raise ValueError(f"Reloaded dummy output shape verification failed: expected (1, 5), got {reloaded_dummy_output.shape}")

    # 10. Verify original H5 modification timestamp is unchanged
    final_h5_mtime = os.path.getmtime(h5_path)
    if initial_h5_mtime != final_h5_mtime:
        raise RuntimeError("CRITICAL ERROR: Original H5 file modification timestamp changed!")

    print("\n>>> DETERMINISTIC CONVERSION & VERIFICATION COMPLETE! All checks passed. Original H5 untouched. <<<")

if __name__ == '__main__':
    convert()
