# convert_weights.py
import h5py
import numpy as np
import torch
from model import build_custom_unet


def get_keras_weights(f, layer_name):
    """Helper to safely extract kernel and bias from a Keras layer group."""
    weights_group = f["model_weights"] if "model_weights" in f else f
    if layer_name not in weights_group:
        print(f"⚠️ Layer {layer_name} not found in H5 file!")
        return None, None

    layer = weights_group[layer_name]
    # Keras usually nests weights inside another subgroup with the same name
    sub_group = layer[layer_name] if layer_name in layer else layer

    kernel, bias = None, None
    for key in sorted(sub_group.keys()):
        if "kernel" in key:
            kernel = np.array(sub_group[key])
        elif "bias" in key:
            bias = np.array(sub_group[key])

    return kernel, bias


def convert_keras_to_pytorch(h5_path, pth_path):
    pytorch_model = build_custom_unet()
    pt_state_dict = pytorch_model.state_dict()

    # Explicit mapping based on the chronological execution of your Keras U-Net
    mapping = {
        # --- ENCODER ---
        "conv1_1": "conv2d_1",
        "conv1_2": "conv2d_2",
        "pool1": "p1_stride",
        "conv2_1": "conv2d_3",
        "conv2_2": "conv2d_4",
        "pool2": "p2_stride",
        "conv3_1": "conv2d_5",
        "conv3_2": "conv2d_6",
        "pool3": "p3_stride",
        "conv4_1": "conv2d_7",
        "conv4_2": "conv2d_8",
        "pool4": "p4_stride",
        # --- BRIDGE ---
        "conv5_1": "conv2d_9",
        "conv5_2": "conv2d_10",
        # --- DECODER ---
        "up6": "u6",
        "conv6_1": "conv2d_11",
        "conv6_2": "conv2d_12",
        "up7": "u7",
        "conv7_1": "conv2d_13",
        "conv7_2": "conv2d_14",
        "up8": "u8",
        "conv8_1": "conv2d_15",
        "conv8_2": "conv2d_16",
        "up9": "u9",
        "conv9_1": "conv2d_17",
        "conv9_2": "conv2d_18",
        "output": "output_layer",
    }

    print("Opening Keras weights and mapping to PyTorch...")

    with h5py.File(h5_path, "r") as f:
        for pt_layer_name, keras_layer_name in mapping.items():
            k_kernel, k_bias = get_keras_weights(f, keras_layer_name)

            if k_kernel is None:
                continue

            # 1. Transpose Convolutional Weights
            # Keras: (H, W, In, Out) -> PyTorch: (Out, In, H, W)
            if len(k_kernel.shape) == 4:
                k_kernel = np.transpose(k_kernel, (3, 2, 0, 1))

            # 2. Assign to PyTorch state dictionary
            pt_state_dict[f"{pt_layer_name}.weight"] = torch.from_numpy(
                k_kernel
            )
            if k_bias is not None:
                pt_state_dict[f"{pt_layer_name}.bias"] = torch.from_numpy(
                    k_bias
                )

    # Save the strictly mapped state dictionary
    torch.save(pt_state_dict, pth_path)
    print(f"🎉 Successfully converted and saved to: {pth_path}")


if __name__ == "__main__":
    keras_h5_file = "./model_weights_strade.h5"
    pytorch_pth_file = "./model_weights.pth"

    convert_keras_to_pytorch(keras_h5_file, pytorch_pth_file)