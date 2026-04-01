import h5py

with h5py.File("./model_weights_strade.h5", "r") as f:
    weights_group = f["model_weights"] if "model_weights" in f else f

    print("--- ALL KERAS LAYER NAMES FOUND ---")
    for name in weights_group.keys():# type: ignore
        print(name)