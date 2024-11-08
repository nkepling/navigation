import json
import matplotlib.pyplot as plt
import numpy as np


data_dir = "/Users/nathankeplinger/Documents/Vanderbilt/Research/ANSR/pwm/"


with open(data_dir + 'target_target_map.npy',"rb") as f:
    example_maps = np.load(f)


print(example_maps.shape)

plt.imshow(example_maps, cmap='viridis')
plt.colorbar()  # Optional: adds a color scale bar
plt.title("Visualization of .npy data")
plt.show()