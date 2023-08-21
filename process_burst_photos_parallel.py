import numpy as np
import imageio
import rawpy
import os

from joblib import Parallel, delayed

# --------------------------------------------------------
# Specify the camera sensor model and date below
# --------------------------------------------------------

sensor_model = "pixel2_front"
date = "21_05_23"

# --------------------------------------------------------
# Setting up the input and output folders

input_folder = "original_photos/" + sensor_model + "/" #+ date + "/"
output_folder_raw = "processed_photos/" + sensor_model + "/raw/"
output_folder_png = "processed_photos/" + sensor_model + "/png/"
burst_size = 20

if not os.path.exists(output_folder_raw+"denoised/"):
    os.makedirs(output_folder_raw+"denoised/")

if not os.path.exists(output_folder_raw+"original/"):
    os.makedirs(output_folder_raw+"original/")

if not os.path.exists(output_folder_png+"denoised/"):
    os.makedirs(output_folder_png+"denoised/")

if not os.path.exists(output_folder_png+"original/"):
    os.makedirs(output_folder_png+"original/")

# --------------------------------------------------------
# Finding all input images and processing them to get denoised photos

file_names = os.listdir(input_folder)
img_ids = [file_name.split("_")[0] for file_name in file_names if file_name.endswith(".dng") and not file_name.startswith(".")]
img_ids = np.asarray(np.unique(img_ids), dtype=int)

print("---------------------------------")
print("%d images found" % img_ids.shape[0])
print("---------------------------------")

def generate_images(img_id):
    print("Processing image %d" % img_id)
    raw_avg = []
    processed_images = []

    raw_middle = rawpy.imread(input_folder + str(img_id) + "_10.dng")
    print("Processing image second %d" % img_id)

    rgb_middle = raw_middle
    imageio.imsave(output_folder_png + "original/" + str(img_id) + ".png",
                   rgb_middle.postprocess(output_color=rawpy.ColorSpace.Adobe, use_camera_wb=True, no_auto_bright=False))

    raw_middle_np = np.array(raw_middle.raw_image_visible)
    normalized_mid_image = (raw_middle_np / raw_middle_np.max() * 65535).astype(np.uint16)
    imageio.imwrite(output_folder_raw + "original/" + str(img_id) + ".png", normalized_mid_image)

    for i in range(0, burst_size):
        try:
            raw = rawpy.imread(input_folder + str(img_id) + "_" + str(i) + ".dng")
        except:
            print("Failure at: ", img_id, " index: ", i)
            continue
        raw_processed = raw
        processed_image = raw_processed.postprocess(output_color=rawpy.ColorSpace.Adobe, use_camera_wb=True, no_auto_bright=False)
        processed_images.append(processed_image)

        raw_avg.append(raw.raw_image_visible)

    mean_image = np.mean(processed_images, axis=0).astype(np.uint8)
    imageio.imwrite(output_folder_png + "denoised/" + str(img_id) + ".png", mean_image)

    raw_avg = [np.array(image) for image in raw_avg]
    mean_raw = np.mean(raw_avg, axis=0)
    normalized_mean_image = (mean_raw / mean_raw.max() * 65535).astype(np.uint16)
    imageio.imwrite(output_folder_raw + "denoised/" + str(img_id) + ".png", normalized_mean_image)

    print("Done image %d" % img_id)

img_ids = np.asarray(sorted(img_ids))
print(img_ids)
end_index=np.where(img_ids == 210)[0][0]
print("start_index",end_index)
Parallel(n_jobs=8)(delayed(generate_images)(i) for i in sorted(img_ids[:end_index]))

# pixel 2 problems at: 572, 573
