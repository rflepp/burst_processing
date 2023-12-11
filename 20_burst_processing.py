import numpy as np
import imageio
import rawpy
import os

from joblib import Parallel, delayed

# --------------------------------------------------------
# Specify the camera sensor model and date below

sensor_model = "Pixel_Frontcamera"

# --------------------------------------------------------
# Setting up the input and output folders

input_folder = "/Volumes/ELEMENTS/image_denoising_dataset/original_photos/" + sensor_model + "/22_05_2023/"
output_folder_raw = "/Volumes/ETH4_backup/processed_photos/" + sensor_model + "/raw/"
output_folder_png = "/Volumes/ETH4_backup/processed_photos/" + sensor_model + "/png/"

denoised_folder_png = output_folder_png+"/denoised/"
burst_size = 20

if not os.path.exists(output_folder_png+"original_20/"):
    os.makedirs(output_folder_png+"original_20/")

# --------------------------------------------------------
# Finding all input images and processing them to get denoised photos

file_names = os.listdir(input_folder)
img_ids = [file_name.split("_")[0] for file_name in file_names if file_name.endswith(".dng") and not file_name.startswith(".")]
img_ids = np.asarray(np.unique(img_ids), dtype=int)

print("---------------------------------")
print("%d images found" % img_ids.shape[0])
print("---------------------------------")

def generate_images(img_id):
    if img_id not in []:
        print("Processing image %d" % img_id)

        for i in range(0,20):
            try:
                raw_middle = rawpy.imread(input_folder + str(img_id) + "_"+str(i)+".dng")
                print("Processing image second",img_id, ", ", i)

                rgb_image = raw_middle.postprocess(output_color=rawpy.ColorSpace.Adobe, use_camera_wb=True, no_auto_bright=False)

                imageio.imsave(output_folder_png + "original_20/" + str(img_id)+"_file_"+str(i)+ ".png", rgb_image)
            except:
                print("error at: ", img_id)

        print("Done image %d" % img_id)

img_ids = np.asarray(sorted(img_ids))
print(img_ids)
#end_index=np.where(img_ids == 861)[0][0]
#print("start_index",end_index)
Parallel(n_jobs=6)(delayed(generate_images)(i) for i in sorted(img_ids))
