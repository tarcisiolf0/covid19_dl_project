import numpy as np
import pydicom
from PIL import Image
import os
#%%
def get_dicom_paths(path):
    dicom_paths = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".dcm"):
                dicom_paths.append(os.path.join(root, filename))
    return dicom_paths

def diacom_to_png(file_path):
    dimg = pydicom.dcmread(file_path)
    
    im = dimg.pixel_array.astype(float)
    rescaled_image = (np.maximum(im,0)/im.max())*255
    
    final_image = np.uint8(rescaled_image)
    
    final_image = Image.fromarray(final_image)
    
    return final_image
        
#%%
root = './dataset/kaggle/train'
diacom_paths = get_dicom_paths(root)

#%%
for file_path in diacom_paths:
    png_img = diacom_to_png(file_path)
    
    save_path = './dataset/kaggle/png/'
    file_name, ext = os.path.splitext(os.path.basename(file_path))
    
    png_img.save(save_path+file_name+'.png')