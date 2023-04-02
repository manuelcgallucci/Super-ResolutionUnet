import numpy as np
from Thunmpy import Methods
from PIL import Image
import matplotlib.pyplot as plt
from osgeo import gdal

def transform_images(path):
	filename_index = path
	dataset = gdal.Open(filename_index)
	cols = dataset.RasterXSize
	rows = dataset.RasterYSize
	projection_h = dataset.GetProjection()
	geotransform_h = dataset.GetGeoTransform()
	name = dataset.ReadAsArray(0, 0, cols, rows).astype(np.float64)
	return name

def plot_save(x, title, path, vmin=257, vmax=280):
    plt.imshow(x,vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.savefig(path)
    plt.close()

# Load the path of the entry data
path_aster="./data/AST_01262017212447_LST_VALIDATION_250.bsq"
path_lst="./data/MOD_20170126_LST_1km.img"
path_ndvi1="./data/MOD_20170126_NDVI_1km.img"
path_ndvi250="./data/MOD_20170126_NDVI_250.img"

# Transform the entry data to arrays
aster= transform_images(path_aster)
lst1= transform_images(path_lst)
ndvi1= transform_images(path_ndvi1)
ndvi250= transform_images(path_ndvi250)

# Save those arrays
np.save("./images/aster", aster)
np.save("./images/lst1", lst1)
np.save("./images/ndvi1", ndvi1)
np.save("./images/ndvi250", ndvi250)

# Run the classical methods on them
lst_TsHARP = Methods.TsHARP(path_lst, path_ndvi1, path_ndvi250, 4, min_T=257, path_image=False)
lst_ATPRK = Methods.ATPRK(path_lst, path_ndvi1, path_ndvi250, 4, 1000, block_size=5, sill=7, ran=1000, min_T=257, path_image=False)
lst_AATPRK = Methods.AATPRK(path_lst, path_ndvi1, path_ndvi250, 4, 1000, b_radius=2, block_size=5, sill=7, ran=1000, min_T=257, path_image=False)

# Save the results
np.save("./images/lst_TsHARP", lst_TsHARP)
np.save("./images/lst_ATPRK", lst_ATPRK)
np.save("./images/lst_AATPRK", lst_AATPRK)

# Choose an area to focus on, according to the aster image this indeces worked best and give images of 224x224
lst_TsHARP_cropped = lst_TsHARP[128:352,208:432]
lst_ATPRK_cropped = lst_ATPRK[128:352,208:432]
lst_AATPRK_cropped = lst_AATPRK[128:352,208:432]

# Display the results on that area as images
plot_save(lst_TsHARP_cropped, "TsHARP", "./images/lst_TsHARP_cropped.png")
plot_save(lst_ATPRK_cropped, "ATPRK", "./images/lst_ATPRK_cropped.png")
plot_save(lst_AATPRK_cropped, "AATPRK", "./images/lst_AATPRK_cropped.png")

