import numpy as np
import os
from skimage import io
from argparse import ArgumentParser

def create_npz(years):
    """
    Create npz file grouping all the tif LST and NDVI images in the given list of years
    input :
    years : A list containing the years for the desired images. Example : [2015,2016]
    """
     
    lst=None
    ndvi=None

    # Add images to the array by year
    for k in range(len(years)) :
        year=years[k]
        
        #LST path
        lst_root_dir      = 'MODIS/MOD_{}_{}'.format(year,'MOD11A1')
        lst_path = os.path.join(lst_root_dir, 'tifs_files/1km')
        lst_tifs = os.listdir(lst_path)
        lst_tifs.sort()

        #NDVI path
        ndvi_root_dir      = 'MODIS/MOD_{}_{}'.format(year,'MOD09GQ')
        ndvi_path = os.path.join(ndvi_root_dir, 'tifs_files/250m')
        ndvi_tifs = os.listdir(ndvi_path)
        ndvi_tifs.sort()

        # Make sure that each LST image has a corresponding NDVI image and sort the lists of the images
        lst_tif_indexes=[]
        ndvi_tif_indexes=[]
        for i in range(len(lst_tifs)):
            lst_tif = lst_tifs[i]
            lst_tif_name = lst_tif.split('.')
            for j in range(len(ndvi_tifs)):
                ndvi_tif=ndvi_tifs[j]
                ndvi_tif_name = ndvi_tif.split('.')
                if lst_tif_name[1:-1] == ndvi_tif_name[1:-1]:
                    lst_tif_indexes.append(i)
                    ndvi_tif_indexes.append(j)

        lst_tifs=[lst_tifs[i] for i in lst_tif_indexes]
        ndvi_tifs=[ndvi_tifs[i] for i in ndvi_tif_indexes]

        lst_tifs.sort()
        ndvi_tifs.sort()

        # Add LST images to the array
        i=0
        ndvi_indexes_to_delete = []
        for index in range(0, len(lst_tifs)):
                tif = lst_tifs[index]
                tif_path = os.path.join(lst_path,tif)
                current_array = io.imread(tif_path)
                # Make sure again that the LST images have no cloud or sea pixel coverage
                if(len(current_array[:,:,0][current_array[:,:,0] == 0]) > 0 or len(current_array[:,:,1][current_array[:,:,1] == 0]) > 0) :
                    ndvi_indexes_to_delete.append(index)
                    continue
                #First image
                if i == 0 and k==0:                
                    lst =  current_array
                #Add the image as a stack to the arrat
                elif i == 1 and k==0:
                    lst = np.stack([lst, current_array], axis=0)
                # Reshape and then concatenate the image if its a new year than the first one
                else :
                    current_array = np.reshape(current_array,(1,64,64,2))
                    lst = np.concatenate([lst, current_array], axis=0)
                i += 1 
        
        #Delete the ndvi image where the corresponding lst image have cloud or sea pixels
        for j in sorted(ndvi_indexes_to_delete,reverse=True):
            del ndvi_tifs[j]

        # Add the ndvi images to the array in the same manner as the lst images
        i=0
        for index in range(0, len(ndvi_tifs)):
                tif = ndvi_tifs[index]
                tif_path = os.path.join(ndvi_path,tif)
                if i == 0 and k==0:
                    ndvi =  io.imread(tif_path)
                elif i == 1 and k==0:
                    ndvi = np.stack([ndvi, io.imread(tif_path)], axis=0)
                else :
                    current_array = np.reshape(io.imread(tif_path),(1,256,256))
                    ndvi = np.concatenate([ndvi, current_array], axis=0)
                i += 1 

    # Save the two arrays to the npz file
    print(np.shape(lst))
    print(np.shape(ndvi))
    np.savez_compressed('data_stats',lst=lst, ndvi=ndvi)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--year_begin', type=int, default=2015)
    parser.add_argument('--year_end', type=int, default=2016)
    args = parser.parse_args()

    year_begin = args.year_begin 
    year_end = args.year_end 

    years = []
    years.append(year_begin)
    while year_begin != year_end :
         year_begin += 1 
         years.append(year_begin)
    
    create_npz(years)
         





    




            

