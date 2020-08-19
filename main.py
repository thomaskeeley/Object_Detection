#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 14:12:13 2020

@author: thomaskeeley
"""

#%%

import warnings
warnings.filterwarnings('ignore')

#%%

from object_detection import ObjectDetection

#%%

if __name__ == '__main__':
    
    instance = ObjectDetection(
    working_dir = '/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/',
    import_image_path = '/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/nk_aoi.tif',
    download_image = False,
    aoi_path = None,
    username = 'sirthomasnewton',
    password = 'Coronavirus1!',
    date_range = ('20190101', '20200101'),
    mask_path = '/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/nk_aoi_b.geojson',
    # mask_path = None,
    out_dir = '/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/',
    step = 4,
    prediction_threshold = 0.99,
    pos_chip_dir = ('/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/positive_chips/'),
    neg_chip_dir = ('/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/negative_chips/'),
    chip_width = 20,
    chip_height = 20,
    augment_pos_chips = True,
    augment_neg_chips = False,
    save_model = False,
    import_model = None,
    segmentation = True)
    
    instance.execute()
    

#%%


im_tensor, left, top, im_x_res, im_y_res, crs = instance.create_tensor()
im_tensor = im_tensor.transpose(2,0,1)
coordinates = instance.detect(im_tensor)
trimmed_coordinates = instance.trim_results(coordinates)

wkt_strings = instance.create_geo_output(trimmed_coordinates, left, top, im_x_res, im_y_res, crs)

df = pd.DataFrame(wkt_strings, columns=['wkt'])
df.to_csv(out_dir + 'obj_det_wkt-{}_{}.csv'.format(crs,date.today().strftime('%Y%m%d')))

test_list = []
final_coords = []   
for (i, item) in enumerate(coordinates):
    coords = []
    score = []
    poly_i = Polygon(item[0])
    for (j, comp) in enumerate(coordinates):
        if abs(item[0][0][0] - comp[0][0][0]) < 20:
            poly_j = Polygon(comp[0])
            intersection = poly_i.intersection(poly_j)
            iou = intersection.area / 400
            
            if iou > 0.1:
                coords.append(item)
                score.append(item[1][0][1])
    print(coords)
    idx = score.index(max(score))
    print(idx)
    final_coords.append(coords[idx])
        
                
                    if item[1][0][1] > comp[1][0][1]:
                        deleted_items.append(comp)
                    else:
                        deleted_items.append(item)
                    trimmed_coordinates = [e for e in coordinates if e not in deleted_items]
        print('\n     Complete')


test_areas = instance.k_means_segmentation(im_tensor)
instance.show_results(test_areas, im_tensor)
instance.create_geo_output(test_areas, left, top, im_x_res, im_y_res, crs)

wkt_strings = []
for i in test_areas:
    x_min = (i[2] * im_x_res) + left
    y_min = top - (i[0] * im_y_res)
    x_max = (i[3] * im_x_res) +left
    y_max = top - (i[1] * im_y_res)
    
    wkt = 'POLYGON (({} {}, {} {}, {} {}, {} {}, {} {}))'.format(
        x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max, x_min, y_min)
    wkt_strings.append(wkt)
            
df = pd.DataFrame(wkt_strings, columns=['wkt'])
df.to_csv(out_dir + 'obj_det_wkt-{}_{}.csv'.format(crs,date.today().strftime('%Y%m%d')))

#%%

def moving_window(x, y, im_tensor):
        """
        
        Parameters
        ----------
        x : Integer
            DESCRIPTION: Current x position in image
        y : Integer
            DESCRIPTION: Current y position in image
        im_tensor : Array of uint8
            DESCRIPTION: 3-D array version of input image

        Returns
        -------
        window : Array of uint8
            DESCRIPTION: Image patch from larger image to run prediction on

        """
        window = np.arange(3 * width * height).reshape(3, width, height)
        for i in range(width):
            for j in range(height):
                window[0][i][j] = im_tensor[0][y+i][x+j]
                window[1][i][j] = im_tensor[1][y+i][x+j]
                window[2][i][j] = im_tensor[2][y+i][x+j]
        window = window.reshape([-1, 3, width, height])
        window = window.transpose([0,2,3,1])
        window = window / 255
        return window  

for idx, i in enumerate(test_areas):
    sys.stdout.write('\r{}%  '.format(round(idx/len(test_areas), 3)))
    x = int(i[2])
    y = int(i[0])
    window = moving_window(x, y, im_tensor)
    if np.count_nonzero(window) == 0:
        x += 20
    else:
        result = model.predict(window)
        if result[0][1] > prediction_threshold:
            if np.count_nonzero(window) > mask_threshold:
                x_min = x*step
                y_min = y*step
                
                x_max = x_min + 20
                y_max = y_min + 20
                
                coords = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
                
                coordinates.append([coords, result])
            
#%%

if __name__ == '__main__':
    
    instance = ObjectDetection(
    working_dir = '/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/',
    import_image_path = None,
    download_image = True,
    aoi_path = '/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/test_aoi_small.geojson',
    username = 'sirthomasnewton',
    password = 'Coronavirus1!',
    date_range = ('20190101', '20200101'),
    mask_path = None,
    out_dir = '/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/',
    step = 4,
    prediction_threshold = 0.90,
    pos_chip_dir = ('/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/positive_chips/'),
    neg_chip_dir = ('/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/negative_chips/'),
    chip_width = 20,
    chip_height = 20,
    augment_pos_chips = True,
    augment_neg_chips = False,
    save_model = True,
    import_model = None,
    segmentation = True)
    
    
    instance.execute()

#%%
im_tensor= instance.create_tensor()[0]

#%%

img=cv2.cvtColor(im_tensor,cv2.COLOR_BGR2RGB)

vectorized = img.reshape((-1,3))

vectorized = np.float32(vectorized)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 3
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

res = center[label.flatten()]
result_image = res.reshape((img.shape))


#%%
plt.figure(1, figsize = (15, 30))

plt.subplot(3, 1, 1)
plt.legend()
plt.imshow(result_image)


plt.show()

#%%

unique, counts = np.unique(result_image, return_counts=True)

#%%

y_shape = result_image.shape[0]
x_shape = result_image.shape[1]
test_areas = []
target = unique[-1]
segment_pix = np.where(result_image == target)
for i in range(0, len(segment_pix[0])):
    
    row = segment_pix[0][i]
    col = segment_pix[1][i]
    
    if row <= 40:
        y_min = 0
        y_max = 80
    if row >= 40:
        y_min = row - 40
        if row + 40 >= y_shape:
            y_max = y_shape
            y_min = y_max - 80
        else: 
            y_max = row + 40
        
    if col <= 40:
        x_min = 0
        x_max = 80
    if col >= 40:    
        x_min = col - 40
        if col + 40 >= x_shape:
            x_max = x_shape
            x_min = x_max - 80
        else:
            x_max = col + 40
    
    bounds = [y_min, y_max, x_min, x_max]
    test_areas.append(bounds)
    
test_areas_set = set(tuple(x) for x in test_areas)
test_areas = [list(x) for x in test_areas_set]







#%%
/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/test_aoi_small.geojson

image.count
        
image.width
image.height

left, bottom, right, top = image.bounds

im_x_res = (right-left)/10980
im_y_res = (top-bottom)/im_height  

crs = str(im.crs).split(':')[1]

image_path = '/Users/thomaskeeley/Documents/School/Capstone-master/Thomas/S2B_MSIL2A_20190629T025549_N0212_R032_T48MYU_20190629T083111.SAFE/GRANULE/L2A_T48MYU_A012070_20190629T031832/IMG_DATA/R10m/T48MYU_20190629T025549_TCI_10m.jp2'
image = rasterio.open(image_path)
crs = str(image.crs).split(':')[1] 
shape = gpd.read_file(aoi_path)
shape = shape.to_crs({'init': 'epsg:{}'.format(crs)})
with fiona.open(aoi_path, "r") as shape:
        geoms = [feature["geometry"] for feature in shape]
    
        clipped_image, out_transform = mask(image, geoms, crop=True)
    
im_tensor = clipped_image.transpose(1, 2, 0)
        
    if self.mask_path:
        with fiona.open(self.mask_path, "r") as shapefile:
            geoms = [feature["geometry"] for feature in shapefile]
        
            out_image, out_transform = rasterio.mask.mask(image, geoms, invert=True)
            
            im_tensor = out_image.transpose(1, 2, 0)
    print('\n     Complete')
    return im_tensor

