import os
import cv2
import numpy as np
from scipy.io import loadmat

from get_density_map_gaussian import get_density_map_gaussian
from get_dot_map import get_dot_map

init_max_size = (2048, 2048)
min_size = (576, 768)

data_path = 'E:/NWPUCrowd/data'  # 输入路径 解压后
''' 
nwpu
  ├─jsons
  ├─mats
  ├─images
  ├─train.txt
  ├─val.txt
  └─test.txt
'''

output_path = 'E:/NWPUCrowd/dataOut'  # 创建txt_list文件夹，把train.txt、val.txt、test.txt放进去,最终用于训练的目录结构如下：

'''
dataOut
   ├── den
   ├── dot
   ├── img
   ├── txt_list
   │   ├── test.txt
   │   ├── train.txt
   │   └── val.txtvis
   └── vis
'''
os.makedirs(output_path, exist_ok=True)
path_img = os.path.join(output_path, 'img/')
os.makedirs(path_img, exist_ok=True)
path_den = os.path.join(output_path, 'den/')
os.makedirs(path_den, exist_ok=True)
path_dot = os.path.join(output_path, 'dot/')
os.makedirs(path_dot, exist_ok=True)
path_vis = os.path.join(output_path, 'vis/')
os.makedirs(path_vis, exist_ok=True)

img_list = [
    f for f in os.listdir(os.path.join(data_path, 'images/'))
    if f.endswith('.jpg')
]

save_type = [1, 1, 1, 1]  # img, density, dot, vis
img_list = sorted(img_list)
for i_img, img_name in enumerate(img_list):
    print(img_name)
    name_split = img_name.split('.jpg')
    mat_name = f'{name_split[0]}.mat'

    img_path = os.path.join(data_path, 'images/', img_name)
    im = cv2.imread(img_path)
    h, w, c = im.shape

    rate = init_max_size[0] / h
    rate_w = w * rate
    if rate_w > init_max_size[1]:
        rate = init_max_size[1] / w

    tmp_h = int(np.ceil(h * rate / 16) * 16)
    if tmp_h < min_size[0]:
        rate = min_size[0] / h

    tmp_w = int(np.ceil(w * rate / 16) * 16)
    if tmp_w < min_size[1]:
        rate = min_size[1] / w

    tmp_h = int(np.ceil(h * rate / 16) * 16)
    tmp_w = int(np.ceil(w * rate / 16) * 16)

    rate_h = float(tmp_h / h)
    rate_w = float(tmp_w / w)
    im = cv2.resize(im, (tmp_w, tmp_h))

    if save_type[0] == 1:
        cv2.imwrite(os.path.join(path_img, img_name), im)

    mat_path = os.path.join(data_path, 'mats/', mat_name)
    if not os.path.isfile(mat_path):
        continue

    # Load .mat file here using scipy.io.loadmat or similar method
    # Assuming you've loaded annPoints into a variable
    mat_file = loadmat(mat_path)
    annPoints = mat_file['annPoints']

    if annPoints.size != 0:
        # 缩放annPoints坐标
        # annPoints[:, 0] *= rate_w
        # annPoints[:, 1] *= rate_h
        annPoints = annPoints.astype(np.float64)
        annPoints[:, 0] *= rate_w
        annPoints[:, 1] *= rate_h

        # 如果需要转回 int 类型，并且知道不会因为缩放而导致精度丢失，则可以这样做：
        annPoints = annPoints.astype(np.int32)
        # 创建检查列表并进行边界检查
        check_list = np.ones(annPoints.shape[0], dtype=bool)
        for j in range(annPoints.shape[0]):
            x = np.ceil(annPoints[j, 0])
            y = np.ceil(annPoints[j, 1])
            if (x > tmp_w) or (y > tmp_h) or (x <= 0) or (y <= 0):
                check_list[j] = False

        # 过滤出在有效范围内的annPoints
        annPoints = annPoints[check_list, :]

    if save_type[1] == 1:
        im_density = get_density_map_gaussian(im, annPoints, 15, 4)
        np.savetxt(os.path.join(path_den, f'{name_split[0]}.csv'),
                   im_density,
                   delimiter=',')

    if save_type[2] == 1:
        im_dots = get_dot_map(im, annPoints)
        im_dots = im_dots.astype(np.uint8)

        cv2.imwrite(os.path.join(path_dot, f'{name_split[0]}.png'), im_dots)

    if save_type[3] == 1:
        if annPoints.size != 0:
            for point in annPoints:
                cv2.circle(im, (int(point[0]), int(point[1])), 5, (0, 0, 255),
                           -1)
        cv2.imwrite(os.path.join(path_vis, f'{name_split[0]}.jpg'), im)
