import numpy as np
import cv2


# def get_density_map_gaussian(im, points, k_size, sigma):
#     # im_density = zeros(size(im,1),size(im,2)); 转换为python代码

#     # 假设im是灰度图像，创建与im同尺寸的浮点型全零数组作为密度图
#     im_density = np.zeros_like(im[:, :, 0], dtype=np.float32)
#     h, w = im_density.shape

#     if len(points) == 0:
#         return

#     for j in range(len(points)):
#         f_sz = k_size
#         H = cv2.getGaussianKernel(f_sz, sigma)
#         H = H @ H.T  # 创建高斯核

#         x = min(w, max(1, abs(int(np.floor(points[j, 0])))))
#         y = min(h, max(1, abs(int(np.floor(points[j, 1])))))

#         if x > w or y > h:
#             continue
#         x1 = x - int(np.floor(f_sz / 2))
#         y1 = y - int(np.floor(f_sz / 2))
#         x2 = x + int(np.floor(f_sz / 2))
#         y2 = y + int(np.floor(f_sz / 2))

#         dfx1, dfy1, dfx2, dfy2 = 0, 0, 0, 0
#         change_H = False

#         if x1 < 1:
#             dfx1 = abs(x1) + 1
#             x1 = 1
#             change_H = True
#         if y1 < 1:
#             dfy1 = abs(y1) + 1
#             y1 = 1
#             change_H = True
#         if x2 > w:
#             dfx2 = x2 - w
#             x2 = w
#             change_H = True
#         if y2 > h:
#             dfy2 = y2 - h
#             y2 = h
#             change_H = True
#         x1h = 1 + dfx1
#         y1h = 1 + dfy1
#         x2h = f_sz - dfx2
#         y2h = f_sz - dfy2

#         if change_H:
#             H = cv2.getGaussianKernel(y2h - y1h + 1, sigma) \
#                 @ cv2.getGaussianKernel(x2h - x1h + 1, sigma).T
#         try:
#             im_density[y1:y2+1, x1:x2+1] += H
#         except:
#             print(f'x2h - x1h + 1={x2h-x1h+1}')
#             print(f'change_H={change_H}')
#             print(f'h={h},w={w},x={x},y={y},j={j}')
#             print(f'x1={x1},y1={y1},x2={x2},y2={y2}')
#             print(f'{x2-x1+1},{y2-y1+1},H.shape={
#                   H.shape},im_density.shape={im_density.shape}')
#             raise

#     # im_density /= np.max(im_density)  # 可选：归一化密度图到[0, 1]区间
#     return im_density
import numpy as np
from scipy.ndimage import gaussian_filter


def get_density_map_gaussian(im, points, k_size, sigma):
    im_density = np.zeros((im.shape[0], im.shape[1]))

    if len(points) == 0:
        return im_density

    h, w = im_density.shape

    for j in range(len(points)):
        kernel_sz = k_size
        H = gaussian_filter(np.ones((kernel_sz, kernel_sz)), sigma)

        x = max(1, min(w - 1, int(np.floor(points[j, 0]))))
        y = max(1, min(h - 1, int(np.floor(points[j, 1]))))

        if x > w - 1 or y > h - 1:
            continue

        x1 = x - int(np.floor(kernel_sz / 2))
        y1 = y - int(np.floor(kernel_sz / 2))
        x2 = x + int(np.floor(kernel_sz / 2))
        y2 = y + int(np.floor(kernel_sz / 2))

        dfx1 = 0
        dfy1 = 0
        dfx2 = 0
        dfy2 = 0
        change_H = False

        if x1 < 0:
            dfx1 = abs(x1)
            x1 = 0
            change_H = True
        if y1 < 0:
            dfy1 = abs(y1)
            y1 = 0
            change_H = True
        if x2 > w - 1:
            dfx2 = x2 - (w - 1)
            x2 = w - 1
            change_H = True
        if y2 > h - 1:
            dfy2 = y2 - (h - 1)
            y2 = h - 1
            change_H = True

        if change_H:
            H = gaussian_filter(np.ones((y2 - y1 + 1, x2 - x1 + 1)), sigma)

        im_density[y1:y2+1, x1:x2+1] += H

    return im_density
