import numpy as np


# def get_dot_map(im, points):
#     dot_map = np.zeros(im.shape[:2], dtype=np.int32)
#     h, w = dot_map.shape
#     print(f'points={len(points)}')
#     if len(points) == 0:
#         return dot_map

#     points = np.array(points, dtype=np.int32)
#     for point in points:
#         x, y = point
#         if 0 < x <= w and 0 < y <= h:
#             dot_map[y-1, x-1] += 1
#     print(f'dot_map={dot_map}')
#     return dot_map


def get_dot_map(im, points):
    dot_map = np.zeros((im.shape[0], im.shape[1]), dtype=float)

    h, w = dot_map.shape

    if len(points) == 0:
        return dot_map

    points = np.round(points).astype(int)  # 将点坐标转换为整数，假设需要这样处理以匹配图像索引
    dot_map[0, 0] = 1
    for i_dot in range(points.shape[0]):
        x, y = points[i_dot]

        if (x > w - 1) or (y > h - 1) or (x < 0) or (y < 0):
            continue

        dot_map[y, x] = dot_map[y, x] + 1
    dot_map = (dot_map - dot_map.min()) / (dot_map.max() - dot_map.min() + 1e-5) * 255
    return dot_map
