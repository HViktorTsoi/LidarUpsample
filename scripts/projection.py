import multiprocessing
import os
import time

import numpy as np
import array
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2
from PIL import Image

MAX_DEPTH = 40


def interp(pc, img_size):
    """
    pointcloud front-view interpolation
    :param pc:
    :param img_size:
    :return:
    """
    assert pc.shape[1] == 5, 'pointcloud must has 3 channel features:{depth, intensity, height}'

    grid_x, grid_y = np.mgrid[0:img_size[0]:1, 0:img_size[1]:1]
    img_ch_depth = griddata(pc[:, 0:2], np.maximum((MAX_DEPTH - pc[:, 2]) / MAX_DEPTH, 0), (grid_x, grid_y),
                            method='linear').T
    img_ch_intensity = griddata(pc[:, 0:2], pc[:, 3], (grid_x, grid_y), method='linear').T
    img_ch_idle = np.zeros_like(img_ch_depth)
    # 原始高度数据
    img_ch_height = np.zeros_like(img_ch_depth)
    img_ch_height[np.int_(pc[:, 1]), np.int_(pc[:, 0])] = (pc[:, 4] + 1.5) / 4.

    # img = np.stack([img_ch_height, img_ch_intensity, img_ch_depth, ], axis=-1)
    img = np.stack([img_ch_idle, img_ch_idle, img_ch_depth, ], axis=-1)
    img = np.int_(img * 255)
    return img


def load_pc(bin_file_path):
    """
    load pointcloud file (velodyne format)
    :param bin_file_path:
    :return:
    """
    with open(bin_file_path, 'rb') as bin_file:
        pc = array.array('f')
        pc.frombytes(bin_file.read())
        pc = np.array(pc).reshape(-1, 4)
        return pc


def load_calib(calib_file_path):
    """
    load calibration file(KITTI object format)
    :param calib_file_path:
    :return:
    """
    calib_file = open(calib_file_path, 'r').readlines()
    calib_file = [line
                      .replace('Tr_velo_to_cam', 'Tr_velo_cam')
                      .replace('R0_rect', 'R_rect')
                      .replace('\n', '')
                      .replace(':', '')
                      .split(' ')
                  for line in calib_file]
    calib_file = {line[0]: [float(item) for item in line[1:] if item != ''] for line in calib_file if len(line) > 1}
    return calib_file


def parse_calib_file(calib_file):
    """
    parse calibration file to calibration matrix
    :param calib_file:
    :return:
    """

    # 外参矩阵
    Tr_velo_cam = np.array(calib_file['Tr_velo_cam']).reshape(3, 4)
    Tr_velo_cam = np.concatenate([Tr_velo_cam, [[0, 0, 0, 1]]], axis=0)
    # 矫正矩阵
    R_rect = np.array(calib_file['R_rect']).reshape(3, 3)
    R_rect = np.pad(R_rect, [[0, 1], [0, 1]], mode='constant')
    R_rect[-1, -1] = 1
    # 内参矩阵
    P2 = np.array(calib_file['P2']).reshape(3, 4)

    return np.matmul(np.matmul(P2, R_rect), Tr_velo_cam)


def zbuffer_projection(pc, img_size):
    """
    serial implement of zbuffer
    :param pc: input pointcloud
    :param img_size: image size
    :return: depth
    """
    z_buffer = np.zeros(img_size[::-1])
    depth = (np.maximum(0, MAX_DEPTH - pc[:, 2]) / MAX_DEPTH) ** 1.1
    plane_coord = np.int_(pc[:, :2])
    for idx, coord in enumerate(plane_coord):
        if depth[idx] > z_buffer[coord[1], coord[0]]:
            z_buffer[coord[1], coord[0]] = depth[idx]
    z_buffer *= 255
    return z_buffer


def project_lidar_to_image(pc, img_size, calib_file, yaw_deg, method='velodyne'):
    """
    获取点云的前视投影
    :param pc: 输入点云(N, 4)
    :param img_size: (w, h)
    :param calib_file: KITTI calib文件的path
    :param yaw_deg: 将点云按z轴旋转的角度，默认设置为0就可以
    :return:
    """
    yaw_deg = yaw_deg / 180 * np.pi
    calib_mat = parse_calib_file(load_calib(calib_file))

    # 投影
    intensity = np.copy(pc[:, 3]).reshape(-1, 1)
    height = np.copy(pc[:, 2]).reshape(-1, 1)
    pc[:, 3] = 1
    # yaw旋转
    rotate_mat = np.array([
        [np.cos(yaw_deg), -np.sin(yaw_deg), 0, 0],
        [np.sin(yaw_deg), np.cos(yaw_deg), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    pc = np.matmul(rotate_mat, pc.T).T

    # 限制前视 并且限制fov在90度之内
    # 计算fov
    fov_h = np.arctan(np.abs(pc[:, 1] / pc[:, 0]))
    fov_v = np.arctan(np.abs(pc[:, 2] / pc[:, 0]))
    indice = np.where(np.bitwise_and(
        pc[:, 0] > 0.5,
        np.bitwise_and(fov_h < np.pi / 4, fov_v < np.pi / 10, )
    ))
    pc = pc[indice]
    print(pc.shape)

    # mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=(1, 0, 0), mode='point')
    # pc = pc[np.random.permutation(len(pc))[:28000], :]
    # mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=(0, 1, 0), mode='point')
    # mlab.show()

    intensity = intensity[indice]
    height = height[indice]
    # 进行投影变换
    pc = np.matmul(calib_mat, pc.T).T

    # z深度归一化
    pc[:, :2] /= pc[:, 2:]

    # 还原intensity
    pc = np.concatenate([pc, intensity, height], axis=1)

    # 按照原图大小裁剪
    pc = pc[np.where(pc[:, 0] >= 0)]
    pc = pc[np.where(pc[:, 0] < img_size[0])]
    pc = pc[np.where(pc[:, 1] >= 0)]
    pc = pc[np.where(pc[:, 1] < img_size[1])]

    if method == 'colormap':
        # 方法1 对点云做color map之后再绘制到前视图上
        # img = color_map(pc, img_size)
        pass
    elif method == 'velodyne':
        # 方法2 下边是不对投影之后的点云做任何处理，直接以点的形式绘制到前视图上
        img = np.zeros([375, 1242, 3])
        # BGR
        img[np.int_(pc[:, 1]), np.int_(pc[:, 0]), 2] = (np.maximum(0, MAX_DEPTH - pc[:, 2]) / MAX_DEPTH) ** 1.1  # depth
        # img[np.int_(pc[:, 1]), np.int_(pc[:, 0]), 1] = (pc[:, 3] + 0.1)  # intensity
        img = np.int_(img * 255)
    elif method == 'interp':
        # 方法3 interp
        img = interp(pc, img_size)
    elif method == 'zbuffer':
        # 方法4 point-zbuffer投影
        img = np.zeros([img_size[1], img_size[0], 3])
        img[:, :, 2] = zbuffer_projection(pc, img_size)
    else:
        img = None

    return img


if __name__ == '__main__':
    # object
    yaw_deg = 0
    object_origin_dataroot = '/home/hviktortsoi/data/KITTI/{}/'
    pc_file = '0000/000100.bin'
    # 读取标定文件
    calib_file_path = os.path.join(object_origin_dataroot.format('calib'), pc_file[5:-6] + '.txt')
    # calib_file = load_calib(calib_file_path)
    # 读取lidar
    bin_file_path = os.path.join(object_origin_dataroot.format('interp'), pc_file)
    # 读取图像尺寸
    origin = cv2.imread(os.path.join(object_origin_dataroot.format('image_2'), pc_file[:-4] + '.png'))
    img_size = origin.T.shape[1:]
    # img_size = Image.open(os.path.join(object_origin_dataroot.format('image_2'), pc_file[:-4] + '.png')).size
    print(bin_file_path, img_size)
    # 投影
    pc = load_pc(bin_file_path)
    img = project_lidar_to_image(pc, img_size, calib_file_path, yaw_deg=yaw_deg, method='zbuffer')
    # 裁剪到仅有lidar的部分
    img = img[120:, ...]

    cv2.imshow('', img.astype(np.uint8))
    cv2.waitKey(0)
    cv2.imwrite('/tmp/{}.png'.format(time.time()), img)
