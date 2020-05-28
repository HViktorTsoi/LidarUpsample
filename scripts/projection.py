#!/home/hviktortsoi/miniconda3/bin/python
# coding=utf-8
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

import upsample_ext

MAX_DEPTH = 60
NUM_THREADS = 72


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


def zbuffer_projection(pc, depth, data, img_size):
    """
    serial implement of zbuffer
    :param pc: input pointcloud
    :param img_size: image size
    :return: rendered img
    """
    z_buffer = np.zeros(img_size[::-1]) + 1e9
    img = np.zeros(img_size[::-1] + (3,))
    proj_coord = np.int_(pc[:, :2])
    for point_idx, coord in enumerate(proj_coord):
        # 深度小于当前深度时才进行投影
        if depth[point_idx] < z_buffer[coord[1], coord[0]]:
            # 处理深度
            z_buffer[coord[1], coord[0]] = depth[point_idx]
            # 处理数据
            # img[coord[1], coord[0], 0] = (pc[point_idx, 3] + 0.1) * 255
            img[coord[1], coord[0], 1] = (np.maximum(0, MAX_DEPTH - pc[point_idx, 2]) / MAX_DEPTH) ** 1.1 * 255
            img[coord[1], coord[0], 2] = (pc[point_idx, 5] + 126) % 255

    return img


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
    intensity = intensity[indice]
    # 还原pc
    pc = np.concatenate([pc[:, :3], intensity], axis=1)

    # 上采样点云
    pc = upsample_ext.upsample(pc, num_threads=NUM_THREADS)
    # pc = pc[np.where(pc[:, -1] == 128)]

    # 备份其他特征
    ground = np.copy(pc[:, 4]).reshape(-1, 1)
    intensity = np.copy(pc[:, 3]).reshape(-1, 1)
    height = np.copy(pc[:, 2]).reshape(-1, 1)
    # 进行投影变换
    pc = np.concatenate([pc[:, :3], np.ones_like(pc[:, 0]).reshape(-1, 1)], axis=1)
    pc = np.matmul(calib_mat, pc.T).T

    # z深度归一化
    pc[:, :2] /= pc[:, 2:]

    # 还原intensity
    pc = np.concatenate([pc, intensity, height, ground], axis=1)

    # 按照原图大小裁剪
    pc = pc[np.where(pc[:, 0] >= 0)]
    pc = pc[np.where(pc[:, 0] < img_size[0])]
    pc = pc[np.where(pc[:, 1] >= 0)]
    pc = pc[np.where(pc[:, 1] < img_size[1])]

    if method == 'colormap':
        # 方法1 对点云做color map之后再绘制到前视图上
        # img = color_map(pc, img_size)
        pass
    elif method == 'points':
        # 方法2 下边是不对投影之后的点云做任何处理，直接以点的形式绘制到前视图上
        img = np.zeros(img_size[::-1] + (3,))
        # BGR
        img[np.int_(pc[:, 1]), np.int_(pc[:, 0]), 1] = (np.maximum(0, MAX_DEPTH - pc[:, 2]) / MAX_DEPTH) ** 1.1  # depth
        # img[np.int_(pc[:, 1]), np.int_(pc[:, 0]), 1] = (pc[:, 3] + 0.1)  # intensity
        img[np.int_(pc[:, 1]), np.int_(pc[:, 0]), 2] = (pc[:, 5] + 126) % 255 / 255  # intensity
        img = np.int_(img * 255)
    elif method == 'interp':
        # 方法3 interp
        img = interp(pc, img_size)
    elif method == 'zbuffer':
        # 方法4 point-zbuffer投影
        img = zbuffer_projection(
            pc,
            depth=pc[:, 2],
            data=pc[:, 2:],
            img_size=img_size
        )
        # img[:, :, 2] = zbuffer_projection(pc, img_size)
    else:
        img = None

    return img


def get_object_file_list(root):
    object_origin_dataroot = root
    file_list = []
    for pc_file in os.listdir(object_origin_dataroot.format('velodyne')):
        file_list.append((
            os.path.join(object_origin_dataroot.format('calib'), pc_file[:-4] + '.txt'),
            os.path.join(object_origin_dataroot.format('velodyne'), pc_file),
            os.path.join(object_origin_dataroot.format('image_2'), pc_file[:-4] + '.png'),
            pc_file[:-4]
        ))
    return file_list


def process_task(calib_file_path, bin_file_path, img_path, file_id):
    # 载入图像 点云
    origin = cv2.imread(img_path)
    img_size = origin.T.shape[1:]
    pc = load_pc(bin_file_path)

    # 投影
    tic = time.time()
    img = project_lidar_to_image(pc, img_size, calib_file_path, yaw_deg=yaw_deg, method='zbuffer')
    toc = time.time()
    print('{}: time used: {}s'.format(file_id, toc - tic))
    # img = project_lidar_to_image(pc, img_size, calib_file_path, yaw_deg=yaw_deg, method='points')
    # 裁剪到仅有lidar的部分
    img = img[120:, ...]

    # 保存结果
    # cv2.imshow('', img.astype(np.uint8))
    # cv2.waitKey(0)
    cv2.imwrite(
        os.path.join(OUTPUT_PATH, '{}.png'.format(file_id)),
        img,
        [cv2.IMWRITE_PNG_COMPRESSION, 0]  # 原图质量
    )


if __name__ == '__main__':
    INPUT_PATH = '/home/bdbc201/dataset/KITTI/object/training/{}/'
    OUTPUT_PATH = '/home/bdbc201/dataset/cgan/mls'
    file_list = get_object_file_list(root=INPUT_PATH)
    # object
    yaw_deg = 0

    pool = multiprocessing.Pool(32)
    for file in sorted(file_list):
        calib_file_path, bin_file_path, img_path, file_id = file
        pool.apply_async(process_task, args=(calib_file_path, bin_file_path, img_path, file_id,))
        # process_task(calib_file_path, bin_file_path, img_path, file_id)
    pool.close()
    pool.join()
