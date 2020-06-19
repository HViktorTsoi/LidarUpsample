#!/home/hviktortsoi/miniconda3/bin/python
# coding=utf-8
import multiprocessing
import os
import time

import numpy as np
import array
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.use('TKAgg', warn=False, force=True)
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2
from PIL import Image
from scipy import spatial

# import upsample_ext

MAX_DEPTH = 60
NUM_THREADS = 72


def interp(pc, img_size):
    """
    pointcloud front-view interpolation
    :param pc:
    :param img_size:
    :return:
    """
    # assert pc.shape[1] == 5, 'pointcloud must has 3 channel features:{depth, intensity, height}'

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


def color_map(pc, img_size):
    """
    将前视投影之后的点云绘制到image上
    :param pc: 输入点云
    :param img_size: 图像大小
    :return:
    """
    # 构建mask
    mask = np.zeros([img_size[1], img_size[0]], dtype=np.uint8)
    mask[np.int_(pc[:, 1]), np.int_(pc[:, 0])] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 泛洪算法 填补图像中的空洞
    flood_fill = mask.copy().astype(np.uint8)
    cv2.floodFill(flood_fill, np.zeros((img_size[1] + 2, img_size[0] + 2), np.uint8), (0, 0), 255)
    mask = mask | cv2.bitwise_not(flood_fill)

    colors = plt.get_cmap('gist_ncar_r')(np.maximum((100 - pc[:, 2]) / 100, 0))
    # colors = plt.get_cmap('hot')(1.5 * pc[:, 3] ** 1.3)
    grid_x, grid_y = np.mgrid[0:img_size[0]:1, 0:img_size[1]:1]
    chs = [griddata(pc[:, 0:2], colors[:, 2 - idx], (grid_x, grid_y), method='nearest').T for idx in range(3)]
    img = np.stack(chs, axis=-1)
    img = np.int_(img * 255)

    # 和mask向掩码
    img = img * np.expand_dims(mask / 255, -1)
    return img


def ray_tracing_projection(pc, img_size):
    # 查找多少个邻域点
    K_search = 8
    # 实际用多少个点插值
    K_interp = 4
    # 结果图像
    img = np.zeros(img_size[::-1] + (3,))

    # 生成2d rays
    rays = np.array([[row, col] for row in range(img_size[0]) for col in range(img_size[1])])

    # 构造kd tree
    tree = spatial.cKDTree(pc[:, :2])

    # 查找近邻点
    dists, indices = tree.query(rays, k=K_search, )
    print(dists, indices)

    new_depth = []
    for (x, y), dist, indice in zip(rays, dists, indices):
        # 如果有太多inf 直接抛弃 不参与计算
        if len(np.where(dist == np.inf)[0]) > 0:
            new_depth.append(100)
            continue

        known_depth = pc[indice, 2]
        known_intensity = pc[indice, 3]

        # ind = np.argsort(known_depth)
        # # 查看最大值和最小值之间的差距
        # depth = abs(known_depth.mean() - known_depth[ind][:K_interp].mean()) - \
        #         abs(known_depth[ind][K_interp:].mean() - known_depth.mean())
        # depth = abs(known_depth[ind][:K_interp].mean() - known_depth.mean())
        # depth = abs(known_depth[ind][-K_interp:].mean() - known_depth.mean())
        # depth = abs(known_depth.mean())
        # depth = np.maximum((100 - depth) / 100, 0)

        # 仅使用深度最浅的几个点
        ind = np.argpartition(known_depth, K_interp)
        known_depth = known_depth[ind][:K_interp]
        dist = dist[ind][:K_interp]
        known_intensity = known_intensity[ind][:K_interp]

        w = 1 / dist ** 2
        depth = np.sum(known_depth * w) / np.sum(w)
        depth = np.maximum((100 - depth) / 100, 0)

        # intensity = np.sum(known_intensity * w) / np.sum(w)
        # intensity += 0.1

        # new_depth.append(intensity)
        new_depth.append(depth)
    # colors = plt.get_cmap('hot')(new_depth)
    colors = plt.get_cmap('gist_ncar_r')(new_depth)
    for idx in range(3):
        img[rays[:, 1], rays[:, 0], 2 - idx] = colors[:, idx] * 255

    # 构建mask
    mask = np.zeros([img_size[1], img_size[0]], dtype=np.uint8)
    mask[np.int_(pc[:, 1]), np.int_(pc[:, 0])] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 泛洪算法 填补图像中的空洞
    flood_fill = mask.copy().astype(np.uint8)
    cv2.floodFill(flood_fill, np.zeros((img_size[1] + 2, img_size[0] + 2), np.uint8), (0, 0), 255)
    mask = mask | cv2.bitwise_not(flood_fill)
    # 和mask向掩码
    img = img * np.expand_dims(mask / 255, -1)

    # # 边缘检测
    # gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # edge = cv2.Laplacian(gray, cv2.CV_32F)  # 拉普拉斯边缘检测
    # edge = np.uint8(np.absolute(edge))  ##对lap去绝对值
    # # edge = cv2.Canny(gray, 30, 150)
    #
    # cv2.imshow('edge', edge)
    # cv2.waitKey(0)
    return img


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
            img[coord[1], coord[0], 0] = (pc[point_idx, 3] + 0.2) * 255
            img[coord[1], coord[0], 1] = (np.maximum(0, MAX_DEPTH - pc[point_idx, 2]) / MAX_DEPTH) ** 1.1 * 255
            img[coord[1], coord[0], 2] = (pc[point_idx, 5] + 126) % 255
            # img[coord[1], coord[0], 2] = (255 - pc[point_idx, 5]) * 2

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
    pc = np.concatenate([pc, intensity], axis=1)
    # pc = upsample_ext.upsample(pc, num_threads=NUM_THREADS)
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
        # img[np.int_(pc[:, 1]), np.int_(pc[:, 0]), 2] = (255 - pc[:, 5]) * 2 / 255  # intensity
        img = np.int_(img * 255)
    elif method == 'interp':
        # 方法3 interp
        # img = interp(pc, img_size)
        img = color_map(pc, img_size)
    elif method == 'zbuffer':
        # 方法4 point-zbuffer投影
        img = zbuffer_projection(
            pc,
            depth=pc[:, 2],
            data=pc[:, 2:],
            img_size=img_size
        )
        # img[:, :, 2] = zbuffer_projection(pc, img_size)
    elif method == 'ray':
        # 方法5 ray tracing插值
        img = ray_tracing_projection(pc, img_size)
    else:
        img = None

    return img


def get_object_file_list(root):
    """
    获取kitti object目录下的所有文件
    :param root:
    :return:
    """
    file_list = []
    for pc_file in os.listdir(root.format('velodyne')):
        file_list.append((
            os.path.join(root.format('calib'), pc_file[:-4] + '.txt'),
            os.path.join(root.format('velodyne'), pc_file),
            os.path.join(root.format('image_2'), pc_file[:-4] + '.png'),
            pc_file[:-4]
        ))
    return file_list


def get_tracking_file_list(root):
    """
    获取kitti tracking目录格式下的所有文件
    :param root:
    :return:
    """
    # 提取tracking中的所有文件
    pc_list = ['{}/{}'.format(path, file) for path in sorted(os.listdir(root.format('velodyne')))
               for file in sorted(os.listdir(os.path.join(root.format('velodyne'), path)))]

    file_list = []
    for pc_file in pc_list:
        file_list.append((
            os.path.join(root.format('calib'), pc_file[:4] + '.txt'),
            os.path.join(root.format('velodyne'), pc_file),
            os.path.join(root.format('image_02'), pc_file[:-4] + '.png'),
            pc_file[:-4].replace('/', '_')  # 文件id去掉目录分割符
        ))
    return file_list


def process_task(calib_file_path, bin_file_path, img_path, file_id, yaw_deg=0, method='zbuffer', visualize=False):
    # 载入图像 点云
    origin = cv2.imread(img_path)
    img_size = origin.T.shape[1:]
    pc = load_pc(bin_file_path)

    # 投影
    tic = time.time()
    img = project_lidar_to_image(pc, img_size, calib_file_path, yaw_deg=yaw_deg, method=method)
    toc = time.time()
    print('{}: time used: {}s'.format(file_id, toc - tic))
    # 裁剪到仅有lidar的部分
    img = img[120:, ...]

    # 保存结果
    if visualize:
        cv2.imshow('', img.astype(np.uint8))
        cv2.waitKey(0)
        cv2.imwrite('/tmp/{}.png'.format(time.time()), img)
    else:
        # print(OUTPUT_PATH, '{}.png'.format(file_id))
        # print(img.shape)
        cv2.imwrite(
            os.path.join(OUTPUT_PATH, '{}.png'.format(file_id)),
            img,
            [cv2.IMWRITE_PNG_COMPRESSION, 0]  # 原图质量
        )


if __name__ == '__main__':
    # INPUT_PATH = '/home/bdbc201/dataset/KITTI/object/training/{}/'
    # OUTPUT_PATH = '/home/bdbc201/dataset/cgan/mls/object_train'
    # file_list = get_object_file_list(root=INPUT_PATH)

    INPUT_PATH = '/home/bdbc201/dataset/KITTI/tracking/training/{}/'
    OUTPUT_PATH = '/home/bdbc201/dataset/cgan/mls/tracking_train'
    file_list = get_tracking_file_list(root=INPUT_PATH)

    # INPUT_PATH = '/home/bdbc201/dataset/KITTI/object/testing/{}/'
    # OUTPUT_PATH = '/home/bdbc201/dataset/cgan/mls/object_test'
    # file_list = get_object_file_list(root=INPUT_PATH)

    # INPUT_PATH = '/home/bdbc201/dataset/KITTI/tracking/testing/{}/'
    # OUTPUT_PATH = '/home/bdbc201/dataset/cgan/mls/tracking_test'
    # file_list = get_tracking_file_list(root=INPUT_PATH)

    # output_list = os.listdir(OUTPUT_PATH)
    # file_list = [item for item in file_list if item[2][-10:] not in output_list]
    # file_list = sorted(file_list)
    # print(file_list)

    file_list = sorted(file_list)

    # file_list = file_list[7000:]
    # object
    yaw_deg = 0

    pool = multiprocessing.Pool(32)
    for file in file_list:
        calib_file_path, bin_file_path, img_path, file_id = file
        pool.apply_async(process_task, args=(calib_file_path, bin_file_path, img_path, file_id, yaw_deg))
        # process_task(calib_file_path, bin_file_path, img_path, file_id)
    pool.close()
    pool.join()
