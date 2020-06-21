#!/home/hviktortsoi/DevTools/anaconda3/envs/hydro/bin/python
# coding=utf-8

import sys

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append('/home/hviktortsoi/Code/pointcloud_upsample')
from scripts.projection import process_task, get_tracking_file_list, get_object_file_list

MAX_DEPTH = 60
NUM_THREADS = 72

if __name__ == '__main__':
    INPUT_PATH = '/media/hvt/95f846d8-d39c-4a04-8b28-030feb1957c6/dataset/KITTI/tracking/training/{}'
    file_list = get_tracking_file_list(root=INPUT_PATH)

    # object
    yaw_deg = 0

    file = file_list[0]
    calib_file_path, bin_file_path, img_path, file_id = file
    process_task(calib_file_path, bin_file_path, img_path, file_id,
                 method='ray', yaw_deg=yaw_deg, visualize=True)
    # process_task(calib_file_path, bin_file_path, img_path, file_id,
    #              method='interp', yaw_deg=yaw_deg, visualize=True)
    # process_task(calib_file_path, bin_file_path, img_path, file_id,
    #              method='zbuffer', yaw_deg=yaw_deg, visualize=True)
    # process_task(calib_file_path, bin_file_path, img_path, file_id,
    #              method='points', yaw_deg=yaw_deg, visualize=True)
