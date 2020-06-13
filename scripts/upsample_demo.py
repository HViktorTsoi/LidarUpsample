#!/home/hviktortsoi/miniconda3/bin/python
# coding=utf-8

from scripts.projection import process_task, get_tracking_file_list, get_object_file_list

MAX_DEPTH = 60
NUM_THREADS = 72

if __name__ == '__main__':
    INPUT_PATH = '/home/hviktortsoi/data/KITTI/{}'
    file_list = get_tracking_file_list(root=INPUT_PATH)

    # object
    yaw_deg = 60

    file = file_list[80]
    calib_file_path, bin_file_path, img_path, file_id = file
    process_task(calib_file_path, bin_file_path, img_path, file_id,
                 method='ray', yaw_deg=yaw_deg, visualize=True)
    # process_task(calib_file_path, bin_file_path, img_path, file_id,
    #              method='interp', yaw_deg=yaw_deg, visualize=True)
    # process_task(calib_file_path, bin_file_path, img_path, file_id,
    #              method='zbuffer', yaw_deg=yaw_deg, visualize=True)
    # process_task(calib_file_path, bin_file_path, img_path, file_id,
    #              method='points', yaw_deg=yaw_deg, visualize=True)
