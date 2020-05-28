//
// Created by hviktortsoi on 20-5-27.
//
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include "upsample.hpp"

namespace py=pybind11;


py::array_t<float> parallel_add(py::array_t<float> input) {
    auto array = input.mutable_unchecked<2>();
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < array.shape(0); ++i) {
        array(i, 1) += pow(sin(i), 2) * cos(i) + 1;
    }
    return input;
}

py::array_t<float> upsample(py::array_t<float> input, unsigned num_threads = 4) {
    auto ref_input = input.unchecked<2>();
    // 初始化pointcloud 数量是输入的numpy array中的point数量
    PointCloudPtr cloud(new PointCloud(ref_input.shape(0), 1));
//#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < ref_input.shape(0); ++i) {
        cloud->points[i].x = ref_input(i, 0);
        cloud->points[i].y = ref_input(i, 1);
        cloud->points[i].z = ref_input(i, 2);
        cloud->points[i].intensity = ref_input(i, 3);
    }
    std::cout << "INPUT SHAPE: " << ref_input.shape(0) << " " << ref_input.shape(1) << std::endl;

    // filter ground
    FilterGroundResult segmentation = filter_ground(cloud, 0.2);

    // upsample
    PointCloudPtr cloud_upsampled = mls_upsample(segmentation.non_ground, 0.1, 0.03, 0.3, num_threads);

    // ground generation
//    PointCloudPtr ground = segmentation.ground;
//    PointCloudPtr ground = ground_generation(
//            segmentation.ground,
//            segmentation.coef,
//            50, 0.1);
    PointCloudPtr ground = mls_upsample(segmentation.ground, 0.3, 0.15, 0.9, num_threads);

    // 返回结果
    int data_field = 5;
    auto result = py::array_t<float>(py::array::ShapeContainer(
            {(const long) (cloud_upsampled->size() + ground->size()), data_field}
    ));
    std::cout << "RESULT SHAPE: " << result.shape(0) << " " << result.shape(1) << std::endl;

//    // 这里为了效率 直接把cloud_upsampled的data memcopy到return的buffer中
//    // 这里的假定是每个PointXYZI由4个float32组成，且return array是n x 4大小的float32 array
//    py::buffer_info buf = result.request();
//    memcpy((float *) buf.ptr, cloud_upsampled->points.data(), sizeof(float) * cloud_upsampled->size());
    float *buf = (float *) result.request().ptr;
    // 非地面点
    for (int i = 0; i < cloud_upsampled->size(); ++i) {
        int buf_index_base = i * data_field;
        buf[buf_index_base + 0] = cloud_upsampled->points[i].x;
        buf[buf_index_base + 1] = cloud_upsampled->points[i].y;
        buf[buf_index_base + 2] = cloud_upsampled->points[i].z;
        buf[buf_index_base + 3] = cloud_upsampled->points[i].intensity;
        buf[buf_index_base + 4] = 128.0;
    }
    // 地面点
    for (int i = 0; i < ground->size(); ++i) {
        int buf_index_base = (cloud_upsampled->size() + i) * data_field;
        buf[buf_index_base + 0] = ground->points[i].x;
        buf[buf_index_base + 1] = ground->points[i].y;
        buf[buf_index_base + 2] = ground->points[i].z;
        buf[buf_index_base + 3] = ground->points[i].intensity;
        buf[buf_index_base + 4] = 255.0;
    }
    return result;
}

PYBIND11_MODULE(upsample_ext, m) {
    m.doc() = "MLS upsample.";
    m.def("test", &parallel_add, "numpy add function");
    m.def("upsample", &upsample, "upsample pointcloud using MLS upsample",
          py::arg("input"), py::arg("num_threads") = 4);
}
