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
#include <algorithm>

namespace py=pybind11;


py::array_t<float> parallel_add(py::array_t<float> input) {
    auto array = input.mutable_unchecked<2>();
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < array.shape(0); ++i) {
        array(i, 1) += pow(sin(i), 2) * cos(i) + 1;
    }
    return input;
}

py::array_t<float> mls_upsample_kernel(py::array_t<float> input, unsigned num_threads = 4) {
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
//    std::cout << "INPUT SHAPE: " << ref_input.shape(0) << " " << ref_input.shape(1) << std::endl;

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
//    std::cout << "RESULT SHAPE: " << result.shape(0) << " " << result.shape(1) << std::endl;

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

py::array_t<float> ray_tracing_interpolation_kernel(
        const py::array_t<float> &pc,
        const py::array_t<float> &dists,
        const py::array_t<float> &indices,
        int K_interp,
        int num_threads = 4
) {
    assert(dists.shape(0) == indices.shape(0));
    // 存储结果的array
    const int num_unknown_points = dists.shape(0), num_neighbors = dists.shape(1);
    const int data_field = 2;
    auto unknown = py::array_t<float>(py::array::ShapeContainer(
            {(const long) (num_unknown_points), data_field}
    ));
    float *unknown_buffer = (float *) unknown.request().ptr;

    // 访问ref
    auto ref_pc = pc.unchecked<2>(), ref_dists = dists.unchecked<2>(), ref_indices = indices.unchecked<2>();

    // 遍历所有点插值
#pragma omp parallel for num_threads(num_threads)
    for (int point_id = 0; point_id < num_unknown_points; ++point_id) {
        // 找到当前点邻居在2d投影上对应的深度
        std::vector<float> known_depth, known_intensity;
        for (int i = 0; i < num_neighbors; ++i) {
            known_depth.push_back(
                    ref_pc(ref_indices(point_id, i), 2)
            );
            known_intensity.push_back(
                    ref_pc(ref_indices(point_id, i), 3)
            );
        }

        // 对known_depth进行排序
        auto ind = argsort(known_depth);

        // 用idw的邻居距离权重来计算当前未知点的数值
        // 注意这里选择的邻居是深度最浅的几个邻居点
        float sumN_depth = 0, sumN_intensity = 0, sumD = 0;
        for (int ni = 0; ni < K_interp; ++ni) {
            // 获取深度最浅的邻居点的idx
            int shallow_neighbor_idx = ind[ni];

            // 计算idw权重
            float w = 1 / std::pow(ref_dists(point_id, shallow_neighbor_idx), 4);
            sumN_depth += w * known_depth[shallow_neighbor_idx];
            sumN_intensity += w * known_intensity[shallow_neighbor_idx];
            sumD += w;
        }

        unknown_buffer[point_id * data_field] = sumN_depth / sumD;
        unknown_buffer[point_id * data_field + 1] = sumN_intensity / sumD;
    }
    return unknown;
}

PYBIND11_MODULE(upsample_ext, m) {
    m.doc() = "Pointcloud upsample kernels.";

    m.def("test", &parallel_add, "numpy add function");

    m.def("mls_upsample_kernel", &mls_upsample_kernel, "upsample pointcloud using MLS upsample",
          py::arg("input"), py::arg("num_threads") = 4);

    m.def("ray_tracing_interpolation_kernel", &ray_tracing_interpolation_kernel,
          py::arg("pc"), py::arg("dists"), py::arg("indices"), py::arg("K_interp"), py::arg("num_threads") = 4,
          "ray-tracing interpolation");
}
