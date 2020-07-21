#include <iostream>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <cmath>
#include <vector>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/surface/impl/mls.hpp>
#include <pcl/surface/impl/gp3.hpp>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <algorithm>
#include <pcl/features/normal_3d.h>
#include <pcl/features/impl/normal_3d.hpp>
#include "upsample.hpp"
#include "mls_lidar.hpp"

template<typename TPointType>
typename pcl::PointCloud<TPointType>::Ptr vector_to_pointcloud(std::vector<std::vector<float >> pc) {
    typename pcl::PointCloud<TPointType>::Ptr pcl_pc(new pcl::PointCloud<TPointType>);
    for (auto &i : pc) {
        TPointType point;
        point.x = i[0];
        point.y = i[1];
        point.z = i[2];
        point.intensity = i[3] * 255;

        pcl_pc->points.push_back(point);
    }
    return pcl_pc;
}

/**
 * visulize pointcloud
 * @param cloud
 */
void visualize(PointCloudPtr cloud) {
    // vis
    pcl::visualization::CloudViewer viewer("Demo viewer");
    viewer.showCloud(cloud);
    while (!viewer.wasStopped()) {}
}

/**
 * 分割出视野ROI
 * @param input
 * @param fov
 * @return
 */
PointCloudPtr crop_ROI(const PointCloudPtr &input, float fov, float forward_distance = 200, float height_limit = 8) {

    // filter
    pcl::PassThrough<XYZI> pass;
    pass.setInputCloud(input);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(0, forward_distance);
    pass.filter(*input);

    pass.setInputCloud(input);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-2, height_limit);
    pass.filter(*input);


    PointCloudPtr output(new PointCloud);
    for (auto &p:*input) {
        if (abs(atan(p.y / p.x)) < fov) {
            output->push_back(p);
        }
    }
    return output;
}


PointCloudPtr range_partition(PointCloudPtr input, int start, int end) {
    // 设置比较条件
    pcl::ConditionAnd<XYZI>::Ptr cond(new pcl::ConditionAnd<XYZI>);
    cond->addComparison(pcl::FieldComparison<XYZI>::Ptr(
            new pcl::FieldComparison<XYZI>("x", pcl::ComparisonOps::GE, start)));
    cond->addComparison(pcl::FieldComparison<XYZI>::Ptr(
            new pcl::FieldComparison<XYZI>("x", pcl::ComparisonOps::LT, end)));

    pcl::ConditionalRemoval<XYZI> cond_rmv;
    cond_rmv.setCondition(cond);
    cond_rmv.setInputCloud(input);

    // filter
    PointCloudPtr output(new PointCloud);
    cond_rmv.filter(*output);
    return output;
}

PointCloudPtr noise_remove(PointCloudPtr &input) {
    // Create the filtering object
    pcl::StatisticalOutlierRemoval<XYZI> sor;
    sor.setInputCloud(input);
    sor.setMeanK(50);
    sor.setStddevMulThresh(0.5);
    sor.filter(*input);
    return input;
}

PointCloudPtr range_interpolation(PointCloudPtr input) {
    float parameter[][5] = {
            {0,  10,  0.09, 0.03, 0.3},
            {10, 60,  0.6,  0.2,  2.0},
            {60, 200, 1.5,  0.5,  9.00},
    };
    PointCloudPtr output(new PointCloud);
    for (auto &param :parameter) {
        PointCloudPtr part = range_partition(input, param[0], param[1]);
        // upsample
        PointCloudPtr part_upsample = mls_upsample(part, param[2], param[3], param[4]);

        // noise remove
        part_upsample = noise_remove(part_upsample);

        *output += *part_upsample;
    }
    return output;
}

void spatial_visualize(const PointCloudPtr &input) {
    // visualize
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer);
    pcl::visualization::PointCloudColorHandlerGenericField<XYZI> color(input, "x");
    viewer->addPointCloud(input, color, "raw");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

void mesh_visualize(const PointCloudPtr &input) {
    // segmentation

    auto ground_segmentation = filter_ground(input, 0.2);
    // upsample points
//    *input = *mls_upsample(ground_segmentation.non_ground, 0.09, 0.06);
    *input = *ground_segmentation.non_ground;
    std::cout << input->size() << std::endl;

    // compute normal
    pcl::NormalEstimation<XYZI, pcl::Normal> ne;
    ne.setInputCloud(input);
    pcl::search::KdTree<XYZI>::Ptr tree(new pcl::search::KdTree<XYZI>);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(0.6);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*normals);

    // compute triangle
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::concatenateFields(*input, *normals, *cloud_with_normals);
    pcl::search::KdTree<pcl::PointXYZINormal>::Ptr normal_tree(new pcl::search::KdTree<pcl::PointXYZINormal>);
    normal_tree->setInputCloud(cloud_with_normals);

    pcl::GreedyProjectionTriangulation<pcl::PointXYZINormal> gp3;
    pcl::PolygonMesh triangles;
    gp3.setSearchRadius(0.2);
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(20);
    gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
    gp3.setMinimumAngle(M_PI / 18); // 10 degrees
    gp3.setMaximumAngle(2 * M_PI / 3); // 120 degrees
    gp3.setNormalConsistency(false);

    gp3.setSearchMethod(normal_tree);
    gp3.setInputCloud(cloud_with_normals);
    gp3.reconstruct(triangles);

    // visualize
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer);
    pcl::visualization::PointCloudColorHandlerGenericField<XYZI> color(input, "x");
    viewer->addPointCloud(input, color, "raw");
    viewer->addPointCloudNormals<XYZI, pcl::Normal>(input, normals, 10, 0.05, "normals");
    viewer->addPolygonMesh(triangles, "mesh");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();


    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

}

PointCloudPtr ada_mls_upsample(PointCloudPtr cloud,
                               float radius, float step, float weight, float max_up_radius, int order = 3,
                               float voxel_leaf_size = 0.03, bool remove_vertical = false) {
    PointCloudPtr mls_points(new PointCloud), mls_points_voxel(new PointCloud);

    MovingLeastSquaresLiDAR<XYZI, XYZI> mls;
    pcl::search::KdTree<XYZI>::Ptr tree(new pcl::search::KdTree<XYZI>);
    mls.setInputCloud(cloud);
    mls.setPolynomialOrder(order);
    mls.setComputeNormals(true);
    mls.setSearchRadius(0.3);
    mls.setSearchMethod(tree);

    mls.setUpsamplingMethod(pcl::MovingLeastSquares<XYZI, XYZI>::SAMPLE_LOCAL_PLANE);
    mls.setUpsamplingRadius(radius);
    mls.setUpsamplingStepSize(step);
    mls.setRadiusWeight(weight);
    mls.setRadiusMax(max_up_radius);
    mls.setRemoveVerticalObject(remove_vertical);

    mls.process(*mls_points);

    std::cout << mls_points->size() << std::endl;

//    // 去除超出roi之外的点
//    mls_points = crop_ROI(mls_points, 3.14159 / 4, 200);

    // voxel 平滑
    pcl::VoxelGrid<XYZI> vg;
    vg.setInputCloud(mls_points);
    vg.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    vg.filter(*mls_points_voxel);
    std::cout << mls_points_voxel->size() << std::endl;

//     还原intensity
    intensity_propagation(cloud, mls_points_voxel);

//    visualize(mls_points);
//    spatial_visualize(mls_points);
    return mls_points_voxel;
}

//PointCloud fille_plane()

int main(int argc, char *argv[]) {

    // load pointcloud
    auto cloud = vector_to_pointcloud<XYZI>(load_KITTI_pointcloud(std::string(argv[1])));
//    visualize(cloud);
//    mesh_visualize(cloud);
//    return 0;

    // 在限制roi之前进行分割 获取更准确的地面
    FilterGroundResult segmentation = filter_ground(cloud, 0.2);
    auto landscape = segmentation.non_ground, ground = segmentation.ground;

    // crop reigon
    landscape = crop_ROI(landscape, 3.14159 / 4, 200);
    ground = crop_ROI(ground, 3.14159 / 4, 200);
    std::cout << landscape->size() << std::endl;
    std::cout << ground->size() << std::endl;
//    visualize(cloud);

    auto landscape_upsampled = mls_upsample(landscape);
//    auto ground_upsampled = mls_upsample(ground, 0.4, 0.2, 1.0, 5, 3, 0.05, true);

//    auto landscape_upsampled = ada_mls_upsample(landscape, 0.09, 0.03, 1.0, 1, 5, 0.03);
//    auto ground_upsampled = ada_mls_upsample(ground, 0.4, 0.2, 1.0, 5, 3, 0.05, true);

//    auto landscape_upsampled = ada_mls_upsample(landscape, 0.1, 0.03, 1.5, 3, 5, 0.05);
//    auto ground_upsampled = ada_mls_upsample(ground, 0.6, 0.2, 4, 5, 3, 0.05, true);
//    auto ground_upsampled = ground;

//    // reconstruct ground
//    auto ground = ground_generation(segmentation.ground, segmentation.coef, 50, 0.1);
////    auto ground = segmentation.ground;
//    std::cout << ground->size() << std::endl;
////    visualize(landscape);

    // concat and write
//    *landscape_upsampled += *ground;
//    *landscape_upsampled += *ground_upsampled;

    visualize(landscape_upsampled);

    save_KITTI_pointcloud(landscape_upsampled, std::string(argv[2]));

    return 0;
}