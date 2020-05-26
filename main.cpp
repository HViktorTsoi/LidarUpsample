#include <iostream>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <cmath>
#include <fstream>
#include <vector>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/surface/impl/mls.hpp>
#include <pcl/surface/impl/gp3.hpp>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/impl/sac_segmentation.hpp>
#include <algorithm>
#include <pcl/features/normal_3d.h>
#include <pcl/features/impl/normal_3d.hpp>

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudPtr;
typedef pcl::PointXYZI XYZI;

std::vector<std::vector<float>> load_KITTI_pointcloud(const std::string &path) {
    ifstream fin;
    fin.open(path, std::ios::in | std::ios::binary);
    std::vector<std::vector<float>> pointcloud;
    std::vector<float> point;

    const int data_stride = 4;
    int data_cnt = 0;
    float data;
    while (fin.peek() != EOF) {
        // read 4 bytes of data
        fin.read(reinterpret_cast<char *> (&data), sizeof(float));
        point.push_back(data);

        if (++data_cnt % data_stride == 0) {
            data_cnt = 0;
            pointcloud.push_back(point);
            point.clear();
        }
    }
    fin.close();
    return pointcloud;
}

void save_KITTI_pointcloud(const PointCloudPtr &pc, const std::string &path) {
    ofstream fout;
    fout.open(path, std::ios::out | std::ios::binary);
    for (auto &p:*pc) {
        fout.write(reinterpret_cast<char *>(&(p.x)), sizeof(float));
        fout.write(reinterpret_cast<char *>(&(p.y)), sizeof(float));
        fout.write(reinterpret_cast<char *>(&(p.z)), sizeof(float));
        fout.write(reinterpret_cast<char *>(&(p.intensity)), sizeof(float));
    }
    fout.close();
}

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

// limit横向视野
PointCloudPtr limit_fov(const PointCloudPtr &input, float fov) {
    PointCloudPtr output(new PointCloud);
    for (auto &p:*input) {
        if (abs(atan(p.y / p.x)) < fov) {
            output->push_back(p);
        }
    }
    return output;
}

void visualize(PointCloudPtr cloud) {
    // vis
    pcl::visualization::CloudViewer viewer("Demo viewer");
    viewer.showCloud(cloud);
    while (!viewer.wasStopped()) {}
}

/*
 * mls上采样
 *
 * @input 输入点云
 * @radius 每个点云上采样时扩张的半径 单位是m
 * @step_size 在radius中增加的点的距离 单位是m
 */
PointCloudPtr
mls_upsample(PointCloudPtr &input, float radius = 0.09, float step_size = 0.03, float search_radius = 0.3) {
    // resample
    pcl::search::KdTree<XYZI>::Ptr tree(new pcl::search::KdTree<XYZI>);

    PointCloudPtr mls_points(new PointCloud);

    pcl::MovingLeastSquares<XYZI, XYZI> mls;

    mls.setComputeNormals(true);
    mls.setInputCloud(input);
    mls.setPolynomialOrder(3);
    mls.setSearchMethod(tree);
    mls.setSearchRadius(search_radius);
    mls.setUpsamplingMethod(pcl::MovingLeastSquares<XYZI, XYZI>::SAMPLE_LOCAL_PLANE);
    mls.setUpsamplingRadius(radius);
    mls.setUpsamplingStepSize(step_size);
    mls.process(*mls_points);
    return mls_points;
}

PointCloudPtr ground_interpolation(PointCloudPtr &input, PointCloudPtr &unknown) {
    PointCloudPtr pc(new PointCloud);

    pcl::KdTreeFLANN<XYZI>::Ptr tree(new pcl::KdTreeFLANN<XYZI>);
    tree->setInputCloud(input);
    auto K = 3;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    for (auto &p : *unknown) {
        // find nearest points
        tree->nearestKSearch(p, K, pointIdxNKNSearch, pointNKNSquaredDistance);
        float sum = std::accumulate(pointNKNSquaredDistance.begin(), pointNKNSquaredDistance.end(), float(0));
//        std::cout << sum << std::endl;
        if (sum < 100) {
            // intensity IDW interpolation
            float weight = 0;
            for (int k = 0; k < K; ++k) {
                auto neighbor = input->points[pointIdxNKNSearch[k]];
                auto dist = pointNKNSquaredDistance[k];
                weight += neighbor.intensity * dist;
            }
            p.intensity = weight / sum;
//            std::cout << p.intensity << std::endl;
            pc->points.push_back(p);
        }
    }
//    visualize(pc);
    return pc;
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

struct FilterGroundResult {
    PointCloudPtr ground;
    PointCloudPtr non_ground;
    pcl::ModelCoefficients::Ptr coef;

    FilterGroundResult(const PointCloudPtr &ground, const PointCloudPtr &nonGround,
                       const pcl::ModelCoefficients::Ptr &coef) :
            ground(ground),
            non_ground(nonGround),
            coef(coef) {}
};

/**
 * filter ground
 * @param input 输入点云
 * @return
 */
FilterGroundResult filter_ground(PointCloudPtr input) {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<XYZI> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.2);// 值越大 地面越少
    seg.setMaxIterations(200);
    seg.setInputCloud(input);

    seg.segment(*inliers, *coefficients);

    for (auto &c:coefficients->values) {
        std::cout << c << " ";
    }
    std::cout << std::endl;

    PointCloudPtr landscape(new PointCloud), ground(new PointCloud);
    // extract points
    pcl::ExtractIndices<XYZI> extract;
    extract.setInputCloud(input);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*landscape);

    extract.setNegative(false);
    extract.filter(*ground);

    return FilterGroundResult(ground, landscape, coefficients);
}


void mesh_visualize(const PointCloudPtr &input) {
    // segmentation

    auto ground_segmentation = filter_ground(input);
    *input = *ground_segmentation.non_ground;

    // compute normal
    pcl::NormalEstimation<XYZI, pcl::Normal> ne;
    ne.setInputCloud(input);
    pcl::search::KdTree<XYZI>::Ptr tree(new pcl::search::KdTree<XYZI>);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(1);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*normals);

    // compute triangle
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::concatenateFields(*input, *normals, *cloud_with_normals);
    pcl::search::KdTree<pcl::PointXYZINormal>::Ptr normal_tree(new pcl::search::KdTree<pcl::PointXYZINormal>);
    normal_tree->setInputCloud(cloud_with_normals);

    pcl::GreedyProjectionTriangulation<pcl::PointXYZINormal> gp3;
    pcl::PolygonMesh triangles;
    gp3.setSearchRadius(1);
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(500);
    gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
    gp3.setMinimumAngle(M_PI / 18); // 10 degrees
    gp3.setMaximumAngle(2 * M_PI / 3); // 120 degrees
    gp3.setNormalConsistency(false);

    gp3.setSearchMethod(normal_tree);
    gp3.setInputCloud(cloud_with_normals);
    gp3.reconstruct(triangles);

    // visualize
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer);
//    pcl::visualization::PointCloudColorHandlerGenericField<XYZI> color(input, "x");
//    viewer->addPointCloud(input, color, "raw");
//    viewer->addPointCloudNormals<XYZI, pcl::Normal>(input, normals, 10, 0.05, "normals");
    viewer->addPolygonMesh(triangles, "mesh");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();


    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

}


//PointCloud fille_plane()

int main(int argc, char *argv[]) {

    // load pointcloud
    auto cloud = vector_to_pointcloud<XYZI>(load_KITTI_pointcloud(std::string(argv[1])));

    // filter
    pcl::PassThrough<XYZI> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(0, 200);
    pass.filter(*cloud);

    cloud = limit_fov(cloud, 3.14159 / 4);
    std::cout << cloud->size() << std::endl;

    mesh_visualize(cloud);
    return 0;

    FilterGroundResult segmentation = filter_ground(cloud);
    auto ground = segmentation.ground, landscape = segmentation.non_ground;
    auto coefficients = segmentation.coef;

    for (auto &c:coefficients->values) {
        std::cout << c << " ";
    }
    std::cout << std::endl;

    // generate plane
    PointCloudPtr plane(new PointCloud);
    float interval = 0.05;
    for (float x = 0; x < 50; x += interval) {
        for (float y = -50; y < 50; y += interval) {
            float z = (-coefficients->values[3] - coefficients->values[1] * x - coefficients->values[0] * y) /
                      coefficients->values[2];
            XYZI new_point;
            new_point.x = x;
            new_point.y = y;
            new_point.z = z;
            new_point.intensity = 0;

            plane->points.push_back(new_point);
        }
    }
    plane = limit_fov(plane, 3.14159 / 4);

    // reconstruct ground
    ground = ground_interpolation(ground, plane);
//    visualize(landscape);

    // resample landscape
    auto landscape_upsampled = mls_upsample(landscape, 0.09, 0.03);
//    auto landscape_upsampled = mls_upsample(landscape, 0.3, 0.1);

//    auto landscape_upsampled = range_interpolation(landscape);

//    landscape = range_partition(landscape, 30, 60);
//    auto landscape_upsampled = mls_upsample(landscape, 0.6, 0.1, 2.0);
//    landscape = range_partition(landscape, 60, 120);
//    auto landscape_upsampled = mls_upsample(landscape, 1.5, 0.5, 9.0);


//    // resample ground
//    auto ground_upsampled = mls_upsample(ground, 0.2, 0.05);

    // concat and write
    *landscape_upsampled += *ground;

    //    viewer.showCloud(cloud);
    visualize(landscape_upsampled);

    save_KITTI_pointcloud(landscape_upsampled, std::string(argv[2]));

    return 0;
}