#include <pcl/filters/passthrough.h>
#include <pcl/surface/impl/mls.hpp>
#include <pcl/segmentation/impl/sac_segmentation.hpp>
#include <pcl/filters/extract_indices.h>
#include <fstream>

typedef pcl::PointCloud<pcl::PointXYZI> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudPtr;
typedef pcl::PointXYZI XYZI;

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
 * load KITTI bin format pointcloud
 * @param path file path
 * @return
 */
std::vector<std::vector<float>> load_KITTI_pointcloud(const std::string &path) {
    std::ifstream fin;
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


/**
 * save pcl pointcloud to KITTI bin format
 * @param pc pointcloud to save
 * @param path saving path
 */
void save_KITTI_pointcloud(const PointCloudPtr &pc, const std::string &path) {
    std::ofstream fout;
    fout.open(path, std::ios::out | std::ios::binary);
    for (auto &p:*pc) {
        fout.write(reinterpret_cast<char *>(&(p.x)), sizeof(float));
        fout.write(reinterpret_cast<char *>(&(p.y)), sizeof(float));
        fout.write(reinterpret_cast<char *>(&(p.z)), sizeof(float));
        fout.write(reinterpret_cast<char *>(&(p.intensity)), sizeof(float));
    }
    fout.close();
}


/**
 * mls上采样
 * @param input 输入点云
 * @param radius 每个点云上采样时扩张的半径 单位是m
 * @param step_size 在radius中增加的点的距离 单位是m
 * @param search_radius kdtree的搜索半径
 * @return 上采样之后的点云
 */
PointCloudPtr
mls_upsample(PointCloudPtr &input,
             float radius = 0.09,
             float step_size = 0.03,
             float search_radius = 0.3,
             unsigned num_threads = 4) {
    // resample
    pcl::search::KdTree<XYZI>::Ptr tree(new pcl::search::KdTree<XYZI>);

    PointCloudPtr mls_points(new PointCloud);

//    pcl::MovingLeastSquares<XYZI, XYZI> mls;
    pcl::MovingLeastSquaresOMP<XYZI, XYZI> mls(num_threads);

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


/**
 * filter ground
 * @param input 输入点云
 * @param distance_th 分割阈值 值越大 地面越多 地上物体残留的地面越少
 * @return
 */
FilterGroundResult filter_ground(PointCloudPtr input, double distance_th) {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<XYZI> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distance_th);
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

PointCloudPtr ground_generation(
        PointCloudPtr &raw_ground,
        pcl::ModelCoefficients::Ptr ground_plane_coef,
        float ground_size = 50,
        float ground_interval = 0.1
) {
    // generate pure plane ground
    PointCloudPtr surface(new PointCloud);
    // x starts from 0, assume we only generate frontview
    for (float x = 0; x < ground_size; x += ground_interval) {
        for (float y = -ground_size; y < ground_size; y += ground_interval) {
            float z = (-ground_plane_coef->values[3] - ground_plane_coef->values[1] * x -
                       ground_plane_coef->values[0] * y) / ground_plane_coef->values[2];
            XYZI new_point;
            new_point.x = x;
            new_point.y = y;
            new_point.z = z;
            new_point.intensity = 0;

            surface->points.push_back(new_point);
        }
    }

    // limit groud to the same shape of up ground objects
    PointCloudPtr new_ground(new PointCloud);

    pcl::KdTreeFLANN<XYZI>::Ptr tree(new pcl::KdTreeFLANN<XYZI>);
    tree->setInputCloud(raw_ground);
    auto K = 3;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    for (auto &p : *surface) {
        // find nearest points
        tree->nearestKSearch(p, K, pointIdxNKNSearch, pointNKNSquaredDistance);
        float sum = std::accumulate(pointNKNSquaredDistance.begin(), pointNKNSquaredDistance.end(), float(0));
//        std::cout << sum << std::endl;
        if (sum < 10) {
            // intensity IDW interpolation
            float weight = 0;
            for (int k = 0; k < K; ++k) {
                auto neighbor = raw_ground->points[pointIdxNKNSearch[k]];
                auto dist = pointNKNSquaredDistance[k];
                weight += neighbor.intensity * dist;
            }
            p.intensity = weight / sum;
            new_ground->points.push_back(p);
        }
    }
//    *new_ground += *raw_ground;
    return new_ground;
}
