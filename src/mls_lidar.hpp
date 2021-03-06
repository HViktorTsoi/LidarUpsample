#include <iostream>
#include <pcl/surface/impl/mls.hpp>
#include <pcl/point_traits.h>
#include <pcl/surface/mls.h>
#include <pcl/common/io.h>
#include <pcl/common/copy_point.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/geometry.h>


template<typename PointInT, typename PointOutT>
class MovingLeastSquaresLiDAR : public pcl::MovingLeastSquares<PointInT, PointOutT> {
public:
    typedef boost::shared_ptr<pcl::MovingLeastSquares<PointInT, PointOutT> > Ptr;
    typedef boost::shared_ptr<const pcl::MovingLeastSquares<PointInT, PointOutT> > ConstPtr;

    using pcl::PCLBase<PointInT>::input_;
    using pcl::PCLBase<PointInT>::indices_;
    using pcl::MovingLeastSquares<PointInT, PointOutT>::normals_;
    using pcl::MovingLeastSquares<PointInT, PointOutT>::corresponding_input_indices_;
    using pcl::MovingLeastSquares<PointInT, PointOutT>::nr_coeff_;
    using pcl::MovingLeastSquares<PointInT, PointOutT>::order_;
    using pcl::MovingLeastSquares<PointInT, PointOutT>::compute_normals_;
    using pcl::MovingLeastSquares<PointInT, PointOutT>::upsample_method_;
    using pcl::MovingLeastSquares<PointInT, PointOutT>::VOXEL_GRID_DILATION;
    using pcl::MovingLeastSquares<PointInT, PointOutT>::DISTINCT_CLOUD;

    typedef pcl::PointCloud<pcl::Normal> NormalCloud;
    typedef pcl::PointCloud<pcl::Normal>::Ptr NormalCloudPtr;

    typedef pcl::PointCloud<PointOutT> PointCloudOut;
    typedef typename PointCloudOut::Ptr PointCloudOutPtr;
    typedef typename PointCloudOut::ConstPtr PointCloudOutConstPtr;

    MovingLeastSquaresLiDAR() :
            radius_weight_(1.0), radius_max_(10.0), remove_vertical_object_(false) {}

    inline void setRadiusWeight(float weight) { radius_weight_ = weight; }

    inline void setRadiusMax(float radiusMax) { radius_max_ = radiusMax; }

    inline void setRemoveVerticalObject(bool flag) { remove_vertical_object_ = flag; }

private:

    /** 上采样半径随着距离增长的剧烈程度 **/
    float radius_weight_;
    float radius_max_;
    bool remove_vertical_object_;

private:

    inline float ada_radius_kernel(float base, float distance) {
        // 远处的限制最大半径 防止半径过大导致崩溃
        return std::min(std::max(base * (float) std::pow(distance / 10, 2) * radius_weight_, base), radius_max_);
    }

    inline float ada_step_kernel(float base, float distance) {
//        return std::max(base, base * (float) std::pow(distance / 40, 2));
        return std::max(base, base * (float) std::pow(distance / 15, 2));
    }

protected:
    /** \brief Abstract surface reconstruction method.
        * \param[out] output the result of the reconstruction
        */
    virtual void performProcessing(PointCloudOut &output);
};

template<typename PointInT, typename PointOutT>
void MovingLeastSquaresLiDAR<PointInT, PointOutT>::performProcessing(PointCloudOut &output) {

    std::cout << "Input size: " << this->input_->points.size() << std::endl;
    std::cout << "Order: " << this->order_ << std::endl;


    // Compute the number of coefficients
    nr_coeff_ = (order_ + 1) * (order_ + 2) / 2;

    // Allocate enough space to hold the results of nearest neighbor searches
    // \note resize is irrelevant for a radiusSearch ().
    std::vector<int> nn_indices;
    std::vector<float> nn_sqr_dists;

    size_t mls_result_index = 0;

    float base_search_radius = this->search_radius_,
            base_upsample_radius = this->upsampling_radius_,
            base_upsample_step = this->upsampling_step_;

    // 此处indices就是输入点云中所有的点的索引
    // For all points
    for (size_t cp = 0; cp < indices_->size(); ++cp) {

        int index = (*indices_)[cp];

        // 计算当前点到中心距离
        float distance = Eigen::Vector2f(
                this->input_->points[index].x,
                this->input_->points[index].y
//                this->input_->points[index].z
        ).norm();

        // 设置搜索的半径
        this->search_radius_ = ada_radius_kernel(base_search_radius, distance);
        // Get the initial estimates of point positions and their neighborhoods
        if (!this->searchForNeighbors((*indices_)[cp], nn_indices, nn_sqr_dists))
            continue;


        // Check the number of nearest neighbors for normal estimation (and later
        // for polynomial fit as well)
        if (nn_indices.size() < 3)
            continue;


        PointCloudOut projected_points, filtered_projected_points;
        NormalCloud projected_points_normals, filtered_projected_points_normals;
        // Get a plane approximating the local surface's tangent and project point onto it

        if (upsample_method_ == VOXEL_GRID_DILATION || upsample_method_ == DISTINCT_CLOUD)
            mls_result_index = index; // otherwise we give it a dummy location.

        // 设置upsampling radius和step
        this->upsampling_radius_ = ada_radius_kernel(base_upsample_radius, distance);
        this->upsampling_step_ = ada_step_kernel(base_upsample_step, distance);


        this->computeMLSPointNormal(index, nn_indices, nn_sqr_dists, projected_points, projected_points_normals,
                                    *corresponding_input_indices_, this->mls_results_[mls_result_index]);


        // 这里得到的intensity粗粒度的，错误的结果，直接复制了代表点的特征到整个plane上 ，不应该被使用
        // Copy all information from the input cloud to the output points (not doing any interpolation)
        // 仅保留非异常点
        for (size_t pp = 0; pp < projected_points.size(); ++pp) {
//            this->copyMissingFields(input_->points[(*indices_)[cp]], projected_points[pp]);
            // 过滤掉采样点是垂直的点
            if (this->remove_vertical_object_) {
                float normal_vert = std::abs(projected_points_normals.points[pp].normal_x) +
                                    std::abs(projected_points_normals.points[pp].normal_y);
                if (normal_vert > 0.7) {
                    continue;
                }
//                projected_points[pp].intensity = normal_vert;
            }

            // 过滤掉采样点到原始点过远的点
            float local_distance = Eigen::Vector3f(
                    projected_points.points[pp].x - input_->points[(*indices_)[cp]].x,
                    projected_points.points[pp].y - input_->points[(*indices_)[cp]].y,
                    projected_points.points[pp].z - input_->points[(*indices_)[cp]].z
            ).norm();
            if (local_distance * 1.2 > this->upsampling_radius_) {
                continue;
            }

            // 加入到新点集
            filtered_projected_points.push_back(projected_points.points[pp]);
            if (compute_normals_)
                filtered_projected_points_normals.push_back(projected_points_normals.points[pp]);

        }

        // Append projected points to output
        output.insert(output.end(), filtered_projected_points.begin(), filtered_projected_points.end());
        if (compute_normals_)
            normals_->insert(normals_->end(), filtered_projected_points_normals.begin(),
                             filtered_projected_points_normals.end());
    }

    // Perform the distinct-cloud or voxel-grid upsampling
    this->performUpsampling(output);

}

