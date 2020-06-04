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

    MovingLeastSquaresLiDAR() {}

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

    // 此处indices就是输入点云中所有的点的索引
    // For all points
    for (size_t cp = 0; cp < indices_->size(); ++cp) {

//        float distance =
        // TODO 在这里设置搜索的半径
//        this->search_radius_ =
        // Get the initial estimates of point positions and their neighborhoods
        if (!this->searchForNeighbors((*indices_)[cp], nn_indices, nn_sqr_dists))
            continue;


        // Check the number of nearest neighbors for normal estimation (and later
        // for polynomial fit as well)
        if (nn_indices.size() < 3)
            continue;


        PointCloudOut projected_points;
        NormalCloud projected_points_normals;
        // Get a plane approximating the local surface's tangent and project point onto it
        int index = (*indices_)[cp];

        if (upsample_method_ == VOXEL_GRID_DILATION || upsample_method_ == DISTINCT_CLOUD)
            mls_result_index = index; // otherwise we give it a dummy location.

        // TODO 在这里设置upsampling radius
        this->computeMLSPointNormal(index, nn_indices, nn_sqr_dists, projected_points, projected_points_normals,
                                    *corresponding_input_indices_, this->mls_results_[mls_result_index]);


        // Copy all information from the input cloud to the output points (not doing any interpolation)
        for (size_t pp = 0; pp < projected_points.size(); ++pp)
            this->copyMissingFields(input_->points[(*indices_)[cp]], projected_points[pp]);


        // Append projected points to output
        output.insert(output.end(), projected_points.begin(), projected_points.end());
        if (compute_normals_)
            normals_->insert(normals_->end(), projected_points_normals.begin(), projected_points_normals.end());
    }

    // Perform the distinct-cloud or voxel-grid upsampling
    this->performUpsampling(output);

}

