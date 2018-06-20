#ifndef GEOM_REGION_PROPOSAL_UPDATED_H
#define GEOM_REGION_PROPOSAL_UPDATED_H

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>

#include <geom_rcnn/GeomValsConfig.h>

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PolygonStamped.h>
#include <geometry_msgs/Point32.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/filters/project_inliers.h>
#include <pcl/surface/concave_hull.h>

#include <pcl/segmentation/extract_polygonal_prism_data.h>


// callbacks
void reconfig_cb(const geom_rcnn::GeomValsConfig &config, uint32_t level);
void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input);

// helper functions
void passthrough_x(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered);
void passthrough_y(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered);
void passthrough_z(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered);

// TUNABLE PARAMS - see dynamic reconfigure file for default values
double xmin = -0.4;
double xmax = 1.7;
double ymin = -1.4;
double ymax = 1.4;
double zmin = -0.3;
double zmax = 1.8;
double euc_cluster_tolerance = 0.4;
double euc_cluster_min = 30;
double euc_cluster_max = 25000;
double concave_hull_alpha = 0.1;
double segmentation_distance_thresh = 0.2;

#endif
