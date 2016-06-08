#include <ros/ros.h>

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

ros::Publisher pub0, pub1, pub2, pub3, pub4, pub5;
float xmin, xmax, ymin, ymax, zmin, zmax;
bool verbose;

void passthrough_z(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered)
{
    // Create the filtering object
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(*cloud_filtered));
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.5, 1.1);
    pass.filter (*cloud_filtered);
}

void passthrough_x(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered)
{
    // Create the filtering object
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(*cloud_filtered));
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (-1.0, 1.0);
    pass.filter (*cloud_filtered);
}

void passthrough_y(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered)
{
    // Create the filtering object
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (boost::make_shared<pcl::PointCloud<pcl::PointXYZ> >(*cloud_filtered));
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (-1.0, 3.0);
    pass.filter (*cloud_filtered);
}

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& input)
{

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*input, *cloud);
  if (verbose == true) {
    std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*
  }

  // Get Min & Max
  pcl::PointXYZ min_full;
  pcl::PointXYZ max_full;
  pcl::getMinMax3D (*cloud, min_full, max_full);

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud (cloud);
  vg.setLeafSize (0.01f, 0.01f, 0.01f);
  vg.filter (*cloud_filtered);
  if (verbose == true) {
    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*
  }

  // Create the passthrough filter object: downsample the dataset using a leaf size of 1cm
  passthrough_z(cloud_filtered);
  passthrough_x(cloud_filtered);
  passthrough_y(cloud_filtered);
  sensor_msgs::PointCloud2::Ptr cloud_filtered_ros (new sensor_msgs::PointCloud2 ());
  pcl::toROSMsg(*cloud_filtered, *cloud_filtered_ros);
  cloud_filtered_ros->header.frame_id = input->header.frame_id;

  pub3.publish(*cloud_filtered_ros);

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.02);

  // Segment the largest planar component from the remaining cloud
  seg.setInputCloud (cloud_filtered);
  seg.segment (*inliers, *coefficients);

  // Extract the planar inliers from the input cloud
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud (cloud_filtered);
  extract.setIndices (inliers);
  extract.setNegative (false);

  // Get the points associated with the planar surface
  extract.filter (*cloud_plane);
  if (verbose == true) {
    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;
  }

  // Publish plane points
  sensor_msgs::PointCloud2::Ptr cloud_plane_ros (new sensor_msgs::PointCloud2 ());
  pcl::toROSMsg(*cloud_plane, *cloud_plane_ros);
  cloud_plane_ros->header.frame_id = input->header.frame_id;

  pub0.publish(*cloud_plane_ros);

  // Remove the planar inliers, extract the rest
  extract.setNegative (true);
  extract.filter (*cloud_f);

  // Publish outlier points
  sensor_msgs::PointCloud2::Ptr cloud_outlier_ros (new sensor_msgs::PointCloud2 ());
  pcl::toROSMsg(*cloud_f, *cloud_outlier_ros);
  cloud_outlier_ros->header.frame_id = input->header.frame_id;

  pub1.publish(*cloud_outlier_ros);

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_f);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.05); // 2cm
  ec.setMinClusterSize (50);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_f);
  ec.extract (cluster_indices);

  if (verbose == true) {
    std::cout << "Number of clusters : " << cluster_indices.size() << '\n';
  }

  // seed rng for cloud colors
  srand (100);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr full_cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    int r = rand() % 256;
    int b = rand() % 256;
    int g = rand() % 256;
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
      cloud_cluster->points.push_back (cloud_f->points[*pit]); //*
      pcl::PointXYZRGB color_point = pcl::PointXYZRGB(r, g, b);
      color_point.x = cloud_f->points[*pit].x;
      color_point.y = cloud_f->points[*pit].y;
      color_point.z = cloud_f->points[*pit].z;
      full_cloud_cluster->points.push_back(color_point);
    }

    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    full_cloud_cluster->width = full_cloud_cluster->width + cloud_cluster->points.size ();
    full_cloud_cluster-> height = 1;
    full_cloud_cluster->is_dense = true;

    // Get centroid
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid (*cloud_cluster, centroid);

    // Get Min & Max
    pcl::PointXYZ min;
    pcl::PointXYZ max;
    pcl::getMinMax3D (*cloud_cluster, min, max);

    geometry_msgs::PolygonStamped poly;
    geometry_msgs::PolygonStamped poly_depth;
    geometry_msgs::Point32 ul, ll, lr, ur, cen;
    ul.x = min.x; ul.y = max.y; ul.z = centroid[2]; // max.z;
    ll.x = min.x; ll.y = min.y; ll.z = centroid[2]; // max.z;
    lr.x = max.x; lr.y = min.y; lr.z = centroid[2]; // max.z;
    ur.x = max.x; ur.y = max.y; ur.z = centroid[2]; // max.z;
    cen.x = centroid[0]; cen.y = centroid[1]; cen.z = centroid[2];

    poly.polygon.points.push_back(ul);
    poly.polygon.points.push_back(ll);
    poly.polygon.points.push_back(lr);
    poly.polygon.points.push_back(ur);
    poly.header.frame_id = input->header.frame_id;

    pub4.publish(poly);

    poly_depth.polygon.points.push_back(ul);
    poly_depth.polygon.points.push_back(lr);
    poly_depth.polygon.points.push_back(cen);
    poly_depth.header.frame_id = input->header.frame_id;

    pub5.publish(poly_depth);

    if (verbose == true) {
      std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    }
  }

  sensor_msgs::PointCloud2::Ptr cloud_cluster_ros (new sensor_msgs::PointCloud2 ());
  pcl::toROSMsg(*full_cloud_cluster, *cloud_cluster_ros);
  cloud_cluster_ros->header.frame_id = input->header.frame_id;

  pub2.publish(*cloud_cluster_ros);

}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "geom_region_proposal_node");
  ros::NodeHandle nh;

  // Load parameters
  nh.getParam("/geom_region_proposal/xmin", xmin);
  nh.getParam("/geom_region_proposal/xmax", xmax);
  nh.getParam("/geom_region_proposal/ymin", ymin);
  nh.getParam("/geom_region_proposal/ymax", ymax);
  nh.getParam("/geom_region_proposal/zmin", zmin);
  nh.getParam("/geom_region_proposal/zmax", zmax);
  nh.getParam("/geom_region_proposal/verbose", verbose);

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("camera/depth_registered/points", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pub0 = nh.advertise<sensor_msgs::PointCloud2> ("plane_inliers", 1);
  pub1 = nh.advertise<sensor_msgs::PointCloud2> ("plane_outliers", 1);
  pub2 = nh.advertise<sensor_msgs::PointCloud2> ("object_clusters", 1);
  pub3 = nh.advertise<sensor_msgs::PointCloud2> ("cloud_filtered", 1);
  pub4 = nh.advertise<geometry_msgs::PolygonStamped> ("candidate_regions", 1);
  pub5 = nh.advertise<geometry_msgs::PolygonStamped> ("candidate_regions_depth", 1);


  // Spin
  ros::spin ();
}
