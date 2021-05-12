#ifndef YOUBOT_PCL_LIBRARY_H
#define YOUBOT_PCL_LIBRARY_H
#include <ros/ros.h>
#include <iostream>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/surface/concave_hull.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

namespace youbot_pcl
{
	void updateCloudViewer(pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
	void cloudPreprocess(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredcloud);
	std::vector<pcl::PointXYZRGB> planarSegmentation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
}
#endif
