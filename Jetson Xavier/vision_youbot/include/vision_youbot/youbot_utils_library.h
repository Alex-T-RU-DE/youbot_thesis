#ifndef YOUBOT_UTILS_LIBRARY_H
#define YOUBOT_UTILS_LIBRARY_H
#include <iostream>

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h> 

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/PoseStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <vision_youbot/DetectedObj.h>
#include <vision_youbot/DetectedObjArr.h>

namespace youbot_utils
{
	float pointHeightInRobotFrame(float x, float y, float z, geometry_msgs::TransformStamped& camToBase);
	float getPlaneHeight(std::vector<pcl::PointXYZRGB>, geometry_msgs::TransformStamped& camToBase);
	float getObjectHeight(int xMin, int yMin, int width, int height, float planeHeight, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, geometry_msgs::TransformStamped& camToBase);
	void getObjectState(float objectHeight, float objectWidth, int classNumber, vision_youbot::DetectedObj& objMsg);
	cv::Point2f getPointOnPlane(cv::Point uv, float planeHeight, Eigen::Matrix<double,3,3,Eigen::RowMajor> inv_cam_intrinsics, geometry_msgs::TransformStamped& camToBase);
	float getObjectWidth(cv::Point centerPix, cv::Point widthPix, float massCenterHeight, Eigen::Matrix<double,3,3,Eigen::RowMajor> inv_cam_intrinsics, geometry_msgs::TransformStamped& camToBase);
	}
#endif
