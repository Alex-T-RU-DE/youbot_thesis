#ifndef YOUBOT_OPENCV_LIBRARY_H
#define YOUBOT_OPENCV_LIBRARY_H
#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vision_youbot/DetectedObj.h>
#include <vision_youbot/DetectedObjArr.h>

namespace youbot_opencv
{
	std::vector<std::vector<cv::Point> > contoursAndCanny(cv::Mat &croppedImg);

	std::vector<cv::Point> findTheBiggestContour(std::vector<std::vector<cv::Point> > &contours);

	void drawOneContour(cv::Mat& img, std::vector<cv::Point>& c);

	std::vector<cv::Point> findCorners(std::vector<cv::Point>& c);

	std::vector<cv::Point> findConvexHull(std::vector<cv::Point>& c);

	cv::PCA pca (std::vector<cv::Point> &c);

	std::vector<cv::Point> PCAPose(cv::Mat &croppedImg, std::vector<cv::Point> &c);

	std::vector<cv::Point> rectPose (cv::Mat &croppedImg, std::vector<cv::Point> &c);

	std::vector<cv::Point> getObjectPose(float xs,float ys, cv::Mat croppedImg,  int classname, bool show_image);
}
#endif
