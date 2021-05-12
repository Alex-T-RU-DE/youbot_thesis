//tkDNN libs
#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <mutex>

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"

//ros libs
#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

//ros tf
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

//opencv
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//messages
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>

//pcl
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

//custom cv libs
#include "vision_youbot/DetectedObj.h"
#include "vision_youbot/DetectedObjArr.h"
#include "vision_youbot/youbot_pcl_library.h"
#include "vision_youbot/youbot_utils_library.h"

class RosYolo
{
public:
    //detection results publisher
    ros::NodeHandle n_;
    ros::Publisher yoloBoxesPub_;
	
    //camera intrinsics
    Eigen::Matrix3d invCamIntrinsics_;
	
    //cvbridge frames
    cv_bridge::CvImagePtr rgbFramePtr_;
	
    //synchronized subscribers
    message_filters::Subscriber<sensor_msgs::Image> rgbSub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> pcloudSub_;
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2> sync_;
	
    //pcl variables
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloudFramePtr_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredcloudPtr_;
    pcl::visualization::PCLVisualizer::Ptr viewer_;
	
    //tf variables
    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener tfListener_;
    geometry_msgs::TransformStamped currentCamInBaseFrame_;
    float currentPlaneHeight_ = 0;
    
     //flags to show cloud and camera 
    bool showImage_ = false;
    bool showCloud_ = false;
    
    //tkDNN default params
	std::string net;
    int n_classes;
    int n_batch;
    float conf_thresh;
    
	tk::dnn::Yolo3Detection yolo;
    tk::dnn::DetectionNN *detNN; 
    
    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;
    
   
    
    RosYolo(bool show_im, bool show_cl) : rgbSub_(n_, "/camera/color/image_raw", 1),
					  pcloudSub_(n_, "/camera/depth_registered/points",1),
					  sync_(rgbSub_, pcloudSub_, 10),
					  pcloudFramePtr_(new pcl::PointCloud<pcl::PointXYZRGB>),
					  filteredcloudPtr_(new pcl::PointCloud<pcl::PointXYZRGB>),
					  tfListener_(tfBuffer_)
    { 
	//get flags
	showImage_ = show_im;
	showCloud_ = show_cl;

	//get inverted camera intrinsics for deprojection
	std::cout << "waiting for camera intrinsics" << std::endl;
	sensor_msgs::CameraInfo::ConstPtr camInfo = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/camera/aligned_depth_to_color/camera_info",n_);   
	std::cout << "intrinsics received" << std::endl;   

	Eigen::Matrix<double,3,3,Eigen::RowMajor> camIntrinsics ( camInfo->K.data() );
	invCamIntrinsics_ = camIntrinsics.inverse();

	//init publishers and subscribers
	sync_.registerCallback(&RosYolo::cameraCallback_, this);
	yoloBoxesPub_ = n_.advertise<vision_youbot::DetectedObjArr>("detected_objects", 1);

	//create cloud viewer
	if(showCloud_)
	{
		pcl::visualization::PCLVisualizer::Ptr v (new pcl::visualization::PCLVisualizer ("3D Viewer"));
		viewer_ = v;

		viewer_->setBackgroundColor(0, 0, 0);
		viewer_->addCoordinateSystem();
		viewer_->initCameraParameters();
	}

	//create image window
	if(showImage_)
	{
		cv::namedWindow("detection", cv::WINDOW_NORMAL);
	}

	//init tkDNN
	net = "yolo4_custom_fp16.rt";
	n_classes = 2;
	n_batch = 1;
	conf_thresh=0.3;   

	detNN = &yolo;
	detNN->init(net, n_classes, n_batch, conf_thresh);
    }

    ~RosYolo()
    {
	  
    }
    
    //callback from camera
    void cameraCallback_(const sensor_msgs::Image::ConstPtr& rgbmsg, const sensor_msgs::PointCloud2ConstPtr& pcloudmsg)
    {
		//get tf
		try
		{	
			currentCamInBaseFrame_ = tfBuffer_.lookupTransform("base_ground", "arm_camera_link", ros::Time(0));
		} 
		catch (tf2::TransformException &ex) 
		{
			ROS_WARN("Could NOT transform arm_camera_link to base_ground: %s", ex.what());
			return;
		}

		//parse images from cvbridge
		try
		{
		  rgbFramePtr_ = cv_bridge::toCvCopy(rgbmsg, sensor_msgs::image_encodings::BGR8); 
		}
		catch (cv_bridge::Exception& e)
		{
		  ROS_ERROR("cv_bridge exception: %s", e.what());
		  return;
		}

		//parse point cloud
		pcl::fromROSMsg(*pcloudmsg, *pcloudFramePtr_);
		youbot_pcl::cloudPreprocess(pcloudFramePtr_, filteredcloudPtr_);

		if(filteredcloudPtr_->size() >= 10)
		{
			//get height of a service area
			std::vector<pcl::PointXYZRGB> planePoints = youbot_pcl::planarSegmentation(filteredcloudPtr_);
			currentPlaneHeight_ = youbot_utils::getPlaneHeight(planePoints, currentCamInBaseFrame_);

			//prepare image
			batch_dnn_input.clear();
			batch_frame.clear();
			for(int bi=0; bi< n_batch; ++bi)
			{
				batch_frame.push_back(rgbFramePtr_->image);
				batch_dnn_input.push_back(rgbFramePtr_->image.clone());
			} 

			//execute yolo
			detNN->update(batch_dnn_input, n_batch);
			detNN->drawYOLO(currentPlaneHeight_, currentCamInBaseFrame_, invCamIntrinsics_, pcloudFramePtr_, batch_frame, yoloBoxesPub_, showImage_);

			//show cloud
			if(showCloud_)
			{
				youbot_pcl::updateCloudViewer(viewer_, filteredcloudPtr_);
			}

			//show processed image
			if(showImage_)
			{  
				for(int bi=0; bi< n_batch; ++bi)
				{
					cv::imshow("detection", batch_frame[bi]);
					cv::waitKey(1);
				}
			}
		}
	}
};

int main(int argc, char *argv[]) 
{
    //init ros node
    ros::init(argc, argv, "jetson_yolo_node");
 
    //get args
	bool showImage_ = false;
	bool showCloud_ = false;
    
    std::cout<<"detection end\n"; 
    
    
	if(argc > 2)
        showImage_ = atoi(argv[2]); 
    if(argc > 3)
        showCloud_ = atoi(argv[3]);
         
    //init 
    RosYolo ry(showImage_, showCloud_);
    
	ros::spin();

	//evaluate tkDNN
    std::cout<<"detection end\n";   
    double mean = 0; 
    
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(ry.detNN->stats.begin(), ry.detNN->stats.end())/ry.n_batch<<" ms\n";    
    std::cout<<"Max: "<<*std::max_element(ry.detNN->stats.begin(), ry.detNN->stats.end())/ry.n_batch<<" ms\n";    
    for(int i=0; i<ry.detNN->stats.size(); i++) mean += ry.detNN->stats[i]; mean /= ry.detNN->stats.size();
    std::cout<<"Avg: "<<mean/ry.n_batch<<" ms\t"<<1000/(mean/ry.n_batch)<<" FPS\n"<<COL_END;   
    

    return 0;
}

