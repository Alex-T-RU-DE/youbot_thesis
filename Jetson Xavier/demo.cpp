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
#include "ros/ros.h"
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
#include "std_msgs/String.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>

//pcl
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

//our libs
#include "vision_youbot/DetectedObj.h"
#include "vision_youbot/DetectedObjArr.h"
#include "vision_youbot/youbot_pcl_library.h"
#include "vision_youbot/youbot_utils_library.h"

class RosYolo
{
public:
	//detection results publisher
	ros::NodeHandle n;
	ros::Publisher yolo_boxes_pub;
	
	//camera intrinsics
	Eigen::Matrix3d inv_cam_intrinsics;
	
	//cvbridge frames
	cv_bridge::CvImagePtr rgb_frame_ptr;
	
	//synchronized subscribers
	message_filters::Subscriber<sensor_msgs::Image> rgb_sub;
	message_filters::Subscriber<sensor_msgs::PointCloud2> pcloud_sub;
	message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2> sync;
	
	//pcl variables
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud_frame_ptr;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredcloud_ptr;
	pcl::visualization::PCLVisualizer::Ptr viewer;
	
	//tf variables
	tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener;
    geometry_msgs::TransformStamped currentCamInBaseFrame;
    float currentPlaneHeight = 0;
    
    //tkDNN params
	std::string net;
    int n_classes;
    int n_batch;
    float conf_thresh;
    
	tk::dnn::Yolo3Detection yolo;
    tk::dnn::DetectionNN *detNN; 
    
    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;
    
    //flags to show cloud and camera 
    bool show_image = false;
    bool show_cloud = false;
    
    RosYolo(bool show_im, bool show_cl) : rgb_sub(n, "/camera/color/image_raw", 1),
										  pcloud_sub(n, "/camera/depth_registered/points",1),
										  sync(rgb_sub, pcloud_sub, 10),
										  pcloud_frame_ptr(new pcl::PointCloud<pcl::PointXYZRGB>),
										  filteredcloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>),
										  tfListener(tfBuffer)
    { 
		//get flags
		show_image = show_im;
		show_cloud = show_cl;
		
		//get inverted camera intrinsics for deprojection
		std::cout << "waiting for camera intrinsics" << std::endl;
		sensor_msgs::CameraInfo::ConstPtr camInfo = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/camera/aligned_depth_to_color/camera_info",n);   
		std::cout << "intrinsics received" << std::endl;   
		
		Eigen::Matrix<double,3,3,Eigen::RowMajor> cam_intrinsics ( camInfo->K.data() );
		inv_cam_intrinsics = cam_intrinsics.inverse();
		
		//init publishers and subscribers
		sync.registerCallback(&RosYolo::cam_cb, this);
		yolo_boxes_pub = n.advertise<vision_youbot::DetectedObjArr>("detected_objects", 1);
		
		//create cloud viewer
		if(show_cloud)
		{
			pcl::visualization::PCLVisualizer::Ptr v (new pcl::visualization::PCLVisualizer ("3D Viewer"));
			viewer = v;
			
			viewer->setBackgroundColor(0, 0, 0);
			viewer->addCoordinateSystem();
			viewer->initCameraParameters();
		}
		
		//create image window
		if(show_image)
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
	void cam_cb(const sensor_msgs::Image::ConstPtr& rgbmsg, const sensor_msgs::PointCloud2ConstPtr& pcloudmsg)
	{
		//get tf
		try
		{	
			currentCamInBaseFrame = tfBuffer.lookupTransform("base_ground", "arm_camera_link", ros::Time(0));
		} 
		catch (tf2::TransformException &ex) 
		{
			ROS_WARN("Could NOT transform arm_camera_link to base_ground: %s", ex.what());
			return;
		}
		
		//parse images from cvbridge
		try
		{
		  rgb_frame_ptr = cv_bridge::toCvCopy(rgbmsg, sensor_msgs::image_encodings::BGR8); 
		}
		catch (cv_bridge::Exception& e)
		{
		  ROS_ERROR("cv_bridge exception: %s", e.what());
		  return;
		}
		
		//parse point cloud
		pcl::fromROSMsg(*pcloudmsg, *pcloud_frame_ptr);
		youbot_pcl::cloudPreprocess(pcloud_frame_ptr, filteredcloud_ptr);
		
		if(filteredcloud_ptr->size() >= 10)
		{
			//get height of a service area
			std::vector<pcl::PointXYZRGB> planePoints = youbot_pcl::planarSegmentation(filteredcloud_ptr);
			currentPlaneHeight = youbot_utils::getPlaneHeight(planePoints, currentCamInBaseFrame);
			
			//prepare image
			batch_dnn_input.clear();
			batch_frame.clear();
			for(int bi=0; bi< n_batch; ++bi)
			{
				batch_frame.push_back(rgb_frame_ptr->image);
				batch_dnn_input.push_back(rgb_frame_ptr->image.clone());
			} 
			   
			//execute yolo
			detNN->update(batch_dnn_input, n_batch);
			detNN->drawYOLO(currentPlaneHeight, currentCamInBaseFrame, inv_cam_intrinsics, pcloud_frame_ptr, batch_frame, yolo_boxes_pub, show_image);
			
			//show cloud
			if(show_cloud)
			{
				youbot_pcl::updateCloudViewer(viewer, filteredcloud_ptr);
			}
			
			//show processed image
			if(show_image)
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
	bool show_image = false;
	bool show_cloud = false;
    
    std::cout<<"detection end\n"; 
    
    
	if(argc > 2)
        show_image = atoi(argv[2]); 
    if(argc > 3)
        show_cloud = atoi(argv[3]);
         
    //init 
    RosYolo ry(show_image, show_cloud);
    
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

