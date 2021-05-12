#include "vision_youbot/youbot_utils_library.h"

namespace youbot_utils
{

	float pointHeightInRobotFrame(float x, float y, float z, geometry_msgs::TransformStamped& camToBase)
	{
		//tranform a point between two frames
		geometry_msgs::Point oldP;
		geometry_msgs::Point newP;
		oldP.x = x;
		oldP.y = y;
		oldP.z = z;
		tf2::doTransform(oldP, newP, camToBase);
		
		return newP.z;
	}
	
	float getPlaneHeight(std::vector<pcl::PointXYZRGB> planePoints, geometry_msgs::TransformStamped& camToBase)
	{
		//get a median height of plane points (average plane height)
		float median = 0;
		for(auto p : planePoints)
		{
			float tp = pointHeightInRobotFrame(p.x, p.y, p.z, camToBase);
			median += tp;
	    }
	    
		return median/planePoints.size();
	}
	
	float getObjectHeight(int xMin, int yMin, int width, int height, float planeHeight, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, geometry_msgs::TransformStamped& camToBase)
	{
		float maxHeight = 0;
		
		//iterate over a cropped cloud
		int rowMax = std::min(yMin+height, 479);
		int colMax = std::min(xMin+width, 639);
		
		for(int i=yMin; i<rowMax; i++)
		{
			for(int j=xMin; j<colMax; j++)
			{
				//calculate height of points in robot frame and take the highest
				pcl::PointXYZRGB p = cloud->at(j,i);
				if(std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z))
				{
				  continue;
				} 
				else 
				{
				  float pixHeight = pointHeightInRobotFrame(p.x, p.y, p.z, camToBase);
				  maxHeight = std::max(maxHeight, pixHeight);
				}
				
			}
		}
		//substract plane height to get an actual height of the object only
		float objectHeight = maxHeight - planeHeight;
		
		return objectHeight;
	}
	
	void getObjectState(float objectHeight, float objectWidth, int classNumber, vision_youbot::DetectedObj& objMsg)
	{
		std::string name = "";
		int state = 0;
		
		//convert to mm
		objectHeight = objectHeight*1000;
		objectWidth = objectWidth*1000;
		
		/* 
		Map class from YOLO name to Rulebook notation 
		based on object`s height and width
		https://github.com/robocup-at-work/rulebook
	
		Screw 0 M20_100
		Nut 1 M20/M30
		Axis 2 AXIS
		Bearing 3 BEARING
		BearingBox 4 BEARING_BOX
		BlackProfile 5 F20_20_B/S40_40_B
		DistanceTube 6 DISTANCE_TUBE
		GreyProfile 7 F20_20_G/S40_40_G
		Motor 8 MOTOR
		PlasticTube 9 R20
		BlueContainer 10 CONTAINER_BLUE
		RedContainer 11 CONTAINER_RED
		*/
		switch(classNumber)
		{
			case 0:
				if(objectHeight > 72.25)
				{
					state = 1;
				}
				name = "M20_100";
				break;
			case 1:
				if(objectHeight < 21)
				{
					name = "M20";
				}
				else if(objectHeight < 43)
				{
					if(objectWidth < 43)
					{
						name = "M20";
						state = 1;
					} 
					else
					{
						name = "M30";
					}
				}
				else
				{
					name = "M30";
					state = 1;
				}
				break;
			case 2:
				if(objectHeight > 61.5)
				{
					state = 1;
				}
				name = "AXIS";
				break;
			case 3:
				if(objectHeight > 22.5)
				{
					state = 1;
				}
				name = "BEARING";
				break;
			case 4:
				if(objectHeight > 35)
				{
					state = 1;
				}
				name = "BEARING_BOX";
				break;
			case 5:
				if(objectHeight < 38)
				{
					name = "F20_20_B";
				}
				else if(objectHeight < 90)
				{
					name = "S40_40_B";
				}
				else if(objectWidth < 38)
				{
					name = "F20_20_B";
					state = 1;
				}
				else
				{
					name = "S40_40_B";
					state = 1;
				}
				break;
			case 6:
				if(objectHeight > 21)
				{
					state = 1;
				}
				name = "DISTANCE_TUBE";
				break;
			case 7:
				if(objectHeight < 38)
				{
					name = "F20_20_G";
				}
				else if(objectHeight < 90)
				{
					name = "S40_40_G";
				}
				else if(objectWidth < 38)
				{
					name = "F20_20_G";
					state = 1;
				}
				else
				{
					name = "S40_40_G";
					state = 1;
				}
				break;
			
			case 8:
				if(objectHeight > 64.5)
				{
					state = 1;
				}
				name = "MOTOR";
				break;
			case 9:
				if(objectHeight > 37.5)
				{
					state = 1;
				}
				name = "R20";
				break;
			case 10:
				name = "CONTAINER_BLUE";
				break;
			case 11:
				name = "CONTAINER_RED";
				break;
		}
		
		//fill a ros object message with a defined name and state
		objMsg.name = name;
		objMsg.state = state;
	}
	
	
	//inverse of pinhole pixel projection 
	//https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
	cv::Point2f getPointOnPlane(cv::Point uv, float planeHeight, Eigen::Matrix<double,3,3,Eigen::RowMajor> inv_cam_intrinsics, geometry_msgs::TransformStamped& camToBase)
	{
		//get a ray in camera frame
		Eigen::Vector3d pix (uv.x, uv.y, 1);
		Eigen::Vector3d cameraRayEigen = inv_cam_intrinsics*pix;
    
		//map a ray to robot frame
		geometry_msgs::Vector3 cameraRay;
		cameraRay.x = cameraRayEigen(0);
		cameraRay.y = cameraRayEigen(1);
		cameraRay.z = cameraRayEigen(2);
		geometry_msgs::Vector3 baseRay;
		tf2::doTransform(cameraRay, baseRay, camToBase);
		
		//get scale factor
		float cameraHeight = camToBase.transform.translation.z;
	    float scale = (planeHeight -cameraHeight) / baseRay.z;
	 
		//get coordinates on plane
	    float x = camToBase.transform.translation.x + baseRay.x * scale;
        float y = camToBase.transform.translation.y + baseRay.y * scale;
        
        cv::Point2f planePoint (x, y);
        return planePoint;
	}
    
 
    float getObjectWidth(cv::Point centerPix, cv::Point sizePix, float massCenterHeight, Eigen::Matrix<double,3,3,Eigen::RowMajor> inv_cam_intrinsics, geometry_msgs::TransformStamped& camToBase)
    {
		//get a distance in robot frame between centroid and sizepoint
		cv::Point2f ce = youbot_utils::getPointOnPlane(centerPix, massCenterHeight*2, inv_cam_intrinsics, camToBase);
		cv::Point2f si = youbot_utils::getPointOnPlane(sizePix, massCenterHeight*2, inv_cam_intrinsics, camToBase);
		double width = cv::norm(ce-si);
		
		//multiply by 2 to get an object`s width
		width = width*2;
		
		std::cout << "WIDTH " << width << std::endl;
		
		return width;				
	}
}
