#include "vision_youbot/youbot_pcl_library.h"
namespace youbot_pcl
{
	void updateCloudViewer(pcl::visualization::PCLVisualizer::Ptr viewer, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
	{
		viewer->removeAllPointClouds();

		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
		viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");

		viewer->spinOnce();
	}
	
	
	void cloudPreprocess(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredcloud)
	{
		//filter
		pcl::PassThrough<pcl::PointXYZRGB> Cloud_Filter; 
		Cloud_Filter.setInputCloud (cloud);           
		Cloud_Filter.setFilterFieldName ("z");       
		Cloud_Filter.setFilterLimits (0.0, 0.5);      
		Cloud_Filter.filter(*filteredcloud);
     
		//downsample
		pcl::VoxelGrid<pcl::PointXYZRGB> sor;
		sor.setInputCloud(filteredcloud);
		sor.setLeafSize(0.02f, 0.02f, 0.02f);
		sor.filter(*filteredcloud);
	}

	std::vector<pcl::PointXYZRGB> planarSegmentation(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
	{
		//segment plane
		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
		pcl::SACSegmentation<pcl::PointXYZRGB> seg;
		seg.setOptimizeCoefficients (true);
		seg.setModelType (pcl::SACMODEL_PLANE);
		seg.setMethodType (pcl::SAC_RANSAC);
		seg.setDistanceThreshold (0.002);
		seg.setInputCloud (cloud);
		seg.segment (*inliers, *coefficients);
		for (const auto& idx: inliers->indices)
		{
		cloud->points[idx].r = 255;
		cloud->points[idx].g = 0;
		cloud->points[idx].b = 0;
		}

		//get 10 random points of the plane
		std::vector<pcl::PointXYZRGB> planePoints;
		for(int i=0; i<10; i++)
		{
		  int randidx = std::rand() % inliers->indices.size();
		  int cind = inliers->indices[randidx];
		  cloud->points[cind].r = 0;
		  cloud->points[cind].g = 255;
		  cloud->points[cind].b = 0;
		  pcl::PointXYZRGB p =  cloud->points[cind];
		  planePoints.push_back(p);
		}

		return planePoints;
	}                      
	
}
