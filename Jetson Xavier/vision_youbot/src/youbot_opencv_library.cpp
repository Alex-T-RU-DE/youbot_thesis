#include "vision_youbot/youbot_opencv_library.h"
namespace youbot_opencv
{
	std::vector<std::vector<cv::Point> > contoursAndCanny(cv::Mat &croppedImg){
		//grayscale
		cv::Mat gray;
		cv::cvtColor(croppedImg, gray, cv::COLOR_BGR2GRAY);
		//blur
		cv::medianBlur(gray,gray, 5);
	    //canny
		cv::Canny(gray,gray, 20,200);
	    //dilation
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
		cv::dilate(gray, gray, kernel);
		//get contours
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(gray, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	  
		return contours;
		
	}

	std::vector<cv::Point> findTheBiggestContour(std::vector<std::vector<cv::Point> > &contours){
		double maxArea = 0;
		int maxAreaContourId = -1;
		for (int j = 0; j < contours.size(); j++) {
			double newArea = cv::contourArea(contours.at(j));
			if (newArea > maxArea) {
				maxArea = newArea;
				maxAreaContourId = j;
			} 
		} 
		
		std::vector<cv::Point> c;
		
		if(maxAreaContourId == -1){
			return c;
		} 
		
		c = contours.at(maxAreaContourId);
	
		return c;
	}

	void drawOneContour(cv::Mat& img, std::vector<cv::Point>& c){
		std::vector<std::vector<cv::Point> > draw; 
		draw.push_back(c);
		cv::Scalar color(0,255,0);
		cv::drawContours(img, draw, -1, color);
	}

	std::vector<cv::Point> findCorners(std::vector<cv::Point>& c){
		double peri = cv::arcLength(c, true);
		std::vector<cv::Point> approx;
		cv::approxPolyDP(c, approx, 0.04 * peri, true);
		std::cout << "Corners count: " << approx.size() << std::endl;
		
		return approx;
	}

	std::vector<cv::Point> findConvexHull(std::vector<cv::Point>& c){
		
		std::vector<cv::Point> hull;
		
		cv::convexHull(c, hull);
		
		return hull;
	}

	cv::PCA pca (std::vector<cv::Point> &c){
		
		int sz = static_cast<int>(c.size());
		cv::Mat data_pts = cv::Mat(sz, 2, CV_64F);
		for (int i = 0; i < data_pts.rows; i++)
		{
			data_pts.at<double>(i, 0) = c[i].x;
			data_pts.at<double>(i, 1) = c[i].y;
		}
		
		cv::PCA pca_analysis(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
		
		return pca_analysis;
	}

	std::vector<cv::Point> PCAPose(cv::Mat &croppedImg, std::vector<cv::Point> &c)
	{
		//perform PCA
		cv::PCA pca_analysis = pca(c);
		
		//get center
		cv::Point cntr = cv::Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
								   static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
								   
		std::vector<cv::Point2d> eigen_vecs(2);
		std::vector<double> eigen_val(2);
		
		//get eigenvectors and eigenvalues
		for (int i = 0; i < 2; i++)
		{
			eigen_vecs[i] = cv::Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
										pca_analysis.eigenvectors.at<double>(i, 1));
			eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
		}
		// Draw the principal components
		
		cv::Point anglePoint = cntr + 0.01 *  cv::Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
		
		cv::circle(croppedImg, cntr, 3, cv::Scalar(0, 0, 255), 2);
		cv::line(croppedImg, cntr, anglePoint, cv::Scalar(255, 0, 0), 1);
		cv::circle(croppedImg, anglePoint, 3, cv::Scalar(0, 0, 255), 2);
		
		std::vector<cv::Point> centroidAngle;
		centroidAngle.push_back(cntr);
		centroidAngle.push_back(anglePoint);
		
		return centroidAngle;
	}



	std::vector<cv::Point> rectPose (cv::Mat &croppedImg, std::vector<cv::Point> &c)
	{
		//find a minimal rotated rectangle
		cv::RotatedRect rect = cv::minAreaRect(c);
		
		cv::Point2f vertices[4];
		rect.points(vertices);
		for (int i = 0; i < 4; i++)
		   line(croppedImg, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0), 2);
		
		//calculate rectangle`s edges
		cv::Point2f edge1 = cv::Vec2f(vertices[1].x, vertices[1].y) - cv::Vec2f(vertices[0].x, vertices[0].y);
		cv::Point2f edge2 = cv::Vec2f(vertices[2].x, vertices[2].y) - cv::Vec2f(vertices[1].x, vertices[1].y);
		
		cv::Point2f cntr = rect.center;
		cv::Point2f anglePoint;
		cv::Point2f sizePoint;
		
		//define anglePoint and sizePoints depending on edges` length
		if(cv::norm(edge2) > cv::norm(edge1))
		{
			anglePoint = cv::Vec2f(cntr.x, cntr.y)+cv::Vec2f(edge2.x/2, edge2.y/2);
			sizePoint = cv::Vec2f(cntr.x, cntr.y)+cv::Vec2f(edge1.x/2, edge1.y/2);
		}
		else
		{
			anglePoint = cv::Vec2f(cntr.x, cntr.y)+cv::Vec2f(edge1.x/2, edge1.y/2);
			sizePoint = cv::Vec2f(cntr.x, cntr.y)+cv::Vec2f(edge2.x/2, edge2.y/2);
		}
		
		//draw
		cv::circle(croppedImg, cntr, 3, cv::Scalar(0, 0, 255), 2);
		cv::line(croppedImg, cntr, anglePoint, cv::Scalar(255, 0, 0), 1);
		cv::circle(croppedImg, anglePoint, 3, cv::Scalar(0, 0, 255), 2);
		cv::circle(croppedImg, sizePoint, 3, cv::Scalar(255, 0, 0), 2);
		
		std::vector<cv::Point> centroidAngle;
		centroidAngle.push_back((cv::Point)cntr);
		centroidAngle.push_back((cv::Point)anglePoint);
		centroidAngle.push_back((cv::Point)sizePoint);
		
		return centroidAngle;
		
	}


							  
	std::vector<cv::Point> getObjectPose(float xs, float ys, cv::Mat croppedImg,  int classname, bool show_image)
	{
		//preprocess image
		std::vector<std::vector<cv::Point> > cntrs = contoursAndCanny(croppedImg);
		std::vector<cv::Point> c = findTheBiggestContour(cntrs);
		
		std::vector<cv::Point> centroidAngle;
		
		if(c.empty())
		{
			return centroidAngle;
		} 
		else
		{
			drawOneContour(croppedImg, c);
			std::vector<cv::Point> hull = findConvexHull(c);
			drawOneContour(croppedImg, hull);
			
			//choose strategy depending on object`s class
			switch(classname)
			{
				case 0:
					centroidAngle = PCAPose(croppedImg, hull);
					break;
				case 1:
					centroidAngle = rectPose(croppedImg, hull);
					break;
				case 2:
					centroidAngle = PCAPose(croppedImg, hull);
					break;
				case 3:
					centroidAngle = rectPose(croppedImg, hull);
					break;
				case 4:
					centroidAngle = rectPose(croppedImg, hull);
					break;
				case 5:
					centroidAngle = rectPose(croppedImg, hull);
					break;
				case 6:
					centroidAngle = rectPose(croppedImg, hull);
					break;
				case 7:
					centroidAngle = rectPose(croppedImg, hull);
					break;
				case 8:
					centroidAngle = rectPose(croppedImg, hull);
					break;
				case 9:
					centroidAngle = rectPose(croppedImg, hull);
					break;
				case 10:
					centroidAngle = rectPose(croppedImg, hull);
					break;
				case 11:
					centroidAngle = rectPose(croppedImg, hull);
					break;
			}
		}
		if(show_image)
		{
			cv::imshow("crop", croppedImg);
		}
		
		//add an offset from an original image
		for(cv::Point& p : centroidAngle)
		{
			p.x += (int)xs;
			p.y += (int)ys;
		}
		
		return centroidAngle;
	}
}
