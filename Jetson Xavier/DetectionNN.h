
#ifndef DETECTIONNN_H
#define DETECTIONNN_H

//default tkDNN libs
#include <iostream>
#include <signal.h>
#include <stdlib.h>    
#include <unistd.h>
#include <mutex>
#include "utils.h"
#include <cmath>

#include "tkdnn.h"

// #define OPENCV_CUDACONTRIB //if OPENCV has been compiled with CUDA and contrib.

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//custom libs

#include "ros/ros.h"
#include "vision_youbot/DetectedObj.h"
#include "vision_youbot/DetectedObjArr.h"

#include "vision_youbot/youbot_opencv_library.h"
#include "vision_youbot/youbot_utils_library.h"

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#ifdef OPENCV_CUDACONTRIB
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

#endif


namespace tk { namespace dnn {

class DetectionNN {

    protected:
        tk::dnn::NetworkRT *netRT = nullptr;
        dnnType *input_d;

        std::vector<cv::Size> originalSize;

        cv::Scalar colors[256];

        int nBatches = 1;

#ifdef OPENCV_CUDACONTRIB
        cv::cuda::GpuMat bgr[3];
        cv::cuda::GpuMat imagePreproc;
#else
        cv::Mat bgr[3];
        cv::Mat imagePreproc;
        dnnType *input;
#endif

        /**
         * This method preprocess the image, before feeding it to the NN.
         *
         * @param frame original frame to adapt for inference.
         * @param bi batch index
         */
        virtual void preprocess(cv::Mat &frame, const int bi=0) = 0;

        /**
         * This method postprocess the output of the NN to obtain the correct 
         * boundig boxes. 
         * 
         * @param bi batch index
         * @param mAP set to true only if all the probabilities for a bounding 
         *            box are needed, as in some cases for the mAP calculation
         */
        virtual void postprocess(const int bi=0,const bool mAP=false) = 0;

    public:
        int classes = 0;
        float confThreshold = 0.3; /*threshold on the confidence of the boxes*/

        std::vector<tk::dnn::box> detected; /*bounding boxes in output*/
        std::vector<std::vector<tk::dnn::box>> batchDetected; /*bounding boxes in output*/
        std::vector<double> stats; /*keeps track of inference times (ms)*/
        std::vector<std::string> classesNames;

        DetectionNN() {};
        ~DetectionNN(){};

        /**
         * Method used to initialize the class, allocate memory and compute 
         * needed data.
         * 
         * @param tensor_path path to the rt file of the NN.
         * @param n_classes number of classes for the given dataset.
         * @param n_batches maximum number of batches to use in inference
         * @return true if everything is correct, false otherwise.
         */
        virtual bool init(const std::string& tensor_path, const int n_classes=80, const int n_batches=1, const float conf_thresh=0.3) = 0;
        
        /**
         * This method performs the whole detection of the NN.
         * 
         * @param frames frames to run detection on.
         * @param cur_batches number of batches to use in inference
         * @param save_times if set to true, preprocess, inference and postprocess times 
         *        are saved on a csv file, otherwise not.
         * @param times pointer to the output stream where to write times
         * @param mAP set to true only if all the probabilities for a bounding 
         *            box are needed, as in some cases for the mAP calculation
         */
        void update(std::vector<cv::Mat>& frames, const int cur_batches=1, bool save_times=false, std::ofstream *times=nullptr, const bool mAP=false){
            if(save_times && times==nullptr)
                FatalError("save_times set to true, but no valid ofstream given");
            if(cur_batches > nBatches)
                FatalError("A batch size greater than nBatches cannot be used");

            originalSize.clear();
            if(TKDNN_VERBOSE) printCenteredTitle(" TENSORRT detection ", '=', 30); 
            {
                TKDNN_TSTART
                for(int bi=0; bi<cur_batches;++bi){
                    if(!frames[bi].data)
                        FatalError("No image data feed to detection");
                    originalSize.push_back(frames[bi].size());
                    preprocess(frames[bi], bi);    
                }
                TKDNN_TSTOP
                if(save_times) *times<<t_ns<<";";
            }

            //do inference
            tk::dnn::dataDim_t dim = netRT->input_dim;
            dim.n = cur_batches;
            {
                if(TKDNN_VERBOSE) dim.print();
                TKDNN_TSTART
                netRT->infer(dim, input_d);
                TKDNN_TSTOP
                if(TKDNN_VERBOSE) dim.print();
                stats.push_back(t_ns);
                if(save_times) *times<<t_ns<<";";
            }

            batchDetected.clear();
            {
                TKDNN_TSTART
                for(int bi=0; bi<cur_batches;++bi)
                    postprocess(bi, mAP);
                TKDNN_TSTOP
                if(save_times) *times<<t_ns<<"\n";
            }
        }      

        /**
         * Method to draw bounding boxes and labels on a frame.
         * 
         * @param frames original frame to draw bounding box on.
         */
         //objects_pub::yolo_box_array
        void draw(std::vector<cv::Mat>& frames) {
            tk::dnn::box b;
            int x0, w, x1, y0, h, y1;
            int objClass;
            std::string det_class;

            int baseline = 0;
            float font_scale = 0.5;
            int thickness = 2;   
            
            for(int bi=0; bi<frames.size(); ++bi){
				
                // draw dets
                if(batchDetected[bi].size()>0){
			int a;
			a = 2;
			for(int i=0; i<batchDetected[bi].size(); i++) { 
				b           = batchDetected[bi][i];
				x0   		= b.x;
				x1   		= b.x + b.w;
				y0   		= b.y;
				y1   		= b.y + b.h;
				det_class 	= classesNames[b.cl];

				// draw rectangle
				cv::rectangle(frames[bi], cv::Point(x0, y0), cv::Point(x1, y1), colors[b.cl], 2); 

				// draw label
				cv::Size text_size = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
				cv::rectangle(frames[bi], cv::Point(x0, y0), cv::Point((x0 + text_size.width - 2), (y0 - text_size.height - 2)), colors[b.cl], -1);                      
				cv::putText(frames[bi], det_class+" "+ std::to_string(b.prob), cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);

			}
		}
           
            }    
        }
        
        void drawYOLO(  float planeHeight, 
			geometry_msgs::TransformStamped& camToBase, 
			Eigen::Matrix3d inv_cam_intrinsics, 
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, 
			std::vector<cv::Mat>& frames, 
			ros::Publisher& pub,
			bool showImage) {
			//bbox parameters
            tk::dnn::box b;
            int x0, w, x1, y0, h, y1;
            int objClass;
            std::string det_class;

            int baseline = 0;
            float font_scale = 0.5;
            int thickness = 2;   
            
            for(int bi=0; bi<frames.size(); ++bi){
                if(batchDetected[bi].size()>0){
			vision_youbot::DetectedObjArr boxes_array;

			for(int i=0; i<batchDetected[bi].size(); i++) { 
				b           = batchDetected[bi][i];
				x0   		= b.x;
				x1   		= b.x + b.w;
				y0   		= b.y;
				y1   		= b.y + b.h;
				det_class 	= classesNames[b.cl];

				//crop bbox
				float xs = b.x-10;
				float ys = b.y-10;
				float wi = b.w+10;
				float hi = b.h+10;
				if(xs < 0){
					xs = 0;
				} else if (xs > frames[bi].cols){
					xs = frames[bi].cols;
				} if(ys<0){
					ys = 0;
				} else if (ys > frames[bi].rows){
					ys = frames[bi].rows;
				} if(xs+wi > frames[bi].cols){
					wi = frames[bi].cols-xs;
				} if(ys+hi > frames[bi].rows){
					hi= frames[bi].rows-ys;
				}

				cv::Rect crop_region(xs, ys, wi, hi);
				cv::Mat croppedImg = frames[bi](crop_region);

				//output obj
				vision_youbot::DetectedObj objectMsg;

				//get object height
				float objectHeight = youbot_utils::getObjectHeight(xs, ys, wi, hi, planeHeight, cloud, camToBase);
				//object size points
				std::vector<cv::Point> centroidAndAngle = youbot_opencv::getObjectPose(xs, ys, croppedImg, b.cl, showImage);

				if(!centroidAndAngle.empty()){
					float massCenterHeight = planeHeight+objectHeight/2;
					//get 3D size points in robot frame
					cv::Point2f center = youbot_utils::getPointOnPlane(centroidAndAngle.at(0), massCenterHeight, inv_cam_intrinsics, camToBase);
					cv::Point2f anglePoint = youbot_utils::getPointOnPlane(centroidAndAngle.at(1), massCenterHeight, inv_cam_intrinsics, camToBase);

					//get the state of an object (stays vertically/lies horizontally)
					if(b.cl == 1 || b.cl == 5 || b.cl == 7){ 					
						float objectWidth = youbot_utils::getObjectWidth(centroidAndAngle.at(0), centroidAndAngle.at(2), massCenterHeight, inv_cam_intrinsics, camToBase);
						youbot_utils::getObjectState(objectHeight, objectWidth, b.cl, objectMsg);
					} else{
						youbot_utils::getObjectState(objectHeight, 0, b.cl, objectMsg);
					}

					//calculate object orientation
					float angle  = std::atan2(anglePoint.x-center.x, anglePoint.y-center.y);
					angle = angle + 1.571;
					if(angle < -1.571){
						angle = angle + 3.141;
					} else if (angle > 1.571){
						angle = angle - 3.141;
					}

					//fill output array
					objectMsg.x = center.x;
					objectMsg.y = center.y;
					objectMsg.angle = angle;
					objectMsg.h = objectHeight;
					objectMsg.plane_h = planeHeight;

					std::cout << objectMsg.name << " " << std::endl;

					boxes_array.objects.push_back(objectMsg);
				}

				if(showImage){
					cv::rectangle(frames[bi], cv::Point(x0, y0), cv::Point(x1, y1), colors[b.cl], 2); 
					cv::Size text_size = getTextSize(det_class, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
					cv::rectangle(frames[bi], cv::Point(x0, y0), cv::Point((x0 + text_size.width - 2), (y0 - text_size.height - 2)), colors[b.cl], -1);                      
					cv::putText(frames[bi], det_class+" "+ std::to_string(b.prob), cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
				}


			}
			//publish detected objects
			pub.publish(boxes_array);
		}
           
            }    
        }

};

}}

#endif /* DETECTIONNN_H*/
