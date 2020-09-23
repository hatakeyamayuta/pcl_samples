#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include<iostream>
#include<pcl/filters/voxel_grid_occlusion_estimation.h>
#include <pcl/common/transforms.h>
int main(int argc, char * argv[]) { 
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>); 

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_occluded(new pcl::PointCloud<pcl::PointXYZ>); 

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_visible(new pcl::PointCloud<pcl::PointXYZ>); 

	std::string name = argv[1];
	pcl::io::loadPCDFile(name,*cloud_in); 
	
	Eigen::Matrix4d trasns;
	trasns << 1.0 , 0.0 , 0.0, 0.0,
			  0.0 , 1.0 , 0.0, 5.5,
			  0.0 , 0.0 , 1.0, -0.7,
			  0.0 , 0.0 , 0.0, 1.0;
	pcl::transformPointCloud<pcl::PointXYZ>(*cloud_in,*cloud_in,trasns);
	Eigen::Quaternionf quat(1,0,0,0); 
	cloud_in->sensor_origin_ = Eigen::Vector4f(0,0,0,0); 
	cloud_in->sensor_orientation_= quat; 
	pcl::VoxelGridOcclusionEstimation<pcl::PointXYZ> voxelFilter; 
	voxelFilter.setInputCloud (cloud_in); 
	float leaf_size=0.02; 
	voxelFilter.setLeafSize (leaf_size, leaf_size, leaf_size); 
	voxelFilter.initializeVoxelGrid(); 

	for (size_t i=0;i<cloud_in->size();i++){ 
		pcl::PointXYZ pt=cloud_in->points[i]; 

		Eigen::Vector3i grid_cordinates=voxelFilter.getGridCoordinates (pt.x, pt.y, pt.z); 

		int grid_state; 

		int ret=voxelFilter.occlusionEstimation(grid_state, grid_cordinates); 
		if (grid_state==1){ 
		 cloud_occluded->push_back(cloud_in->points[i]); 
		}else{ 
			cloud_visible->push_back(cloud_in->points[i]); 
		} 

	} 
	pcl::io::savePCDFile(argv[2],*cloud_occluded); 
	pcl::io::savePCDFile(argv[2],*cloud_visible); 

	return 0; 
} 
