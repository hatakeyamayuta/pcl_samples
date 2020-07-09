#include <iostream>
#include <pcl/filters/passthrough.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <chrono>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>

inline void pass_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{   
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(2.9,9.5);
    pass.filter(*cloud);
}

inline void voxel_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{

    pcl::VoxelGrid<pcl::PointXYZRGB>  voxelGrid;
    voxelGrid.setInputCloud(cloud);
    voxelGrid.setLeafSize(0.1f, 0.1f, 0.1f);
    voxelGrid.filter(*cloud);
}

inline void outlier_filter(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (50);
    sor.setStddevMulThresh (2.0);
    sor.filter (*cloud);
}


int main(int argc,char* argv[]) {
    pcl::PLYReader reader;
    pcl::PLYWriter writer;
    
	std::string modelname1 = argv[1];
    std::string modelname2 = argv[2];
	
	int flag = 0;
	float deg = 90*M_PI/180;
    

	Eigen::Matrix4f transform_matrix;
    transform_matrix <<
          1.0,       0.0,        0.0,     0.0,
          0.0,   cos(deg),   -sin(deg),   0.0,
          0.0,   sin(deg),    cos(deg),   0.0,
          0.0,       0.0,        0.0,     1.0;  

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr model(new pcl::PointCloud<pcl::PointXYZRGB>);

    reader.read(modelname1, *cloud);
    reader.read(modelname2, *model);

	pcl::transformPointCloud(*cloud, *cloud, transform_matrix);
	*cloud +=*model;
    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
 	
	if(flag == 1){
		pass_filter(cloud);
		outlier_filter(cloud);
	    voxel_filter(cloud);
	}
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << "finish_time " << elapsed << "ms"<< std::endl;
	
    writer.write("output.ply",*cloud);



    return 0;
}
