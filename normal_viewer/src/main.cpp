#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/gpu/features/features.hpp>
#include <ostream>

bool save_normals = true;

void get_normals_hist(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud,std::string filename){
	if(save_normals == true){
		std::ofstream outfiles;
		outfiles.open(filename,std::ios::out);
		for(auto &normals : *cloud){
			outfiles <<  normals.normal_x <<	" " << normals.normal_y << " " << normals.normal_z << std::endl;
		}
	outfiles.close();
	}
}


void XZplane(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &cloud){
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr res(new pcl::PointCloud<pcl::PointXYZRGBNormal> );
	
	pcl::ModelCoefficients::Ptr cff(new pcl::ModelCoefficients());
	cff->values.resize(4);
	cff->values[0] = 0.0;
	cff->values[1] = 1.0;
	cff->values[2] = 0.0;
	cff->values[3] = 0.0;

	pcl::ProjectInliers<pcl::PointXYZRGBNormal> proj;
	proj.setModelType(pcl::SACMODEL_PLANE);
	proj.setInputCloud(cloud);
	proj.setModelCoefficients(cff);
	proj.filter(*res);	
	
	get_normals_hist(res,"normals_plane.txt");
	pcl::io::savePLYFile("test.ply",*res);
}

pcl::PointCloud<pcl::Normal>::Ptr get_normals(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud){
	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal>::Ptr ne(new pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal>);
	ne->setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	ne->setSearchMethod(tree);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
	ne->setViewPoint(0.0f, 0.0f, 0.0f);
	ne->setRadiusSearch(0.3);
	ne->compute(*cloud_normals);
	return cloud_normals;
}

void remove_normal(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &norm_cloud){
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr output_expet(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

	float x,y,z;
	float a[3],b[3],c[3];
	float cos;
	float sab,ab;
	//XY_plane normal
	b[0] = 0.0f;
	b[1] = -1.0f;
	b[2] = 0.0f;
	
	//b[0] = 0.214;
	//b[1] = 0.100;
	//b[2] = -0.992;

	for(const auto &cloud : *norm_cloud){
		a[0] =	cloud.normal_x;
		a[1] =	cloud.normal_y;
		a[2] =	cloud.normal_z;
		
		ab =  a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
		sab = sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]) * sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
		cos = ab/sab;

		if (cos < 0.6){
			if(a[2] < -0.9 ){
				output->push_back(cloud);							
			}else{
				output_expet->push_back(cloud);							
			}

		}
	}
	
	XZplane(output);
	get_normals_hist(output,"normals_hist.txt");
	pcl::io::savePLYFile("removed.ply",*output);
	pcl::io::savePLYFile("removed_ex.ply",*output_expet);
}


int main (int argc, char* argv[]){
	std::string cloud_name = argv[1];
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB> );
	pcl::io::loadPLYFile(cloud_name, *cloud);

	auto normal = get_normals(cloud);
	
	pcl::visualization::PCLVisualizer viewer ("3D Viewer");
    viewer.setBackgroundColor(1.0,0.5,1.0);
    viewer.addPointCloud<pcl::PointXYZRGB> (cloud, "Input cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,"Input cloud");
    viewer.addPointCloudNormals<pcl::PointXYZRGB,pcl::Normal>(cloud,normal,1,0.05,"normals");
	
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr norm_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::concatenateFields(*cloud,*normal,*norm_cloud);
	std::cout <<norm_cloud->points[200] << std::endl;
	
	float x,y,z,a,b,c;
	x =	norm_cloud->points[2000].x;
	y =	norm_cloud->points[2000].y;
	z =	norm_cloud->points[2000].z;

	a =	norm_cloud->points[2000].normal_x;
	b =	norm_cloud->points[2000].normal_y;
	c =	norm_cloud->points[2000].normal_z;

	remove_normal(norm_cloud);

	pcl::ModelCoefficients coeffs;
	coeffs.values.push_back(a);
	coeffs.values.push_back(b);
	coeffs.values.push_back(c);
	coeffs.values.push_back(-a*x-b*y-c*z);

	//std::cout << norm_cloud->points[20000].x*a + norm_cloud->points[20000].y*b + norm_cloud->points[20000].z*c-(x*a+y*b+z*c) << std::endl;

	viewer.addPlane(coeffs,x,y,z,"Plane");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,1.0,1.0,0.0,"Plane");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,0.8 ,"Plane");
	viewer.addArrow(pcl::PointXYZ(x*a, y*b, z*c),cloud->points[2000],0.0, 0.0, 1.0,false);

	while(!viewer.wasStopped())
    {
        viewer.spinOnce (100);
    }
		
	return 0;
}

