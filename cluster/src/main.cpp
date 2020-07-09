#include <pcl/io/ply_io.h>
#include <pcl/segmentation/extract_clusters.h>
#include <iostream>
#include <pcl/common/centroid.h>
#include <pcl/surface/mls.h>
#include <pcl/kdtree/kdtree_flann.h>

pcl::PointCloud<pcl::PointNormal> smooth(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud)
{
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);//Kdtreeの作成

    pcl::PointCloud<pcl::PointNormal> mls_points;//出力する点群の格納場所を作成

    pcl::MovingLeastSquares<pcl::PointXYZRGBNormal, pcl::PointNormal> mls;

    mls.setComputeNormals (true);//法線の計算を行うかどうか

    // 各パラメーターの設定
    mls.setInputCloud (cloud);
    mls.setPolynomialFit (true);
    mls.setSearchMethod (tree);
    mls.setSearchRadius (0.4);

    mls.process (mls_points);//再構築

    return mls_points;//出力
}


float get_center_point(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud){
	Eigen::Vector4d centors;
	pcl::compute3DCentroid(*cloud,centors);
	std::cout << centors[2] << std::endl;
	return centors[2];
}

int main(int argc, char** argv){
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

	if (pcl::io::loadPLYFile<pcl::PointXYZRGBNormal>(argv[1], *cloud) != 0){
		return -1;
	}

	pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
	kdtree->setInputCloud(cloud);

	pcl::EuclideanClusterExtraction<pcl::PointXYZRGBNormal> clustering;
	clustering.setClusterTolerance(0.4);
	clustering.setMinClusterSize(100);
	clustering.setMaxClusterSize(350000);
	clustering.setSearchMethod(kdtree);
	clustering.setInputCloud(cloud);
	std::vector<pcl::PointIndices> clusters;
	clustering.extract(clusters);

	int currentClusterNum = 1;
	float close = 10.0;
	float z = 0;

	for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i){
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGBNormal>);

		for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
			cluster->points.push_back(cloud->points[*point]);
		cluster->width = cluster->points.size();
		cluster->height = 1;
		cluster->is_dense = true;

		if (cluster->points.size() <= 0)
			break;
		std::string fileName = "cluster"+ std::to_string(currentClusterNum) +".ply";
		z = get_center_point(cluster);
		pcl::io::savePLYFileASCII(fileName, *cluster);
		std::cout << "Cluster " << currentClusterNum << " has " << cluster->points.size() << " points." << std::endl;
		currentClusterNum++;
	}
}
