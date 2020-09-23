//
// Created by htk on 19/06/07.
//
#include <pcl/point_types.h>
#include <iostream>
#include <string>
#include <tuple>
#include <chrono>
#include <omp.h>
#include <Eigen/Core>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/warp_point_rigid_3d.h>
#include <pcl/gpu/features/features.hpp>
#include <pcl/gpu/containers/device_array.hpp>
void print4x4Matrix (const Eigen::Matrix4d & matrix)
{
    printf ("Rotation matrix :\n");
    printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
    printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
    printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
    printf ("Translation vector :\n");
    printf ("t = < %6.3f  %6.3f  %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
}

void Rotate2degree (const Eigen::Matrix4d & R)
{
    double Rad2Degree = 180/3.14;
    //double Rad2Degree = 1;
    double angle_x,angle_y, angle_z;
    std::cout <<"deg" << std::endl;
    double PI=180;
    double threshold = 0.0001;
    angle_y = asin(-R(2,0));
    angle_z = atan2(R(1,0), R(0,0));
    angle_x = atan2(R(2,1), R(2,2));

    std::cout <<"x= "<<angle_x*Rad2Degree << std::endl;
    std::cout <<"y= "<<angle_y*Rad2Degree << std::endl;
    std::cout <<"z= "<<angle_z*Rad2Degree << std::endl;

}

auto read_point_cloud(const std::string& filename)
{
    const pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
    const int retval = pcl::io::loadPLYFile(filename, *pointcloud);
    if (retval == -1 || pointcloud->size() <= 0)
    {
        PCL_ERROR("File load error.");
        exit(-1);
    }

    return pointcloud;
}

auto draw_registration_result(
        const std::string& window_name,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& scene1,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& scene2,
        const Eigen::Matrix4f& trans)
{

    const pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(window_name));
    viewer->setBackgroundColor(0.0, 0.0, 0.0);

    const auto scene1_temp = scene1->makeShared();
    const auto scene2_temp = scene2->makeShared();

    pcl::transformPointCloud(*scene1_temp, *scene1_temp, trans);
    pcl::io::savePLYFileASCII ("turned_model.ply", *scene1_temp);
    const pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>::Ptr color1(new pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(scene1_temp, 1.0*255, 0.706 * 255, 0.0 * 255));
    const pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>::Ptr color2(new pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(scene2_temp, 0.0 * 255, 0.651 * 255, 0.929 * 255));
    viewer->addPointCloud<pcl::PointXYZ>(scene1_temp, *color1, "scene1");
    viewer->addPointCloud<pcl::PointXYZ>(scene2_temp, *color2, "scene2");

    return viewer;
}

auto preprocess_point_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& pointcloud, const float voxel_size)
{
    const pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);

    boost::shared_ptr<pcl::gpu::DeviceArray<pcl::PointXYZ>> key(new pcl::gpu::DeviceArray<pcl::PointXYZ>);
    boost::shared_ptr<pcl::gpu::DeviceArray<pcl::PointXYZ>> g_pointcloud(new pcl::gpu::DeviceArray<pcl::PointXYZ>);

    const boost::shared_ptr<pcl::VoxelGrid<pcl::PointXYZ>> sor(new pcl::VoxelGrid<pcl::PointXYZ>);
    sor->setInputCloud(pointcloud);
    sor->setLeafSize(voxel_size, voxel_size, voxel_size);
    sor->filter(*keypoints);


    const float radius_normal = voxel_size * 2.0;
    const auto view_point = pcl::PointXYZ(0.0, 10.0, 10.0);

    //use cpu

    const pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    const pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal>::Ptr ne(new pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal>);
    const pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    ne->setInputCloud(pointcloud);
    ne->setRadiusSearch(radius_normal);
    ne->setSearchMethod(kdtree);
    ne->setViewPoint(view_point.x, view_point.y, view_point.z);
    ne->compute(*normals);
    
    //use gpu

    key->upload(keypoints->points);
    g_pointcloud->upload(pointcloud->points);

    pcl::gpu::NormalEstimation gne;
    pcl::gpu::DeviceArray<pcl::PointXYZ> g_normals;
    gne.setInputCloud(*g_pointcloud);
    gne.setRadiusSearch(radius_normal,10);
    gne.setViewPoint(view_point.x,view_point.y,view_point.z);
    gne.compute(g_normals);

    const float radius_feature = voxel_size * 5.0;


    //use cpu
    const pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh(new pcl::PointCloud<pcl::FPFHSignature33>);
    /*
    const pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>::Ptr fpfhe(new pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>);
    fpfhe->setInputCloud(keypoints);
    fpfhe->setSearchSurface(pointcloud);
    fpfhe->setInputNormals(normals);
    fpfhe->setSearchMethod(kdtree);
    fpfhe->setRadiusSearch(radius_feature);
    fpfhe->compute(*fpfh);
    */
    //use gpu
    boost::shared_ptr<pcl::gpu::DeviceArray2D<pcl::FPFHSignature33>> fpfhg(new pcl::gpu::DeviceArray2D<pcl::FPFHSignature33>);
    pcl::gpu::FPFHEstimation fpfheg;
    fpfheg.setInputCloud(*key);

    fpfheg.setSearchSurface(*g_pointcloud);
    fpfheg.setInputNormals(g_normals);
    fpfheg.setRadiusSearch(radius_feature,10);
    fpfheg.compute(*fpfhg);
    int cols;
    fpfhg->download(fpfh->points,cols);
    //std::cout << fpfh->begin() << std::endl;

    return std::make_pair(keypoints, fpfh);
}


// RANSAC による Global Registration
auto execute_global_registration(
        const std::pair<const pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::FPFHSignature33>::Ptr>& scene1,
        const std::pair<const pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::FPFHSignature33>::Ptr>& scene2,
        const float voxel_size)
{
    const auto& kp1 = scene1.first;
    const auto& kp2 = scene2.first;
    const auto& fpfh1 = scene1.second;
    const auto& fpfh2 = scene2.second;
    Eigen::VectorXf p(3);
    p[0]=0;
    p[1]=0;
    p[2] = 1;


    const float distance_threshold = voxel_size * 2.5;
    //boost::shared_ptr<pcl::registration::WarpPointRigid3D<pcl::PointXYZ, pcl::PointXYZ> >  warp_fcn(new pcl::registration::WarpPointRigid3D<pcl::PointXYZ, pcl::PointXYZ>);
    //warp_fcn->setParam(p);
    //boost::shared_ptr<pcl::registration::TransformationEstimationLM<pcl::PointXYZ, pcl::PointXYZ> > te (new pcl::registration::TransformationEstimationLM<pcl::PointXYZ, pcl::PointXYZ>);
    //te->setWarpFunction (warp_fcn);
    const pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
    const pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33>::Ptr align(new pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33>);
    align->setInputSource(kp1);
    align->setSourceFeatures(fpfh1);
    align->setInputTarget(kp2);
    align->setTargetFeatures(fpfh2);
    //align->setTransformationEstimation(te);
    //align->setMaximumIterations(300000);
    align->setMaximumIterations(100000);
    align->setNumberOfSamples(4);
    align->setCorrespondenceRandomness(2);
    align->setSimilarityThreshold(0.9f);
    align->setMaxCorrespondenceDistance(distance_threshold);
    //align->setInlierFraction(0.25f);
    align->setInlierFraction(0.25f);
    align->align(*output);

     return align->getFinalTransformation();
}

// ICP によるRegistration
Eigen::Matrix4f refine_registration(
        const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& scene1,
        const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& scene2,
		const Eigen::Matrix4f& trans,
        const float voxel_size)
{

    //boost::shared_ptr<pcl::registration::WarpPointRigid3D<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> >  warp_fcn(new pcl::registration::WarpPointRigid3D<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>);
    //boost::shared_ptr<pcl::registration::TransformationEstimationLM<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> > te (new pcl::registration::TransformationEstimationLM<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>);
    //te->setWarpFunction (warp_fcn);
    const auto scene1_temp = scene1->makeShared();
    pcl::transformPointCloud(*scene1_temp, *scene1_temp, trans);
    Eigen::Matrix4f ts= Eigen::Matrix4f::Identity();
    const float distance_threshold = voxel_size * 0.5;
    pcl::registration::TransformationEstimationSymmetricPointToPlaneLLS<pcl::PointXYZRGBNormal,pcl::PointXYZRGBNormal>::Ptr tesp(new pcl::registration::TransformationEstimationSymmetricPointToPlaneLLS<pcl::PointXYZRGBNormal,pcl::PointXYZRGBNormal>);
    tesp->estimateRigidTransformation(*scene1_temp,*scene2,ts);

    return ts;
}

pcl::PointCloud<pcl::Normal>::Ptr get_normals( const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud){
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal>::Ptr ne(new pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal>);
    ne->setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne->setSearchMethod(tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    ne->setViewPoint(0.0f, 0.0f, 0.0f);
    ne->setRadiusSearch(1.0);
    ne->compute(*cloud_normals);
    return cloud_normals;
}       

int main(int argc, char* argv[]) {
    std::string modelname = argv[1];
    std::string cloudname = argv[2];
	float yaw;
	float pos[3];

	std::ifstream readfile;
        
    readfile.open("pose.txt");
    if(!readfile){
        std::cout << "erro could not read file" << std::endl;
        exit(1);
    }   
    readfile >> yaw >> pos[0] >> pos[1];
    readfile.close();



	std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();
	
	yaw = yaw*3.14/180;
    // 読み込み
    const auto scene1 = read_point_cloud(modelname);
    const auto scene2 = read_point_cloud(cloudname);

    double deg = -90;

    Eigen::Matrix4f tmatrix;
    tmatrix <<
            1.0, 0.0, 0.0, 0.0,
            0.0, cos(deg), -sin(deg), 0.0,
            0.0, sin(deg), cos(deg), 0.0,
            0.0, 0.0, 0.0, 1;
	Eigen::Matrix4f tx;
    tx <<
            cos(yaw), 0.0, sin(yaw), pos[0],
            0.0, 1.0, 0.0, 0.0,
            -sin(yaw), 0.0, cos(yaw), pos[1],
            0.0, 0.0, 0.0, 1;


    pcl::transformPointCloud(*scene1, *scene1, tx);
    pcl::transformPointCloud(*scene1, *scene1, tmatrix);
   // pcl::transformPointCloud(*scene2, *scene2, transform_matrix);
    pcl::transformPointCloud(*scene2, *scene2, tmatrix);
    auto normals_1 = get_normals(scene1);
    auto normals_2 = get_normals(scene2);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr norm_cloud1(new pcl::PointCloud<pcl::PointXYZRGBNormal>);                                                       
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr norm_cloud2(new pcl::PointCloud<pcl::PointXYZRGBNormal>);   
    pcl::concatenateFields(*scene1,*normals_1,*norm_cloud1);
    pcl::concatenateFields(*scene2,*normals_2,*norm_cloud2);
    // 位置合わせ前の点群の表示
    const auto viewer1 = draw_registration_result("Initial", scene1, scene2, Eigen::Matrix4f::Identity());

    const float voxel_size = 0.2;

    //RANSAC による Global Registration
    const auto scene1_kp_fpfh = preprocess_point_cloud(scene1, voxel_size);
    const auto scene_kp_fpfh = preprocess_point_cloud(scene2, voxel_size);

    const auto result_ransac = execute_global_registration(scene1_kp_fpfh, scene_kp_fpfh, voxel_size);
    const auto viewer2 = draw_registration_result("Global", scene1, scene2, result_ransac);


    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d transformation_matrix2 = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d transformation_matrix3 = Eigen::Matrix4d::Identity();

    transformation_matrix = (result_ransac).cast<double>();
    print4x4Matrix(transformation_matrix);
    Rotate2degree(transformation_matrix);

    const auto result = refine_registration(norm_cloud1, norm_cloud2, result_ransac, voxel_size);
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << "finish_time " << elapsed << "ms"<< std::endl;
    const auto viewer3 = draw_registration_result("Refined", scene1, scene2, result*result_ransac);

    transformation_matrix2 = result.cast<double>();
    print4x4Matrix(transformation_matrix2*transformation_matrix );
    Rotate2degree(transformation_matrix2*transformation_matrix );

    while (!viewer1->wasStopped() && !viewer2->wasStopped()) {
        viewer1->spinOnce();
        viewer2->spinOnce();
        viewer3->spinOnce();
    }
}
