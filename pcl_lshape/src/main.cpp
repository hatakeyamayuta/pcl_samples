#include <iostream>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/common/transforms.h>

constexpr float operator""_deg(long double deg){
    return deg *(M_PI/180.0);
}

class LshapeFitting{
    float score;
    pcl::PointXYZ minPt, maxPt;
    public:
        float calc_area_criterion(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);        

};

float LshapeFitting::calc_area_criterion(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud){
    pcl::getMinMax3D(*cloud, minPt, maxPt);
    //std::cout << maxPt.x << std::endl;
    score = -(maxPt.x - minPt.x) * (maxPt.z - minPt.z);
    std::cout << score << std::endl;
    return score;
}

class RectangleData{
    public:
    float a[4];
    float b[4];
    float c[4];
    float c_x[5];
    float c_y[5];
    void calc_cross(){
        for(int i=0; i <3; ++i){

            c_x[i] = (b[i] * -c[i+1] - b[i+1] * -c[i]) / (a[i] * b[i+1] - a[i+1] * b[i]);
            c_y[i] = (a[i+1] * -c[i] - a[i] * -c[i+1]) / (a[i] * b[i+1] - a[i+1] * b[i]);
        }
        //c_x[0] = (b[0] * -c[1] - b[1] * -c[0]) / (a[0] * b[1] - a[1] * b[0]);
        //c_y[0] = (a[1] * -c[0] - a[0] * -c[1]) / (a[0] * b[1] - a[1] * b[0]);
        c_x[3] = (b[3] * -c[0] - b[0] * -c[3]) / (a[3] * b[0] - a[0] * b[3]);
        c_y[3] = (a[0] * -c[3] - a[3] * -c[0]) / (a[3] * b[0] - a[0] * b[3]);
        std::cout <<"cccccccccccccccccccccccccccc" <<std::endl;
        std::cout <<c_x[0] <<std::endl;
        std::cout <<c_x[1] <<std::endl;
        std::cout <<c_x[2] <<std::endl;
        std::cout <<c_x[3] <<std::endl;
        std::cout <<c_y[0] <<std::endl;
        std::cout <<c_y[1] <<std::endl;
        std::cout <<c_y[2] <<std::endl;
        std::cout <<c_y[3] <<std::endl;
    } 

};
int main(int argv, char* argc[]){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr rt_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PCDReader reader;
    pcl::PCDWriter writer;
    float res[2] = {-10.0,0};
    float ddeg = 1.0 ;
    float rad = 0;
    float score = 0;
    
    std::string filename = argc[1];
    reader.read(filename, *cloud);
    
    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now();
    
    pcl::ModelCoefficients::Ptr mocff(new pcl::ModelCoefficients());
    mocff->values.resize(4);
    mocff->values[0] = 0.0;
    mocff->values[1] = 1.0;
    mocff->values[2] = 0.0;
    mocff->values[2] = 0.0;
    
    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(cloud);
    proj.setModelCoefficients(mocff);
    proj.filter(*cloud);
    
    Eigen::Matrix4d trans;
    LshapeFitting lshape = LshapeFitting();
    RectangleData rec = RectangleData();
    
    for (int i = 0; rad <= M_PI/2; i++){
        rad += ddeg/180*M_PI;
        trans << cos(rad),   0.0, sin(rad), 0.0,
                      0.0,   1.0,      0.0, 0.0,
                -sin(rad),   0.0, cos(rad), 0.0,
                      0.0,   0.0,      0.0, 1.0;

        pcl::transformPointCloud(*cloud,*rt_cloud,trans);
        score = lshape.calc_area_criterion(rt_cloud);
        if (res[0] < score){
            res[0] = score; 
            res[1] = rad; 
            std::cout << rad*180/M_PI << std::endl;
        }
            std::cout << rad*180/M_PI << std::endl;
    }
    
    pcl::PointXYZ minPt, maxPt;
    trans << cos(res[1]),   0.0, sin(res[1]), 0.0,
                      0.0,   1.0,      0.0, 0.0,
                -sin(res[1]),   0.0, cos(res[1]), 0.0,
                      0.0,   0.0,      0.0, 1.0;
    pcl::transformPointCloud(*cloud,*cloud,trans);
    pcl::getMinMax3D(*cloud, minPt, maxPt);
    rec.a[0] = cos(res[1]);
    rec.b[0] = sin(res[1]);
    rec.c[0] = minPt.x;

    rec.a[1] = -sin(res[1]);
    rec.b[1] = cos(res[1]);
    rec.c[1] = minPt.z;
    
    rec.a[2] = cos(res[1]);
    rec.b[2] = sin(res[1]);
    rec.c[2] = maxPt.x;

    rec.a[3] = -sin(res[1]);
    rec.b[3] = cos(res[1]);
    rec.c[3] = maxPt.z;

    rec.calc_cross();
    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << res[1]*180/M_PI << std::endl;
    std::cout << 1000.0/elapsed<< std::endl;
    return 0;
}