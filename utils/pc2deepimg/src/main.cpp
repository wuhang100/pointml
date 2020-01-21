#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/common/float_image_utils.h>
#include <pcl/console/parse.h>

using namespace std;

int main(int argc, char** argv){
	string file_path_view;
	string file_path_full;
	string save_view;
	string save_full;

	if (argc < 2){
		PCL_INFO ("Input pcd directory for %s \n", argv[0]);
		PCL_INFO ("  -cloud_full   <.pcd>   Full cloud path\n"
				  "  -cloud_view   <.pcd>   View cloud path\n "
				  "  -pos          x,y,z    Camera position\n"
				  "  -focal        x,y,z    Camera focus point\n"
				  "  -view         x,y,z    Camera up view\n"
				  "  -window       x,y      Window size\n"
				  "  -imgshow      <0,1>    Show image\n"
				  "  -save_full    <.pcd>   Full image save path\n"
				  "  -save_view    <.pcd>   View image save path\n");
		return (-1);
	}
	pcl::console::parse_argument(argc, argv, "-cloud_full", file_path_full);
	pcl::console::parse_argument(argc, argv, "-cloud_view", file_path_view);
	pcl::console::parse_argument(argc, argv, "-save_full", save_full);
	pcl::console::parse_argument(argc, argv, "-save_view", save_view);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cam (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cam_c (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile<pcl::PointXYZ> (file_path_view, *cloud);
	pcl::visualization::PCLVisualizer vis;
	Eigen::Matrix4d	view_mat;
	pcl::visualization::Camera cam;

	double posx = 10, posy = 1600, posz = 1000;
	pcl::console::parse_3x_arguments (argc, argv, "-pos", posx, posy, posz);
	double campos[3]={posx,posy,posz};

	double camfocalx = -200, camfocaly = 500, camfocalz = -225;
	pcl::console::parse_3x_arguments (argc, argv, "-focal", camfocalx, camfocaly, camfocalz);
	double camfocal[3]={camfocalx,camfocaly,camfocalz};

	double camviewx = 0, camviewy = 1, camviewz = 0;
	pcl::console::parse_3x_arguments (argc, argv, "-view", camviewx, camviewy, camviewz);
	double camview[3]={camviewx,camviewy,camviewz};

	double camwindowx = 256, camwindowy = 256;
	pcl::console::parse_2x_arguments(argc, argv, "-window", camwindowx, camwindowy);
	double camwindow[3]={camwindowx,camwindowy};

	memcpy(cam.pos,campos,sizeof(campos));
	memcpy(cam.focal,camfocal,sizeof(camfocal));
	memcpy(cam.view,camview,sizeof(camview));
	memcpy(cam.window_size,camwindow,sizeof(camwindow));
	cam.computeViewMatrix(view_mat);
	//cout<<"["<<view_mat(0,0)<<","<<view_mat(0,1)<<","<<view_mat(0,2)<<","<<view_mat(0,3)<<"]"<<endl;
	//cout<<"["<<view_mat(1,0)<<","<<view_mat(1,1)<<","<<view_mat(1,2)<<","<<view_mat(1,3)<<"]"<<endl;
	//cout<<"["<<view_mat(2,0)<<","<<view_mat(2,1)<<","<<view_mat(2,2)<<","<<view_mat(2,3)<<"]"<<endl;

	pcl::transformPointCloud (*cloud, *cloud_cam, view_mat);
	Eigen::Vector4f centroid;  //质心
	pcl::compute3DCentroid(*cloud_cam,centroid);
	//cout << centroid;
	Eigen::Matrix4f transform_2 = Eigen::Matrix4f::Identity();
	transform_2 (0,3) = -centroid(0);
	transform_2 (1,3) = -centroid(1);
	transform_2 (2,3) = -centroid(2);
	//cout << transform_2;
	pcl::transformPointCloud (*cloud_cam, *cloud_cam_c, transform_2);

	vis.setBackgroundColor (0.0, 0.0, 0.0);
	vis.addPointCloud(cloud_cam_c);
	vis.addCoordinateSystem(10.0,"camera",0);

	while (!vis.wasStopped()){
		vis.spinOnce();
	}
	pcl::io::savePCDFileBinary(save_view,*cloud_cam_c);

	cloud->clear();
	cloud_cam->clear();
	cloud_cam_c->clear();
	vis.removePointCloud();

	pcl::io::loadPCDFile<pcl::PointXYZ> (file_path_full, *cloud);
	pcl::transformPointCloud (*cloud, *cloud_cam, view_mat);
	pcl::transformPointCloud (*cloud_cam, *cloud_cam_c, transform_2);
	cout << "The full cloud has " <<cloud->height * cloud->width << " points." << endl;

	vis.addPointCloud(cloud_cam_c);
	vis.resetStoppedFlag();
	while (!vis.wasStopped()){
		vis.spinOnce();
	}

	pcl::io::savePCDFileBinary(save_full,*cloud_cam_c);
	return (0);
}
