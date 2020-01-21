#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/common/common.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/range_image/range_image.h>
#include <math.h>

using namespace std;
using namespace pcl;

void viewcloud (pcl::visualization::PCLVisualizer& viewer, pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud){
	//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> point_color (cloud, r, g, b);
	//pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> point_color (cloud);
	viewer.setBackgroundColor (0.0, 0.0, 0.0);
	//viewer.addPointCloud(cloud,point_color,id);
	viewer.addPointCloud(cloud);
	viewer.addCoordinateSystem();
}

int main(int argc, char** argv){
	string file_path;

	std::vector<int> p_file_indices_pcd = console::parse_file_extension_argument (argc, argv, ".pcd");
	std::stringstream filename_stream;
	if (!p_file_indices_pcd.empty ()){
		filename_stream << argv[p_file_indices_pcd.at (0)];
	}
	file_path = filename_stream.str();



	pcl::visualization::PCLVisualizer viewer;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile<pcl::PointXYZ> (file_path, *cloud);
	viewcloud(viewer,cloud);

	while (!viewer.wasStopped()){
		viewer.spinOnce();
	}


	return (0);
}
