#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace pcl;

void GetFileNames(string path,vector<string>& filenames)
{
	string fname;
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        cout<<"Folder doesn't Exist!"<<endl;
        return;
    }
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
        	fname = ptr->d_name;
        	if (fname[fname.length()-1] == 'j'){
        		filenames.push_back(ptr->d_name);
        	}
    }
    }
    closedir(pDir);
}


int main (int argc,char** argv){
	if (argc < 3){
	    PCL_INFO ("Usage %s [options] choose read a file(0) or files(1) under direction, and the save path\n", argv[0]);
	    PCL_INFO ("-file   select the file in mode 0\n"
	    		  "-dir    select the direction in mode 1\n"
	              "");
	    return (-1);
	  }
	string mode, file, dir, savepath, savename, plyname;
	vector<string> filenames;
	pcl::PolygonMesh mesh;
	//pcl::PolygonMesh mesh;
	mode = argv[1];
	savepath = argv[2];
	if (mode == "0"){
		console::parse_argument (argc, argv, "-file", file);
		cout << file << endl;
		pcl::io::loadOBJFile(file,mesh);
	}
	else if (mode == "1"){
		console::parse_argument (argc, argv, "-dir", dir);
		cout << dir << "contains file: " << endl;
		GetFileNames(dir,filenames);
		sort(filenames.begin(),filenames.end());
		for (int i = 0; i < filenames.size() ; i++){
			cout << "Loading " << dir+filenames[i] << endl;
			pcl::io::loadPolygonFileOBJ (dir+filenames[i],mesh);
			plyname = filenames[i];
		    vector<string> vec;
		    boost::split(vec, plyname,boost::is_any_of("."), boost::token_compress_on);
			savename = savepath+vec[0]+".ply";
			cout << "Saving to " << savename << endl;
			pcl::io::savePLYFileBinary (savename, mesh);
			vec.clear();
		}
	}
	else{
		cout << "Warning: input mode\n"<< endl;
	}
	return(0);
}
