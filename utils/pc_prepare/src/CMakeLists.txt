cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pc_prepare)

find_package(PCL 1.2 REQUIRED)
find_package(Boost REQUIRED)

if(NOT Boost_FOUND)
    message("Boost library Not found")
endif()

include_directories(${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} ${Boost_DEFINITIONS})

add_executable (pc_prepare main.cpp)
target_link_libraries (pc_prepare pcl_common pcl_io pcl_filters pcl_visualization ${VTK_LIBRARIES} ${PCL_LIBRARIES} ${Boost_LIBRARIES})

install(TARGETS pc_prepare RUNTIME DESTINATION bin)
