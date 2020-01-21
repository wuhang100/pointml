cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(view_cloud)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (view_cloud main.cpp)
target_link_libraries (view_cloud ${PCL_LIBRARIES})

install(TARGETS view_cloud RUNTIME DESTINATION bin)

