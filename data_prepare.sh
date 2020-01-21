#!/bin/bash



# sample 2048 data -> name.pcd
# sample single view data -> namef.pc, namev.pcd

for n in $(seq 6 1 8)
do 
  pcl_mesh_sampling data/ply_data/sofa$n.ply nn_input/pcd/sofa$n.pcd -n_samples 4096 -no_vis_result 1

for i in $(seq 10 20 170)
do
  mkdir -p nn_input/pcd/sofa$n'v'$i/
for j in {1,0.5,0.25,0.125,0.1}
do
  echo dealing with sofa $n res $j angle $i 
  utils/pc_prepare/build/pc_prepare data/ply_data/sofa$n.ply -vert_res $j -hor_res $j -full_cloud_path nn_input/pcd/sofa$n.pcd -object_coordinates 1 -cam_distance_ratio 5 -view_angle $i,70 -save_full nn_input/pcd/sofa$n'v'$i.pcd -save_view nn_input/pcd/sofa$n'v'$i/res$j.pcd -vismode 0

done
done
done

python utils/pcd2deep.py
echo Done!

