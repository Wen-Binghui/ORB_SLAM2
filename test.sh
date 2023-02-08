# * 
# ./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM1.yaml data/rgbd_dataset_freiburg1_xyz

# ./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUM2.yaml data/rgbd_dataset_freiburg2_xyz

## * associate rgbd_dataset_freiburg1_xyz
# python3 associate.py data/rgbd_dataset_freiburg1_xyz/rgb.txt data/rgbd_dataset_freiburg1_xyz/depth.txt > freiburg1_xyz_associations.txt
## * RGB mode
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM1.yaml data/rgbd_dataset_freiburg1_xyz freiburg1_xyz_associations.txt