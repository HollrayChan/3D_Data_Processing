# downsample :the size of images and neican
#endoscopy images  is  1600/1200=1.3333
#
from plyfile import PlyData, PlyElement
import numpy as np
import yaml


import os
import imageio
import numpy as np

def read_point_cloud(path):
    lists_3D_points = []
    plydata = PlyData.read(path + "structure_filtered.ply")
    for n in range(plydata['vertex'].count):
        temp = list(plydata['vertex'][n])
        temp[0] = temp[0]
        temp[1] = temp[1]
        temp[2] = temp[2]
        temp.append(1.0)
        lists_3D_points.append(temp)
    return lists_3D_points

def read_camera_intrinsic_per_view(path, tmep_ds_ratio):
    param_count = 0
    temp_camera_intrincis = np.zeros((3, 4))
    with open(path + 'camera_intrinsics_per_view') as fp:
        for line in fp:
            # Focal length
            if param_count == 0:
                temp_camera_intrincis[0][0] = float(line)*tmep_ds_ratio
                param_count = 1
            elif param_count == 1:
                temp_camera_intrincis[1][1] = float(line)*tmep_ds_ratio
                param_count = 2
            elif param_count == 2:
                temp_camera_intrincis[0][2] = float(line)*tmep_ds_ratio
                param_count = 3
            elif param_count == 3:
                temp_camera_intrincis[1][2] = float(line)*tmep_ds_ratio
                temp_camera_intrincis[2][2] = 1.0
                param_count = 4
    return temp_camera_intrincis



def read_pose_data(path):
    stream = open(path + "motion.yaml", 'r')
    doc = yaml.load(stream,yaml.FullLoader)
    keys, values = doc.items()
    poses = values[1]
    return poses, len(poses)

def get_data_balancing_scale(poses, images_count):
    traveling_distance = 0.0
    translation = np.zeros((3,), dtype=np.float)
    for i in range(images_count):
        pre_translation = np.copy(translation)
        translation[0] = poses["poses[" + str(i) + "]"]['position']['x']
        translation[1] = poses["poses[" + str(i) + "]"]['position']['y']
        translation[2] = poses["poses[" + str(i) + "]"]['position']['z']

        if i >= 1:
            traveling_distance += np.linalg.norm(translation - pre_translation)
    traveling_distance /= images_count
    return traveling_distance

def quaternion_matrix(quaternion):  
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(float).eps * 4.0:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def get_extrinsic_matrix_and_projection_matrix(poses, intrinsic_matrix, visible_view_count):
    projection_matrices = []
    extrinsic_matrices = []
    for i in range(visible_view_count):
        rigid_transform = quaternion_matrix(
            [poses["poses[" + str(i) + "]"]['orientation']['w'], 
             poses["poses[" + str(i) + "]"]['orientation']['x'],
             poses["poses[" + str(i) + "]"]['orientation']['y'],
             poses["poses[" + str(i) + "]"]['orientation']['z']])
        rigid_transform[0][3] = poses["poses[" + str(i) + "]"]['position']['x']
        rigid_transform[1][3] = poses["poses[" + str(i) + "]"]['position']['y']
        rigid_transform[2][3] = poses["poses[" + str(i) + "]"]['position']['z']

        transform = np.asmatrix(rigid_transform)
        transform = np.linalg.inv(transform)

        extrinsic_matrices.append(transform)
        projection_matrices.append(np.dot(intrinsic_matrix, transform))
    return extrinsic_matrices, projection_matrices


def read_visible_view_indexes(path):
    visible_view_indexes = []
    with open(path + 'visible_view_indexes_filtered') as fp:
        for line in fp:
            visible_view_indexes.append(int(line))
    return visible_view_indexes


def read_view_indexes_per_point(path, point_cloud_count):
    # Read the view indexes per point into a 2-dimension binary matrix
    visible_view_indexes = read_visible_view_indexes(path)
    view_indexes_per_point = np.zeros((point_cloud_count, len(visible_view_indexes)))
    point_count = -1
    with open(path + 'view_indexes_per_point_filtered') as fp:
        for line in fp:
            if int(line) < 0:
                point_count = point_count + 1
            else:
                view_indexes_per_point[point_count][visible_view_indexes.index(int(line))] = 1
    return view_indexes_per_point

def devide_points_from_images(view_indexes_per_point, lists_3D_points):    
    print('qqq')
    return  

def two_view_point_projection(point_cloud_list, projection_matrices, ori_image_h, ori_image_w):
    coords = np.zeros((len(projection_matrices), ori_image_h, ori_image_w))
    coords_white = np.zeros((len(projection_matrices), ori_image_h, ori_image_w))
    for i in range(len(projection_matrices)):
        projection_matrix = projection_matrices[i]
#        print('projection_matrix',projection_matrix.shape)
        for j in range(len(point_cloud_list)):
            point_projected_undistorted = np.asarray(projection_matrix).dot(point_cloud_list[j])
            point_projected_undistorted = point_projected_undistorted / point_projected_undistorted[2]
#            doc_print(point_projected_undistorted)
            coords[i,int(round(point_projected_undistorted[0])),int(round(point_projected_undistorted[1]))] = point_cloud[j][2]
            coords_white[i,int(round(point_projected_undistorted[0])),int(round(point_projected_undistorted[1]))] = 255
#            point.append(point_projected_undistorted)
#            print(point_projected_undistorted, file = doc)
#            print(point_projected_undistorted)    
    return coords, coords_white
#, np.savetxt('/home/chan/Desktop/3Dpoint_visualizasion/test/image.txt', coords[1,:,:],fmt="%.3f", delimiter='  ') 

def doc_print(varible):
    txet = open('out.txt','w')
    print(varible,file=txet)
    txet.close()
#    with open(path + 'view_indexes_per_point_filtered') as fp:
#        for line in fp:
#            if int(line) < 0:
#                point_count = point_count + 1
    

def multi_dim_image_visual(path, multi_dim_image): 
#    print(multi_dim_image.shape)
    for i in range(multi_dim_image.shape[0]):
        one_multi_dim_image = np.transpose(multi_dim_image[i,:,:])
#        print(one_multi_dim_image.shape)
        img = os.path.join(path,'image_%d.png'%i)
        imageio.imwrite(img, one_multi_dim_image)
        
def multi_dim_image_white_visual(path, multi_dim_image): 
#    print(multi_dim_image.shape)
    for i in range(multi_dim_image.shape[0]):
        one_multi_dim_image = np.transpose(multi_dim_image[i,:,:])
#        print(one_multi_dim_image.shape)
        img = os.path.join(path,'image_white_%d.png'%i)
        imageio.imwrite(img, one_multi_dim_image)


if __name__ == '__main__':
    
    # parameter of input
    path = './test/'
    ds_ratio = 1
    ori_image_h = 1280
    ori_image_w = 720
    
    #processing
    print('Process is beiginning')
    point_cloud = read_point_cloud(path)
#    print(point_cloud[0])
    camera_intrinsics = read_camera_intrinsic_per_view(path, ds_ratio)
    print(' modified camera_intrinsics is \n',camera_intrinsics)
    pose , num_images = read_pose_data(path)
    scale = get_data_balancing_scale(pose, num_images)
    print(scale)
    extrinsic_matrix, projection_matrix = get_extrinsic_matrix_and_projection_matrix(pose, camera_intrinsics, num_images)
    per_point = read_view_indexes_per_point(path , len(point_cloud))
    print(per_point.shape)
    [cat_images, cat_image_white] = two_view_point_projection(point_cloud, projection_matrix, ori_image_h, ori_image_w)
    multi_dim_image_visual(path, cat_images)
    multi_dim_image_white_visual(path, cat_image_white)
    
#multi_view process
#    for i in range(per_point.shape[1]):
#        exec('a%s = []'%i)
##        print('1')
#    for i in range(per_point.shape[1]):  
#        for j in range(per_point.shape[0]):
#            if per_point[j][i] == 1:
#                exec('a%s.append(point_cloud[%s]) ' %(i,j))

    
    

    