import numpy as np
# import scenerf.data.utils.fusion as fusion
# import open3d as o3d
from PIL import Image


# def make_open3d_point_cloud(xyz, color=None):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)
#     if color is not None:
#         if len(color) != len(xyz):
#             color = np.tile(color, (len(xyz), 1))
#         pcd.colors = o3d.utility.Vector3dVector(color)
#     return pcd


def apply_transform(pts, T):
    """
    cam_ptx_from: B, 3
    """
    ones = np.ones((pts.shape[0], 1))
    homo_pts = np.concatenate([pts, ones], axis=1)
    homo_cam_pts_to = (T @ homo_pts.T).T
    cam_pts_to = homo_cam_pts_to[:, :3]

    return cam_pts_to


def dump_xyz(P):
    return P[0:3, 3]


def read_rgb(path):
    img = Image.open(path).convert("RGB")

    # PIL to numpy
    img = np.array(img, dtype=np.float32, copy=False) / 255.0
    img = img[:370, :1220, :]  # crop image        

    return img


def read_poses(path):
    # Read and parse the poses
    poses = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
            T_w_cam0 = T_w_cam0.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            poses.append(T_w_cam0)
    return poses
    

def read_calib(calib_path):
    """
    Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
    :param calib_path: Path to a calibration text file.
    :return: dict with calibration matrices.
    """
    calib_all = {}
    with open(calib_path, "r") as f:
        for line in f.readlines():
            if line == "\n":
                break
            key, value = line.split(":", 1)
            calib_all[key] = np.array([float(x) for x in value.split()])

    # reshape matrices
    calib_out = {}
    # 3x4 projection matrix for left camera
    calib_out["P2"] = calib_all["P2"].reshape(3, 4)
    calib_out["Tr"] = np.identity(4)  # 4x4 matrix
    calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)

    T2 = np.eye(4)
    T2[0, 3] = calib_out["P2"][0, 3] / calib_out["P2"][0, 0]
    calib_out["T_cam0_2_cam2"] = T2
    return calib_out


# def compute_transformation(
#         lidar_path_source, lidar_path_infer, lidar_path_target,
#         pose_source, pose_infer, pose_target,
#         T_velo_2_cam2, T_cam0_2_cam2):
#     pts_velo_source = np.fromfile(lidar_path_source, dtype=np.float32).reshape((-1, 4))[:, :3]
#     pts_velo_infer = np.fromfile(lidar_path_infer, dtype=np.float32).reshape((-1, 4))[:, :3]
#     pts_velo_target = np.fromfile(lidar_path_target, dtype=np.float32).reshape((-1, 4))[:, :3]

#     pts_cam2_source = apply_transform(pts_velo_source, T_velo_2_cam2)
#     pts_cam2_infer = apply_transform(pts_velo_infer, T_velo_2_cam2)
#     pts_cam2_target = apply_transform(pts_velo_target, T_velo_2_cam2)

#     T_cam2_2_cam0 = np.linalg.inv(T_cam0_2_cam2)

#     T_source2infer = T_cam0_2_cam2 @ np.linalg.inv(pose_infer) @ pose_source @ T_cam2_2_cam0
#     T_source2target = T_cam0_2_cam2 @ np.linalg.inv(pose_target) @ pose_source @ T_cam2_2_cam0
#     pts_cam2_source2infer = apply_transform(pts_cam2_source, T_source2infer)
#     pts_cam2_source2target = apply_transform(pts_cam2_source, T_source2target)
#     pcd_source2infer = make_open3d_point_cloud(pts_cam2_source2infer).voxel_down_sample(voxel_size=0.05)
#     pcd_source2target = make_open3d_point_cloud(pts_cam2_source2target).voxel_down_sample(voxel_size=0.05)
#     pcd_infer = make_open3d_point_cloud(pts_cam2_infer).voxel_down_sample(voxel_size=0.05)
#     pcd_target = make_open3d_point_cloud(pts_cam2_target).voxel_down_sample(voxel_size=0.05)

#     reg_source2infer = o3d.pipelines.registration.registration_icp(
#         pcd_source2infer, pcd_infer, 0.2, np.eye(4),
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

#     reg_source2target = o3d.pipelines.registration.registration_icp(
#         pcd_source2target, pcd_target, 0.2, np.eye(4),
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

#     return {
#         "T_source2infer": T_source2infer @ reg_source2infer.transformation,
#         "T_source2target": T_source2target @ reg_source2target.transformation
#     }


  
# def vox2pix(cam_E, cam_K, 
#             vox_origin, voxel_size, 
#             img_W, img_H, 
#             scene_size):
#     """
#     compute the 2D projection of voxels centroids
    
#     Parameters:
#     ----------
#     cam_E: 4x4
#        =camera pose in case of NYUv2 dataset
#        =Transformation from camera to lidar coordinate in case of SemKITTI
#     cam_K: 3x3
#         camera intrinsics
#     vox_origin: (3,)
#         world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
#     img_W: int
#         image width
#     img_H: int
#         image height
#     scene_size: (3,)
#         scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2
    
#     Returns
#     -------
#     projected_pix: (N, 2)
#         Projected 2D positions of voxels
#     fov_mask: (N,)
#         Voxels mask indice voxels inside image's FOV 
#     sensor_distance: (N,)
#         Voxels'distance to the sensor in meter
#     """
#     # Compute the x, y, z bounding of the scene in meter
    
#     vol_bnds = np.zeros((3,2))
#     vol_bnds[:,0] = vox_origin
#     vol_bnds[:,1] = vox_origin + np.array(scene_size)

#     # Compute the voxels centroids in lidar cooridnates
#     vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
#     xv, yv, zv = np.meshgrid(
#             range(vol_dim[0]),
#             range(vol_dim[1]),
#             range(vol_dim[2]),
#             indexing='ij'
#           )
#     vox_coords = np.concatenate([
#             xv.reshape(1,-1),
#             yv.reshape(1,-1),
#             zv.reshape(1,-1)
#           ], axis=0).astype(int).T

#     # Project voxels'centroid from lidar coordinates to camera coordinates
#     cam_pts = fusion.TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
#     cam_pts = fusion.rigid_transform(cam_pts, cam_E)

#     # Project camera coordinates to pixel positions
#     projected_pix = fusion.TSDFVolume.cam2pix(cam_pts, cam_K)
#     pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

#     # Eliminate pixels outside view frustum
#     sensor_distance = cam_pts[:, 2]
#     fov_mask = np.logical_and(pix_x >= 0,
#                 np.logical_and(pix_x < img_W,
#                 np.logical_and(pix_y >= 0,
#                 np.logical_and(pix_y < img_H,
#                 sensor_distance > 0))))


#     return projected_pix, fov_mask, sensor_distance
