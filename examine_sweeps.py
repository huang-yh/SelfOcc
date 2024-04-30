#### check nuscenes sweeps
from nuscenes import NuScenes
import mmengine
from copy import deepcopy
import numpy as np

nusc = NuScenes(dataroot='data/nuscenes', version='v1.0-trainval')
# pkl = mmengine.load('data/nuscenes_infos_train_temporal_v1.pkl')
pkl = mmengine.load('data/nuscenes_infos_val_temporal_v1.pkl')

sensor_types = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'LIDAR_TOP']

def gather_sensor(sample_data_token):
    sample_data = nusc.get('sample_data', sample_data_token)
    data = deepcopy(sample_data)
    data.update(dict(
        pose=nusc.get('ego_pose', data['ego_pose_token']),
        calib=nusc.get('calibrated_sensor', data['calibrated_sensor_token'])))
    return data

scenes = dict()
for sample in pkl['infos']:
    scene_token = nusc.get('sample', sample['token'])['scene_token']
    scene_meta = nusc.get('scene', scene_token)
    scene_name = scene_meta['name']
    if not scene_name in scenes:
        scenes.update({scene_name: scene_token})

all_infos = dict()
meta_data = []

for scene_name, scene_token in scenes.items():
    print(f'processing {scene_name}')
    scene_meta = nusc.get('scene', scene_token)
    sample_sweep_list = []
    
    sample_token = scene_meta['first_sample_token']

    while sample_token:

        sample = nusc.get('sample', sample_token)
        sample_dict = deepcopy(sample)
        data = dict()
        for sensor_type in sensor_types:
            data.update({sensor_type: gather_sensor(sample['data'][sensor_type])})
        data.update({'LIDAR_TOP': gather_sensor(sample['data']['LIDAR_TOP'])})
        sample_dict.update(dict(data=data, is_key_frame=True))
        sample_sweep_list.append(sample_dict)
        meta_data.append((scene_token, len(sample_sweep_list) - 1))

        sweeps = dict()
        for sensor_type in sensor_types:
            sweeps_sensor = []
            sweep_token = data[sensor_type]['next']
            while sweep_token:
                sweep_data = gather_sensor(sweep_token)
                if sweep_data['is_key_frame']:
                    break
                sweeps_sensor.append(sweep_data)
                sweep_token = sweep_data['next']
            sweeps.update({sensor_type: sweeps_sensor})

        least_length = len(sweeps[sensor_types[0]])
        least_sensor_type = sensor_types[0]
        for sensor_type in sensor_types[1:]:
            length = len(sweeps[sensor_type])
            if length < least_length:
                least_length = length
                least_sensor_type = sensor_type
        
        if least_length == 0:
            sample_token = sample['next']
            continue
        
        ref_timestamps = [v['timestamp'] for v in sweeps[least_sensor_type]]
        for timestamp in ref_timestamps:
            sweep_dict = dict(timestamp=timestamp, is_key_frame=False)
            sweep_dict_data = dict()
            for sensor_type in sensor_types:
                intervals = [v['timestamp'] - timestamp for v in sweeps[sensor_type]]
                nearest = np.argmin(np.abs(intervals))
                sweep_dict_data.update({sensor_type: sweeps[sensor_type][nearest]})
            sweep_dict.update({'data': sweep_dict_data})
            sample_sweep_list.append(sweep_dict)
        
        sample_token = sample['next']
    
    all_infos.update({scene_token: sample_sweep_list})

    # sample = nusc.get('sample', sample_token)
    # sweeps = dict()
    # for sensor_type in sensor_types:
    #     sweeps_sensor = []
    #     sweep_token = sample['data'][sensor_type]
    #     while sweep_token:
    #         sweep_meta = nusc.get('sample_data', sweep_token)
    #         sweep = deepcopy(sweep_meta)
    #         sweep.update(dict(
    #             pose=nusc.get('ego_pose', sweep_meta['ego_pose_token']),
    #             calib=nusc.get('calibrated_sensor', sweep_meta['calibrated_sensor_token'])))
    #         sweeps_sensor.append(sweep)
    #         sweep_token = sweep_meta['next']
    #     sweeps.update({sensor_type: sweeps_sensor})
    
    # least_sensor_type = sensor_types[0]
    # least_length = len(sweeps[least_sensor_type])
    # for sensor_type in sensor_types[1:]:
    #     length = len(sweeps[sensor_type])
    #     if length < least_length:
    #         least_length = length
    #         least_sensor_type = sensor_type
    
    # ref_timestamps = [s['timestamp'] for s in sweeps[least_sensor_type]]
    # import pdb; pdb.set_trace()
# mmengine.dump({'infos': all_infos, 'metadata': meta_data}, 'data/nuscenes_infos_train_sweeps_lid.pkl')

mmengine.dump({'infos': all_infos, 'metadata': meta_data}, 'data/nuscenes_infos_val_sweeps_lid.pkl')
pass
