import pickle
import argparse

def arange_according_to_scene(infos, nusc):
    scenes = dict()

    for i, info in enumerate(infos):
        scene_token = nusc.get('sample', info['token'])['scene_token']
        scene_meta = nusc.get('scene', scene_token)
        scene_name = scene_meta['name']
        if not scene_name in scenes:
            scenes[scene_name] = [info]
        else:
            scenes[scene_name].append(info)
    
    sorted_scenes = []
    cnt = 0
    for k, v in scenes.items():
        sorted_v = sorted(v, key=lambda x:x['timestamp'])
        for i, sample in enumerate(sorted_v):
            sample['prev'] = cnt + i - 1
            sample['next'] = cnt + i + 1
        sorted_v[0]['prev'] = -1
        sorted_v[-1]['next'] = -1
        cnt += len(sorted_v)
        # sorted_scenes.append(sorted_v)
        sorted_scenes.extend(sorted_v)
    
    valid_idx = []
    for i, sample in enumerate(sorted_scenes):
        if sample['prev'] != -1 and sample['next'] != -1:
            valid_idx.append(i)
    
    return sorted_scenes, valid_idx


if __name__ == "__main__":

    parse = argparse.ArgumentParser('')
    parse.add_argument('--src-path', type=str, default='', help='path of the original pkl file')
    parse.add_argument('--dst-path', type=str, default='', help='path of the output pkl file')
    parse.add_argument('--data-path', type=str, default='', help='path of the nuScenes dataset')
    parse.add_argument('--hfai', action='store_true', default=False)
    args = parse.parse_args()

    with open(args.src_path, 'rb') as f:
        data = pickle.load(f)

    if args.hfai:
        from ..dataset.custom_nuscenes import CustomNuScenes as NuScenes
    else:
        from nuscenes import NuScenes

    nusc = NuScenes('v1.0-trainval', args.data_path)
    data['infos'], data['valid_idx'] = arange_according_to_scene(data['infos'], nusc)
    
    with open(args.dst_path, 'wb') as f:
        pickle.dump(data, f)
    