NUM_RAYS = {
    # 'nuscenes': [96, 200],
    # 'nuscenes': [384, 800],
    'nuscenes': [450, 800],
    'kitti': [176, 608],
    'kitti_raw': [176, 608],
    # 'kitti_raw': [55, 190],
}

def modify_for_eval(cfg, dataset='nuscenes', novel_depth=False, args=None):
    num_rays = NUM_RAYS[dataset]
    if args is not None and hasattr(args, "num_rays"):
        num_rays = args.num_rays
    cfg.num_rays = num_rays
    if dataset == 'nuscenes':
        cfg.train_dataset_config.update(dict(
            return_color_imgs = False,
            return_temporal_depth = False,
            return_depth = True,
            strict = False,
            cur_prob = 1.0
        ))

        cfg.val_dataset_config.update(dict(
            return_color_imgs = False,
            return_temporal_depth = False,
            return_depth = True,
            strict = False,
            cur_prob = 1.0
        ))
    elif dataset == 'kitti':
        cfg.train_dataset_config.update(dict(
            return_temporal = False,
            return_depth = True,
            strict = False,
            cur_prob = 1.0
        ))

        cfg.val_dataset_config.update(dict(
            return_temporal = False,
            return_depth = True,
            strict = False,
            cur_prob = 1.0
        ))
    elif dataset == 'kitti_raw':
        cfg.train_dataset_config.update(dict(
            cur_prob = 1.0,
            return_depth = True,
            strict = False
        ))

        cfg.val_dataset_config.update(dict(
            cur_prob = 1.0,
            return_depth = True,
            strict = False
        ))

    cfg.train_wrapper_config['phase'] = 'val'
    cfg.train_wrapper_config['use_flip'] = False
    cfg.loss['loss_cfgs'][0]['ray_resize'] = num_rays
    cfg.loss['loss_cfgs'][1]['ray_resize'] = num_rays

    cfg.model.head.update(dict(
        ray_sample_mode = 'fixed',
        ray_number = num_rays,
        trans_kw = 'img2lidar'
    ))

    if novel_depth and dataset == 'kitti':
        data_path = cfg.train_dataset_config['root']
        img_size = cfg.train_dataset_config['crop_size']
        cfg.train_dataset_config = dict(
            _delete_=True,
            type='Kitti_Novel_View_Eval',
            split = 'train',
            root = data_path,
            preprocess_root = data_path + 'preprocess',
            crop_size = img_size,
        )
            
        cfg.val_dataset_config = dict(
            _delete_=True,
            type='Kitti_Novel_View_Eval',
            split = 'val',
            root = data_path,
            preprocess_root = data_path + 'preprocess',
            crop_size = img_size,
        )

        cfg.model.head.update(dict(
            trans_kw = 'render_img2lidar'
        ))
    if novel_depth and dataset == 'nuscenes':
        data_path = cfg.train_dataset_config['data_path']
        img_size = cfg.train_dataset_config['crop_size']

        cfg.train_dataset_config = dict(
            _delete_=True,
            type='nuScenes_One_Frame_Eval',
            data_path = data_path,
            imageset = 'data/nuscenes_infos_train_temporal_v2.pkl',
            crop_size = img_size,
        )
            
        cfg.val_dataset_config = dict(
            _delete_=True,
            type='nuScenes_One_Frame_Eval',
            data_path = data_path,
            imageset = 'data/nuscenes_infos_val_temporal_v2.pkl',
            crop_size = img_size,
        )

        cfg.model.head.update(dict(
            trans_kw = 'render_img2lidar'
        ))

    return cfg