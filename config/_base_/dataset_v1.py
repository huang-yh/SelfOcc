data_path = 'data/nuscenes/'

train_wrapper_config = dict(
    phase='train', 
    scale_rate=0.5,
    photometric_aug=dict(
        use_swap_channel=False,
    )
)

val_wrapper_config = dict(
    phase='val', 
    scale_rate=0.5,
    photometric_aug=dict(
        use_swap_channel=False,
    )
)

nusc = dict(
    version = 'v1.0-trainval',
    dataroot = data_path
)

train_loader = dict(
    batch_size = 1,
    shuffle = True,
    num_workers = 1,
)
    
val_loader = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 1,
)
