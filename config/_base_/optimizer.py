optimizer = dict(
    optimizer=dict(
        type='AdamW',
        lr=2e-5,
        weight_decay=0.0001
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),}
    ),
)

grad_max_norm = 35


