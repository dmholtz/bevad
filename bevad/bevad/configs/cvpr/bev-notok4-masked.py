### BEV Config ###
bev_image_range = 51.2 # fixed, depends on the dataset
bev_range = [-40.0, -40.0, -2.0, 40.0, 40.0, 6.0]
bev_hw_backbone = 25
map_resolution = 200

### Streaming Config
max_stream_length_train = 20 # 2s
max_stream_length_val = 100 # 10s
model_frequency = 5 # Hz

### Perception Config ###
image_backbone = "RADIO"  # "ResNet", "DINO", "RADIO"
image_resolution = (448, 800) # (448, 800) and (896, 1600) are supported resolutions for RADIO and ResNet

### Object Detection Config ###
class_names = ["vehicle", "bike", "person", "static_object", "traffic_light.red", "traffic_light.green", "traffic_sign.stop"]
class_mapping = {
    "ambulance": "vehicle",
    "bicycle": "bike",
    "bus": "vehicle",
    "car": "vehicle",
    "construction": "static_object",
    "motorcycle": "bike",
    "pedestrian": "person",
    "roadway_item": "static_object",
    "traffic_sign.warning.accident": "static_object",
    "traffic_sign.warning.construction": "static_object",
    "traffic_light_state.red": "traffic_light.red",
    "traffic_light_state.yellow": "traffic_light.red",
    "traffic_light_state.green": "traffic_light.green",
    "traffic_light_state.off": "traffic_light.green",
    "traffic_light_state.unknown": "traffic_light.green",
    "traffic_sign.stop": "traffic_sign.stop",
    "truck": "vehicle",
} # maps bevad_sim classes / attributes to BevAD classes

### Neural Network Config ###
_dim_ = 256
_ffn_dim_ = _dim_*2
_num_fpn_levels_ = 1

### planning ###
collision_frame_rate = 10
planning_steps = 15
planning_frame_rate = 5
planning_waypoints = 30

model = dict(
    type="BEVAd",
    freeze_bev=True,
    offline_bev=False,
    task_loss_weight={
        "det": 1.0,
        "map": 1.0,
        "planning": 1.0,
    },
    memory=dict(
        type="BevADMemory",
        queue_frequency=20, # Hz
        model_frequency=model_frequency,
    ),
    bev_backbone=dict(
        type="BevBackbone",
        d_model=_dim_,
        use_grid_mask=False,
        use_dynamics=True,
        use_cams_embeds=False,
        bev_size=bev_hw_backbone,
        bev_range=bev_range,
        num_feature_levels=_num_fpn_levels_,
        img_backbone=dict(
            type='RadioBackbone',
            model_version="c-radio_v3-b",
            trainable="LoRA",
        ),
        img_neck=dict(
            type='RadioNeck',
            d_input=768,
            d_output=_dim_,
        ),
        bev_encoder=dict(
            type='MyBEVFormerEncoder',
            num_layers=3,
            pc_range=bev_range,
            num_points_in_pillar=4,
            bev_h = bev_hw_backbone,
            bev_w = bev_hw_backbone,
            mask_ego = True,
            transformerlayers=dict(
                type='MyBEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='MyTemporalSelfAttention',
                        embed_dims=_dim_,
                        num_levels=1,
                        num_points=8,
                    ),
                    dict(
                        type='MySpatialCrossAttention',
                        deformable_attention=dict(
                            type='MyMSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_points=8,
                            num_levels=_num_fpn_levels_,
                            im2col_step=96, # 64->96 necessary due to large batch size
                        ),
                        embed_dims=_dim_,
                    )
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.2,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
            ),
        ),
    ),
    proxy_bev_backbone=None,
    detection_head=dict(
        type='BevDetectionHead',
        num_classes=len(class_names),
        pc_range=bev_range,
        bev_size=bev_hw_backbone,
        in_channels=_dim_,
        num_query=900,
        object_decoder=dict(
            type='MyDetectionTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    batch_first=True,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1
                        ),
                        dict(
                            type='MyCustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1,
                            im2col_step=96, # 64->96 necessary due to large batch size
                        ),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))
        ),
        sync_cls_avg_factor=False, # normalization synchronization across GPUs not needed with larger batch size
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
                pc_range=None, # legacy, not needed
            ),
        ),
    ),
    map_head=None,
    bev_crop=dict(
        type="BevCrop",
        input_size=bev_hw_backbone,
        cells_ahead=12,
        cells_behind=12,
        cells_leftright=8,
    ),
    planning_head=dict(
        type="DisentangledPointEstimatorPlanner",
        num_commands=6,
        num_planning_steps=planning_steps,
        num_bev_waypoints=planning_waypoints,
        plan_with_speed=True,
        d_model=512,
        d_bev=_dim_,
        bev_pooling=None,
        bev_unshuffling=None,
        bev_size=bev_hw_backbone,
        loss_weight=100,
        disentangled_decoder=dict(
            type="DisentangledPlanningDecoder",
            num_layers=8,
            decoder_layer=dict(
                type="DisentangledPlanningDecoderLayer",
                bev_range=bev_range,
                d_model=512,
                d_bev=_dim_,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                deformable_attn=None,
                query_lengths=(planning_waypoints, planning_steps),
            ),
        ),
        cfg_p_uncond=0.5,
    ),
)

dataset_type = "CoreDataset"
episode_base = "data/episodes"
info_root = "data/infos"
map_root = "data/bench2drive/maps"
map_file = "data/infos/b2d_map_infos.pkl"
ann_file_train=info_root + "/b2d_infos.pkl"
ann_file_val=info_root + "/b2d_infos_val.pkl"
ann_file_test=ann_file_val

data = dict(
    batch_size_train=8,
    batch_size_val=4,
    streaming=True,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        episode_base=episode_base,
        index_file="data/index/train_index.csv",
        map_file=map_file,
        oversample=False,
        camera_augmentation=True,
        max_stream_length=max_stream_length_train,
        model_frequency=model_frequency,
        with_perception=True,
        with_bev_image=True,
        with_objects=True,
        with_map=False,
        with_planning=True,
        with_actions=False,
        with_critical_actors=False,
        bev_range=bev_range,
        backbone_type=image_backbone,
        img_resolution=image_resolution,
        class_names=class_names,
        class_mapping=class_mapping,
        map_resolution=map_resolution,
        planning_frame_rate=planning_frame_rate,
        planning_frames=planning_steps,
        num_bev_waypoints=planning_waypoints,
        bev_waypoint_distance=1.0,
        require_replanning=True,
        fix_lane_change_commands=True,
    ),
    val=dict(
        type=dataset_type,
        episode_base=episode_base,
        index_file="data/index/val_index.csv",
        map_file=map_file,
        oversample=False,
        camera_augmentation=False,
        max_stream_length=max_stream_length_val,
        model_frequency=model_frequency,
        with_perception=True,
        with_bev_image=True,
        with_objects=True,
        with_map=False,
        with_planning=True,
        with_actions=False,
        with_critical_actors=False,
        bev_range=bev_range,
        backbone_type=image_backbone,
        img_resolution=image_resolution,
        class_names=class_names,
        class_mapping=class_mapping,
        map_resolution=map_resolution,
        planning_frame_rate=planning_frame_rate,
        planning_frames=planning_steps,
        num_bev_waypoints=planning_waypoints,
        bev_waypoint_distance=1.0,
        require_replanning=True,
        fix_lane_change_commands=True,
    ),
)

# config of the lightning module
lightning = dict(
    lr=2e-4,
)

# config of the lightning trainer
trainer = dict(
    accumulate_grad_batches=1,
    log_every_n_steps=20,
    max_epochs=8,
)

logger = dict(
    tags=["det", "plan", "radio", "stream", "5hz"]
)
