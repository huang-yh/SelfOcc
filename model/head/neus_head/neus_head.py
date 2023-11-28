import torch, torch.distributed as dist, torch.nn as nn, os
from nerfstudio.models.neus_custom import NeuSCustomModelConfig
from nerfstudio.fields.sdf_custom_field import SDFCustomFieldConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from ..base_head import BaseTaskHead
from ..nerfacc_head.ray_sampler import RaySampler
from ..nerfacc_head.img2lidar import Img2LiDAR
from mmseg.models import HEADS
from mmengine.logging import MMLogger
import math
logger = MMLogger.get_instance('selfocc')
from utils.tb_wrapper import WrappedTBWriter
if 'selfocc' in WrappedTBWriter._instance_dict:
    writer = WrappedTBWriter.get_instance('selfocc')
else:
    writer = None


@HEADS.register_module()
class NeuSHead(BaseTaskHead):

    def __init__(
            self, 
            roi_aabb,
            resolution=0.4,
            near_plane=0.0,
            far_plane=1e10,
            num_samples=64,
            num_samples_importance=64,
            num_up_sample_steps=4,
            base_variance=64,
            beta_init=0.1,

            beta_max=0.195,
            total_iters=3516 * 11,
            use_numerical_gradients=True,
            numerical_gradients_delta=0.01,
            use_uniform_gradient=False,
            nbr_gradient_points=128*128*16,
            calculate_online=False,
            sample_gradient=False,
            use_compact_2nd_grad=False,
            beta_hand_tune=False,

            return_uniform_sdf=False,
            estimate_flow=False,
            return_max_depth=False,
            return_surface_sdf=False,
            return_second_grad=False,
            return_sample_sdf=False,
            return_sem=False,
            disp_sampler=False,

            anneal_aabb=False,
            aabb_every_iters=3516,
            aabb_min_near=10.,
            aabb_min_far_frac=0.25,

            # rays args
            ray_sample_mode='fixed',    # fixed, cellular
            ray_number=[192, 400],      # 192 * 400
            ray_img_size=[768, 1600],
            ray_upper_crop=0,
            ray_x_dsr_max=None,
            ray_y_dsr_max=None,
            # img2lidar args
            trans_kw='img2lidar',
            trans_kw_eval=None,
            novel_view=None,

            # render args
            render_bkgd='white',

            # bev_inner = 128,
            # bev_outer = 32,
            # range_inner = 51.2,
            # range_outer = 51.2,
            # nonlinear_mode = "linear_upscale",
            # z_inner = 20,
            # z_outer = 10,
            # z_ranges = [-5.0, 3.0, 11.0],
            mapping_args=dict(
                nonlinear_mode="linear_upscale",
                h_size=[128, 32],
                h_range=[51.2, 28.8],
                h_half=False,
                w_size=[128, 32],
                w_range=[51.2, 28.8],
                w_half=False,
                d_size=[20, 10],
                d_range=[-4.0, 4.0, 12.0]),

            # mlp decoder
            embed_dims = 128,
            color_dims = 0,
            density_layers = 2,
            sh_deg = 2,
            sh_act = "relu",

            init_cfg=None, 
            print_freq=50,
            two_split=True,
            tpv=False,
            using_2d_img_feats=False,
            **kwargs):
        super().__init__(init_cfg, **kwargs)

        self.ray_sampler = RaySampler(
            ray_sample_mode=ray_sample_mode,
            ray_number=ray_number,
            ray_img_size=ray_img_size,
            ray_upper_crop=ray_upper_crop,
            ray_x_dsr_max=ray_x_dsr_max,
            ray_y_dsr_max=ray_y_dsr_max)
        
        self.ray_sampler_eval = RaySampler(
            ray_sample_mode='fixed',
            ray_number=ray_number,
            ray_img_size=ray_img_size,
            ray_upper_crop=ray_upper_crop)
        
        self.img2lidar = Img2LiDAR(
            trans_kw=trans_kw,
            trans_kw_eval=trans_kw_eval,
            novel_view=novel_view)

        model_config = NeuSCustomModelConfig(
            enable_collider=True,
            near_plane=near_plane,
            far_plane=far_plane,
            background_color=render_bkgd,
            background_model='none',
            scene_contraction_activate=False,
            num_samples=num_samples,
            num_samples_importance=num_samples_importance,
            num_up_sample_steps=num_up_sample_steps,
            base_variance=base_variance,
            beta_hand_tune=beta_hand_tune,
            beta_min=beta_init,
            beta_max=beta_max,
            total_iters=total_iters,
            perturb=True,
            use_lpips=False,

            anneal_aabb=anneal_aabb,
            aabb_every_iters=aabb_every_iters,
            aabb_max_near=near_plane,
            aabb_min_near=aabb_min_near,
            aabb_min_far_frac=aabb_min_far_frac,
            aabb_original=roi_aabb,

            disp_sampler=disp_sampler,
            return_sem=return_sem,

            sdf_field=SDFCustomFieldConfig(
                beta_init = beta_init,
                # beta_hand_tune=beta_hand_tune,
                use_numerical_gradients = use_numerical_gradients,
                # bev_inner = bev_inner,
                # bev_outer = bev_outer,
                # range_inner = range_inner,
                # range_outer = range_outer,
                # nonlinear_mode = nonlinear_mode,
                # z_inner = z_inner,
                # z_outer = z_outer,
                # z_ranges = z_ranges,
                mapping_args = mapping_args,
                # mlp decoder
                embed_dims = embed_dims,
                color_dims = color_dims,
                density_layers = density_layers,
                sh_deg = sh_deg,
                sh_act = sh_act,
                beta_learnable = not beta_hand_tune,
                second_derivative= return_second_grad,
                tpv = tpv,
                use_uniform_gradient=use_uniform_gradient,
                nbr_gradient_points=nbr_gradient_points,
                calculate_online=calculate_online,
                sample_gradient=sample_gradient,
                use_compact_2nd_grad=use_compact_2nd_grad,
                using_2d_img_feats=using_2d_img_feats,
                return_sem=return_sem,
            ),
        )

        self.model = model_config.setup(
            scene_box=SceneBox(
                aabb=torch.tensor([roi_aabb[:3], roi_aabb[3:]]),
                near=near_plane,
                far=far_plane,
                collider_type='box'
            ),
            num_train_data=0,
        )
        self.model.field.set_numerical_gradients_delta(numerical_gradients_delta)
        self.print_freq = print_freq
        self.resolution = resolution
        self.aabb = roi_aabb
        self.return_uniform_sdf = return_uniform_sdf
        self.return_max_depth = return_max_depth
        self.return_surface_sdf = return_surface_sdf
        self.return_second_grad = return_second_grad
        self.return_sample_sdf = return_sample_sdf
        self.return_sem = return_sem

        self.estimate_flow = estimate_flow
        self.z_size = self.model.field.mapping.size_d
        self.bev_size = [self.model.field.mapping.size_h, self.model.field.mapping.size_w]
        self.two_split = two_split
        self.using_2d_img_feats = using_2d_img_feats

        if estimate_flow:
            # self.flow_net = nn.Sequential(
            #     nn.Conv2d(embed_dims*2, embed_dims, 3, 1, 1, bias=False),
            #     nn.BatchNorm2d(embed_dims),
            #     nn.ReLU(True),
            #     nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, bias=False),
            #     nn.BatchNorm2d(embed_dims),
            #     nn.ReLU(True),
            #     nn.Conv2d(embed_dims, self.z_size * 3, 1))
            flow_net = [
                nn.Conv2d(embed_dims*2, embed_dims, 3, 1, 1, bias=False),
                nn.BatchNorm2d(embed_dims),
                nn.ReLU(True),
                nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, bias=False),
                nn.BatchNorm2d(embed_dims),
                nn.ReLU(True)]
            last_lin = nn.Conv2d(embed_dims, self.z_size * 3, 1)
            nn.init.normal_(last_lin.weight.data, 0., 1e-2)
            nn.init.constant_(last_lin.bias.data, 0.)
            flow_net.append(last_lin)
            self.flow_net = nn.Sequential(*flow_net)

    def forward_occ(
            self,
            representation,
            metas=None,
            **kwargs):
        
        if isinstance(representation, (tuple, list)):
            device = representation[0].device
        else:
            device = representation.device
        
        if not self.using_2d_img_feats:
            self.model.field.pre_compute_density_color(representation)
        else:
            self.model.field.pre_compute_density_color(
                representation, img_feats=kwargs['ms_img_feats'], img_metas=metas)
        aabb = kwargs['aabb'] if 'aabb' in kwargs else self.aabb
        reso = kwargs['resolution'] if 'resolution' in kwargs else self.resolution
        if self.return_sem:
            sdf, sem, sem_logits, _ = self.get_uniform_sdf(aabb, reso, device=device)
            # curr_s = self.model.field.deviation_network.get_variance().detach().item()
            # logger.info(f's: {curr_s}')
            return {'sdf': sdf, 'rep': representation, 'sem': sem, 'logits': sem_logits, 'xyz': _}
        sdf, _ = self.get_uniform_sdf(aabb, reso, device=device)
        # curr_s = self.model.field.deviation_network.get_variance().detach().item()
        # logger.info(f's: {curr_s}')
        return {'sdf': sdf, 'rep': representation, 'xyz': _}
    
    def get_uniform_sdf(self, aabb, resolution, device, shift=False):
        xs = torch.linspace(
            aabb[0], aabb[3], int((aabb[3] - aabb[0]) / resolution), device=device)
        ys = torch.linspace(
            aabb[1], aabb[4], int((aabb[4] - aabb[1]) / resolution), device=device)
        zs = torch.linspace(
            aabb[2], aabb[5], int((aabb[5] - aabb[2]) / resolution), device=device)
        W, H, D = len(xs), len(ys), len(zs)
        xyzs = torch.stack([
            xs[None, :, None].expand(H, W, D),
            ys[:, None, None].expand(H, W, D),
            zs[None, None, :].expand(H, W, D)
        ], dim=-1).flatten(0, 2)

        if shift:
            random_shift = torch.rand_like(xyzs) * resolution
            xyzs = xyzs + random_shift
        
        if self.return_sem:
            h = self.model.field.forward_geonetwork(xyzs)
            sdf, sem = h[..., 0], h[..., 4:]
            sdf = sdf.reshape(H, W, D)
            sem_logits = sem.reshape(H, W, D, -1)
            sem = torch.argmax(sem, dim=-1).reshape(H, W, D)
            return sdf, sem, sem_logits, xyzs.reshape(H, W, D, -1)
        else:
            sdf = self.model.field.forward_sdfnetwork(xyzs)
            sdf = sdf.reshape(H, W, D)
            return sdf, xyzs.reshape(H, W, D, -1)
    
    def prepare(
            self, 
            representation, 
            metas=None, 
            **kwargs):
        # self.model.field.pre_compute_density_color(representation)# , torch.float)
        if not self.using_2d_img_feats:
            self.model.field.pre_compute_density_color(representation)
        else:
            self.model.field.pre_compute_density_color(
                representation, img_feats=kwargs['ms_img_feats'], img_metas=metas)
        return {}

    def render(
            self,
            metas=None,
            batch=0,
            **kwargs):
        amp = 'amp' in os.environ and os.environ['amp'] == 'true'
        with torch.cuda.amp.autocast(enabled=amp):
            if os.environ.get('eval', 'false') == 'true':
                ray_sampler = self.ray_sampler_eval
            else:
                ray_sampler = self.ray_sampler 
            rays = ray_sampler()

            origin, direction = self.img2lidar(metas, rays) # B, N, 3; B, N, R, 3
            bs, num_cams, num_rays = direction.shape[:3]
            assert bs == 1, 'only support bs = 1 currently'
            origin = origin.unsqueeze(2).repeat(1, 1, num_rays, 1).flatten(0, 2)
            direction = direction.flatten(0, 2)
            direction_norm = torch.norm(direction, dim=-1, keepdim=True)
            direction = direction / direction_norm

            if batch > 0:
                output = {
                    'rgb': [],
                    'acc': [],
                    'depth': [],
                    'weights': [],
                    'ts': [],
                    'deltas': [],
                    'vis_normal': [],
                    'sem': []
                    # 'sdfs': []
                }
                chunks = direction.shape[0] * 1.0 / batch
                chunks = int(math.ceil(chunks))
                origins = torch.chunk(origin, chunks)
                directions = torch.chunk(direction, chunks)
                direction_norms = torch.chunk(direction_norm, chunks)
                for i_chunk in range(chunks):
                    # torch.cuda.empty_cache()
                    ray_bundle = RayBundle(
                        origins=origins[i_chunk],
                        directions=directions[i_chunk],
                        directions_norm=direction_norms[i_chunk],
                        pixel_area=torch.zeros_like(direction_norms[i_chunk]))
                    output_i = self.model(ray_bundle)

                    output['rgb'].append(output_i['rgb'])
                    if self.return_sem:
                        output['sem'].append(output_i['sem'])
                    output['vis_normal'].append(output_i['normal_vis'])
                    output['acc'].append(output_i['accumulation'])
                    output['depth'].append(output_i['depth'])
                    # output['sdfs'].append(output_i['field_outputs'][FieldHeadNames.SDF])
                    # depth_median = output['depth_median'].reshape(bs, num_cams, num_rays)
                    output['weights'].append(output_i['weights'])
                    ray_samples = output_i['ray_samples']
                    # there is bug here before 8.16, did not divide direction_norm
                    ts = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
                    # ts = ts.reshape(bs, num_cams, num_rays, num_samples_per_ray, 1)
                    ts = ts / direction_norms[i_chunk].unsqueeze(1)
                    output['ts'].append(ts)
                    # import pdb; pdb.set_trace()
                    deltas = (ray_samples.frustums.ends - ray_samples.frustums.starts)
                    # deltas = deltas.reshape(bs, num_cams, num_rays, num_samples_per_ray, 1)
                    deltas = deltas / direction_norms[i_chunk].unsqueeze(1)
                    output['deltas'].append(deltas)
                
                rgb = torch.cat(output['rgb']).reshape(bs, num_cams, num_rays, -1)
                if self.return_sem:
                    sem = torch.cat(output['sem']).reshape(bs, num_cams, num_rays, -1)
                vis_normal = torch.cat(output['vis_normal']).reshape(bs, num_cams, num_rays, -1)
                acc = torch.cat(output['acc']).reshape(bs, num_cams, num_rays)
                depth = torch.cat(output['depth']).reshape(bs, num_cams, num_rays)
                # sdfs = torch.cat(output['sdfs']).reshape(bs, num_cams, num_rays, -1)
                weights = torch.cat(output['weights']).reshape(bs, num_cams, num_rays, -1, 1)
                ts = torch.cat(output['ts'])
                deltas = torch.cat(output['deltas'])

            else:
                ray_bundle = RayBundle(
                    origins=origin,#.float(),
                    directions=direction,#.float(),
                    directions_norm=direction_norm,#.float(),
                    pixel_area=torch.zeros_like(direction_norm))#, dtype=torch.float))

                output = self.model(ray_bundle)

                """
                output: dict(
                    "rgb": B, N, R, 3
                    "accumulation": B, N, R, 1
                    "depth": B, N, R, 1
                    "normal": B, N, R, 3
                    "weights": B, N, R, S, 1,
                    "ray_points": B, N, R, S, 3
                    "directions_norm": B, N, R, 1
                    "normal_vis":
                    while training
                    "eik_grad": 
                    "points_norm":
                )
                """
                rgb = output['rgb'].reshape(bs, num_cams, num_rays, -1)
                if self.return_sem:
                    sem = output['sem'].reshape(bs, num_cams, num_rays, -1)
                vis_normal = output['normal_vis'].reshape(bs, num_cams, num_rays, -1)
                acc = output['accumulation'].reshape(bs, num_cams, num_rays)
                depth = output['depth'].reshape(bs, num_cams, num_rays)
                sdfs = output['field_outputs'][FieldHeadNames.SDF].reshape(bs, num_cams, num_rays, -1)
                # depth_median = output['depth_median'].reshape(bs, num_cams, num_rays)
                weights = output['weights'].reshape(bs, num_cams, num_rays, -1, 1)
                ray_samples = output['ray_samples']
                # there is bug here before 8.16, did not divide direction_norm
                ts = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
                # ts = ts.reshape(bs, num_cams, num_rays, num_samples_per_ray, 1)
                ts = ts / direction_norm.unsqueeze(1)
                # import pdb; pdb.set_trace()
                deltas = (ray_samples.frustums.ends - ray_samples.frustums.starts)
                # deltas = deltas.reshape(bs, num_cams, num_rays, num_samples_per_ray, 1)
                deltas = deltas / direction_norm.unsqueeze(1)

        if self.return_max_depth:
            eps = torch.finfo(deltas.dtype).eps
            deltas_ = deltas.reshape(bs, num_cams, num_rays, -1).cpu()
            weights_ = weights.reshape(bs, num_cams, num_rays, -1).cpu() # .clone()
            weights_[deltas_ < eps] = 0.
            w_per_d = weights_ / deltas_.clamp_min(eps)
            indices = w_per_d.argmax(dim=-1, keepdim=True) # bs, num_cams, num_rays, 1
            max_depth = ts.reshape(bs, num_cams, num_rays, -1).cpu()
            max_depth = torch.gather(max_depth, -1, indices).squeeze(-1).to(rgb.device)

        # www = weights.squeeze(0).squeeze(-1).cpu()
        # ddd = deltas.reshape(bs, num_cams, num_rays, -1).squeeze(0).cpu()
        # sss = sdfs
        # sss = sss.reshape(bs, num_cams, num_rays, -1).squeeze(0).cpu()
        # ttt = ts.reshape(bs, num_cams, num_rays, -1).squeeze(0).cpu()
        # import matplotlib.pyplot as plt
        # while True:
        #     plt.cla()
        #     fig, ax = plt.subplots(3, 1)
        #     cam = 0; ray=1643
        #     import pdb; pdb.set_trace()
        #     ax[0].plot(ttt[cam, ray], sss[cam, ray])
        #     ax[0].plot(ttt[cam, ray], torch.zeros(256))
        #     ax[1].plot(ttt[cam, ray], www[cam, ray])
        #     ax[2].plot(ttt[cam, ray], www[cam, ray] / (ddd[cam, ray] + 1e-6))
        #     plt.savefig(f'wdelta_w_sdf_t_pro_{cam}_{ray}.png')
        #     print(cam, ray)
        #     import pdb; pdb.set_trace()

        outputs = {
            'ms_depths': [depth],
            # 'ms_depths_median': [depth_median],
            'ms_colors': [rgb],
            'vis_normal': [vis_normal],
            'ms_accs': [acc],
            'ms_rays': rays}
        if self.return_max_depth:
            outputs.update({
                'ms_max_depths': [max_depth]})
        if self.return_sem:
            outputs.update({'sem': [sem]})
        return outputs

    def forward(
            self, 
            representation, 
            metas=None, 
            **kwargs):
        
        estimate_flow = self.estimate_flow and (kwargs.get('prev_rep', None) is not None or self.training)
        
        # self.model.field.pre_compute_density_color(representation)#, torch.float)
        if not self.using_2d_img_feats:
            self.model.field.pre_compute_density_color(representation)
        else:
            self.model.field.pre_compute_density_color(
                representation, img_feats=kwargs['ms_img_feats'], img_metas=metas)

        if estimate_flow:
            assert not isinstance(representation, list) # TODO
            prev_rep = kwargs.get('prev_rep', None)
            next_rep = kwargs.get('next_rep', None)
            assert prev_rep is not None and \
                next_rep is not None
            prev_rep = prev_rep.unflatten(1, self.bev_size).permute(0, 3, 1, 2)
            next_rep = next_rep.unflatten(1, self.bev_size).permute(0, 3, 1, 2)
            curr_rep = representation.unflatten(1, self.bev_size).permute(0, 3, 1, 2)
            prev_curr = torch.cat([prev_rep, curr_rep], dim=1)
            next_curr = torch.cat([next_rep, curr_rep], dim=1)
            curr_prev = torch.cat([curr_rep, prev_rep], dim=1)
            curr_next = torch.cat([curr_rep, next_rep], dim=1)
            curr2prev_flow = self.flow_net(prev_curr).unflatten(1, (3, self.z_size)).permute(0, 1, 3, 4, 2) # bs, 3, h, w, z
            curr2next_flow = self.flow_net(next_curr).unflatten(1, (3, self.z_size)).permute(0, 1, 3, 4, 2)
            prev2curr_flow = self.flow_net(curr_prev).unflatten(1, (3, self.z_size)).permute(0, 1, 3, 4, 2)
            next2curr_flow = self.flow_net(curr_next).unflatten(1, (3, self.z_size)).permute(0, 1, 3, 4, 2)

        global_iter = kwargs.get('global_iter', None)
        amp = 'amp' in os.environ and os.environ['amp'] == 'true'
        with torch.cuda.amp.autocast(enabled=amp):
            if os.environ.get('eval', 'false') == 'true':
                ray_sampler = self.ray_sampler_eval
            else:
                ray_sampler = self.ray_sampler 
            rays = ray_sampler()

            origin, direction = self.img2lidar(metas, rays) # B, N, 3; B, N, R, 3
            bs, num_cams, num_rays = direction.shape[:3]
            assert bs == 1, 'only support bs = 1 currently'
            origin = origin.unsqueeze(2).repeat(1, 1, num_rays, 1).flatten(0, 2)
            direction = direction.flatten(0, 2)
            direction_norm = torch.norm(direction, dim=-1, keepdim=True)
            direction = direction / direction_norm
            # camera_indices = torch.arange(bs * num_cams, device=origin.device).unsqueeze(-1).repeat(1, num_rays).reshape(-1, 1)

            ray_bundle = RayBundle(
                origins=origin,#.float(),
                directions=direction,#.float(),
                directions_norm=direction_norm,#.float(),
                # camera_indices=camera_indices,
                pixel_area=torch.zeros_like(direction_norm))#, dtype=torch.float)

            output = self.model(ray_bundle, iter=global_iter)
            uniform_sdf = None
            if self.return_uniform_sdf or estimate_flow:
                uniform_data = self.get_uniform_sdf(self.aabb, self.resolution, rays.device, True)
                if self.return_sem:
                    uniform_sdf, _, _, uniform_xyz = uniform_data
                else:
                    uniform_sdf, uniform_xyz = uniform_data
        """
        output: dict(
            "rgb": B, N, R, 3
            "accumulation": B, N, R, 1
            "depth": B, N, R, 1
            "normal": B, N, R, 3
            "weights": B, N, R, S, 1,
            "ray_points": B, N, R, S, 3
            "directions_norm": B, N, R, 1
            "normal_vis":
            while training
            "eik_grad": 
            "points_norm":
        )
        """
        rgb = output['rgb'].reshape(bs, num_cams, num_rays, -1)
        acc = output['accumulation'].reshape(bs, num_cams, num_rays)
        depth = output['depth'].reshape(bs, num_cams, num_rays)
        fars = output['fars'].reshape(bs, num_cams, num_rays)
        if self.return_surface_sdf:
            surface_sdf = self.model.field.forward_sdfnetwork(output['surface_points'])
            surface_sdf = surface_sdf.reshape(bs, num_cams, num_rays)
        if self.return_sample_sdf:
            sample_sdf = output['field_outputs'][FieldHeadNames.SDF]
            sample_sdf = sample_sdf.reshape(bs, num_cams, num_rays, -1)
        if self.return_sem:
            sem = output['sem'].reshape(bs, num_cams, num_rays, -1)
        # depth_median = output['depth_median'].reshape(bs, num_cams, num_rays)
        weights = output['weights'].reshape(bs, num_cams, num_rays, -1, 1)
        num_samples_per_ray = weights.shape[-2]
        ray_samples = output['ray_samples']
        # there is bug here before 8.16, did not divide direction_norm
        ts = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        # ts = ts.reshape(bs, num_cams, num_rays, num_samples_per_ray, 1)
        ts = ts / direction_norm.unsqueeze(1)
        # import pdb; pdb.set_trace()
        deltas = (ray_samples.frustums.ends - ray_samples.frustums.starts)
        # deltas = deltas.reshape(bs, num_cams, num_rays, num_samples_per_ray, 1)
        deltas = deltas / direction_norm.unsqueeze(1)

        if self.return_max_depth:
            eps = torch.finfo(deltas.dtype).eps
            deltas_ = deltas.reshape(bs, num_cams, num_rays, -1)
            weights_ = weights.reshape(bs, num_cams, num_rays, -1).clone()
            weights_[deltas_ < eps] = 0.
            w_per_d = weights_ / deltas_.clamp_min(eps)
            indices = w_per_d.argmax(dim=-1, keepdim=True) # bs, num_cams, num_rays, 1
            max_depth = ts.reshape(bs, num_cams, num_rays, -1)
            max_depth = torch.gather(max_depth, -1, indices).squeeze(-1)


        # www = weights.squeeze(0).cpu()
        # ddd = deltas.reshape(bs, num_cams, num_rays, num_samples_per_ray, 1).squeeze(0).cpu()
        # sss = output['field_outputs'][FieldHeadNames.SDF]
        # sss = sss.reshape(bs, num_cams, num_rays, num_samples_per_ray).squeeze(0).cpu()
        # ttt = ts.reshape(bs, num_cams, num_rays, num_samples_per_ray, 1).squeeze(0).cpu()
        # import matplotlib.pyplot as plt
        # while True:
        #     plt.cla()
        #     fig, ax = plt.subplots(3, 1)
        #     cam = 0; ray=1643
        #     import pdb; pdb.set_trace()
        #     ax[0].plot(ttt[cam, ray], sss[cam, ray])
        #     ax[0].plot(ttt[cam, ray], torch.zeros(256))
        #     ax[1].plot(ttt[cam, ray], www[cam, ray])
        #     ax[2].plot(ttt[cam, ray], www[cam, ray] / (ddd[cam, ray] + 1e-6))
        #     plt.savefig(f'wdelta_w_sdf_t_pro_{cam}_{ray}.png')
        #     print(cam, ray)
        #     import pdb; pdb.set_trace()

        if estimate_flow:
            positions = ray_samples.frustums.get_positions() # B * N * R, S, 3
            positions = positions.reshape(bs, num_cams, num_rays, num_samples_per_ray, 3)
            grids = self.model.field.mapping.meter2grid(positions, True)
            # grids[..., :2] = grids[..., :2] / (self.bev_size - 1)
            # grids[..., 2:] = grids[..., 2:] / (self.z_size - 1)
            prev_sampled_flow = nn.functional.grid_sample(
                curr2prev_flow, 
                grids[..., [2, 1, 0]] * 2 - 1,
                mode='bilinear',
                align_corners=True).permute(0, 2, 3, 4, 1) # bs, N, R, S, 3
            next_sampled_flow = nn.functional.grid_sample(
                curr2next_flow,
                grids[..., [2, 1, 0]] * 2 - 1,
                mode='bilinear',
                align_corners=True).permute(0, 2, 3, 4, 1)
            prev_warp = positions + prev_sampled_flow # B, N, R, S, 3
            next_warp = positions + next_sampled_flow
            
        if global_iter is not None and \
            global_iter % self.print_freq == 0 and \
            ((not dist.is_initialized()) or dist.get_rank() == 0):
            curr_s = output['inv_s']
            logger.info(f'global iter {global_iter} s: {curr_s}, deltas=0: {(deltas == 0).sum().item()}')
            writer.add_scalar('inv_s', curr_s, global_iter)
        
        weights_for_cams = chunk_cams(weights, num_cams)
        ts_for_cams = chunk_cams(ts, num_cams)
        deltas_for_cams = chunk_cams(deltas, num_cams)
        ray_idx_for_cams = [
            torch.arange(num_rays, device=rgb.device).unsqueeze(-1).repeat(1, num_samples_per_ray).flatten()] * num_cams
        eik_grad = output['eik_grad']
        if estimate_flow:
            prev_warp_for_cams = chunk_cams(prev_warp, num_cams)
            next_warp_for_cams = chunk_cams(next_warp, num_cams)
        if self.return_sample_sdf:
            sample_sdf_for_cams = chunk_cams(sample_sdf, num_cams)

        if self.two_split and self.img2lidar.two_split:
            depth = depth[:, :(num_cams//2), :]
            # depth_median = depth_median[:, :(num_cams//2), :]
            rgb = rgb[:, (num_cams//2):, ...]
            acc = acc[:, :(num_cams//2), :]
            fars = fars[:, :(num_cams//2), :]
            ray_idx_for_cams = ray_idx_for_cams[:(num_cams // 2)]
            weights_for_cams = weights_for_cams[:(num_cams // 2)]
            ts_for_cams = ts_for_cams[:(num_cams // 2)]
            deltas_for_cams = deltas_for_cams[:(num_cams // 2)]
            if estimate_flow:
                prev_warp_for_cams = prev_warp_for_cams[:(num_cams // 2)]
                next_warp_for_cams = next_warp_for_cams[:(num_cams // 2)]
            if self.return_max_depth:
                max_depth = max_depth[:, :(num_cams//2), :]
            if self.return_sample_sdf:
                sample_sdf_for_cams = sample_sdf_for_cams[:(num_cams // 2)]
            if self.return_sem:
                sem = sem[:, (num_cams//2):, ...]
        
        outputs = {
            'ms_depths': [depth],
            # 'ms_depths_median': [depth_median],
            'ms_colors': [rgb],
            'ms_accs': [acc],
            'ms_fars': [fars],
            'ms_rays': rays,
            'origin': origin,
            'direction': direction,
            'direction_norm': direction_norm,
            'ray_indices': ray_idx_for_cams,
            'weights': weights_for_cams,
            'ts': ts_for_cams,
            'deltas': deltas_for_cams,
            'eik_grad': eik_grad,
            'uniform_sdf': uniform_sdf}
        if estimate_flow:
            outputs.update({
                'prev_warp': prev_warp_for_cams,
                'next_warp': next_warp_for_cams,
                'curr2prev_flow': curr2prev_flow,
                'curr2next_flow': curr2next_flow,
                'prev2curr_flow': prev2curr_flow,
                'next2curr_flow': next2curr_flow,
                'uniform_xyz': uniform_xyz,
                'uniform_sdf_prev': kwargs['sdf_prev'],
                'uniform_sdf_next': kwargs['sdf_next']
            })
        if self.return_max_depth:
            outputs.update({
                'ms_max_depths': [max_depth]
            })
        if self.return_surface_sdf:
            outputs.update({
                'surface_sdf': surface_sdf
            })
        if self.return_second_grad:
            outputs.update({
                'second_grad': output['field_outputs']['second_grad']
            })
        if self.return_sample_sdf:
            outputs.update({
                'sample_sdf': sample_sdf_for_cams
            })
        if self.return_sem:
            outputs.update({'sem': [sem]})
        return outputs
    

def chunk_cams(tensor, num_cams):
    tensor_for_cams = torch.chunk(
        tensor.reshape(num_cams, -1),
        num_cams, dim=0)
    tensor_for_cams = [t.squeeze() for t in tensor_for_cams]
    return tensor_for_cams