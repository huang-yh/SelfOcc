import os
from ..base_head import BaseTaskHead
from .ray_sampler import RaySampler
from .img2lidar import Img2LiDAR
from .bev_nerf import BEVNeRF
from .rendering import custom_rendering
from .estimator import CustomOccGridEstimator
from mmseg.models import HEADS
import nerfacc, torch, collections, math
from mmengine.logging import MMLogger
logger = MMLogger.get_instance('selfocc')

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))
OCC_THRE = 0.0

@HEADS.register_module()
class NeRFAccHead(BaseTaskHead):

    def __init__(
            self,
            # estimator args
            roi_aabb, 
            resolution,
            reso=0.4,
            # rays args
            ray_sample_mode='fixed',    # fixed, cellular
            ray_number=[192, 400],      # 192 * 400
            ray_img_size=[768, 1600],
            ray_upper_crop=0,
            # img2lidar args
            trans_kw='img2lidar',
            # estimator args
            render_step_size=4e-1,
            near_plane=0.0,
            far_plane=1e10,
            cone_angle: float = 0.0,
            alpha_thre: float = 0.0,
            early_stop_eps: float = 1e-4,
            # render args
            render_bkgd='white',

            # bev nerf
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
            # bev_inner=128,
            # bev_outer=32,
            # range_inner=51.2,
            # range_outer=51.2,
            # nonlinear_mode='linear_upscale',
            # z_inner=20,
            # z_outer=10,
            # z_ranges=[-5.0, 3.0, 11.0],

            # mlp decoder 
            embed_dims=128,
            color_dims=0,
            sem_dims=0,
            density_layers=2,
            sh_deg=2,
            sh_act='relu',
            tpv=False,
            two_split=False,

            novel_view=None,
            
            init_cfg=None, 
            **kwargs):
        super().__init__(init_cfg, **kwargs)

        # self.estimator = nerfacc.OccGridEstimator(
        #     roi_aabb=roi_aabb,
        #     resolution=resolution)
        self.estimator = CustomOccGridEstimator(
            roi_aabb=roi_aabb,
            resolution=resolution)
        self.aabb = roi_aabb
        self.resolution = resolution
        self.reso = reso
        
        self.ray_sampler = RaySampler(
            ray_sample_mode=ray_sample_mode,
            ray_number=ray_number,
            ray_img_size=ray_img_size,
            ray_upper_crop=ray_upper_crop)
        
        self.ray_sampler_eval = RaySampler(
            ray_sample_mode='fixed',
            ray_number=ray_number,
            ray_img_size=ray_img_size,
            ray_upper_crop=ray_upper_crop)
        
        self.img2lidar = Img2LiDAR(
            trans_kw=trans_kw,
            novel_view=novel_view)

        self.radiance_field = BEVNeRF(
            # bev_inner,
            # bev_outer,
            # range_inner,
            # range_outer,
            # nonlinear_mode,
            # z_inner,
            # z_outer,
            # z_ranges,
            mapping_args,
            embed_dims,
            color_dims,
            sem_dims,
            density_layers,
            sh_deg,
            sh_act,
            tpv=tpv)

        self.render_step_size = render_step_size
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.cone_angle = cone_angle
        self.alpha_thre = alpha_thre
        self.early_stop_eps = early_stop_eps
        self.two_split = two_split
        self.return_sem = sem_dims > 0
        # if return_sem: assert color_dims > 3

        if render_bkgd == 'white':
            render_bkgd = torch.ones(3)
            self.register_buffer('render_bkgd', render_bkgd, False)
        elif render_bkgd == 'none':
            self.render_bkgd = None
        else:
            self.render_bkgd = render_bkgd

    def prepare(
            self, 
            representation, 
            metas=None, 
            **kwargs):
        # self.model.field.pre_compute_density_color(representation)# , torch.float)
        self.radiance_field.pre_compute_density_color(representation)
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

            origin, direction = self.img2lidar(metas, rays)
            bs, num_cams, num_rays = direction.shape[:3]
            direction_norm = torch.norm(direction, dim=-1, keepdim=True)
            direction = direction / direction_norm
            origin, direction = origin.squeeze(0), direction.squeeze(0)
            origin = origin.unsqueeze(1).repeat(1, direction.shape[1], 1).flatten(0, 1)
            direction = direction.flatten(0, 1)
            direction_norm = direction_norm.flatten(0, 2)

            def occ_eval_fn(x):
                density = self.radiance_field.query_density(x)
                # logger.info(f'mean of density: {density.mean()}')
                return density * self.render_step_size
            self.estimator._update(
                step=0,
                occ_eval_fn=occ_eval_fn,
                occ_thre=OCC_THRE,
                ema_decay=0.)

            if batch > 0:
                output = {
                    'rgb': [],
                    'acc': [],
                    'depth': [],
                    'weights': [],
                    'ts': [],
                    'sem': []
                }
                chunks = direction.shape[0] * 1.0 / batch
                chunks = int(math.ceil(chunks))
                origins = torch.chunk(origin, chunks)
                directions = torch.chunk(direction, chunks)
                direction_norms = torch.chunk(direction_norm, chunks)
                for i_chunk in range(chunks):
                    # torch.cuda.empty_cache()
                    rgb, acc, depth, sem, n_rendering_samples, ray_indices, weights, ts = self.render_image_with_occgrid(
                        Rays(origins=origins[i_chunk].float(), 
                             viewdirs=directions[i_chunk].float()))
                    depth = depth / direction_norms[i_chunk].reshape(-1, 1)

                    output['rgb'].append(rgb)
                    if self.return_sem:
                        output['sem'].append(sem)
                    output['acc'].append(acc)
                    output['depth'].append(depth)
                
                rgb = torch.cat(output['rgb']).reshape(bs, num_cams, num_rays, -1)
                if self.return_sem:
                    sem = torch.cat(output['sem']).reshape(bs, num_cams, num_rays, -1)
                acc = torch.cat(output['acc']).reshape(bs, num_cams, num_rays)
                depth = torch.cat(output['depth']).reshape(bs, num_cams, num_rays)

            else:
                rgb, acc, depth, sem, n_rendering_samples, ray_indices, weights, ts = self.render_image_with_occgrid(
                    Rays(origins=origin.float(), viewdirs=direction.float()))
                
                # turn depth to camera coordinate
                depth = depth / direction_norm.reshape(-1, 1)

                rgb = rgb.reshape(bs, num_cams, num_rays, -1)
                if self.return_sem:
                    sem = sem.reshape(bs, num_cams, num_rays, -1)
                acc = acc.reshape(bs, num_cams, num_rays)
                depth = depth.reshape(bs, num_cams, num_rays)

        outputs = {
            'ms_depths': [depth],
            'ms_colors': [rgb],
            'ms_accs': [acc],
            'ms_rays': rays}
        if self.return_sem:
            outputs.update({'sem': [sem]})
        return outputs


    @torch.cuda.amp.autocast(enabled=False)
    def render_image_with_occgrid(self, rays):

        """Render the pixels of an image."""
        rays_shape = rays.origins.shape
        if len(rays_shape) == 3:
            height, width, _ = rays_shape
            num_rays = height * width
            rays = namedtuple_map(
                lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
            )
        else:
            num_rays, _ = rays_shape

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = chunk_rays.origins[ray_indices]
            t_dirs = chunk_rays.viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = self.radiance_field.query_density(positions)
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = chunk_rays.origins[ray_indices]
            t_dirs = chunk_rays.viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            rgbs, sigmas, sems = self.radiance_field(positions, t_dirs)
            return rgbs, sigmas.squeeze(-1), sems

        results = []
        chunk = torch.iinfo(torch.int32).max
        for i in range(0, num_rays, chunk):
            chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
            ray_indices, t_starts, t_ends = self.estimator.sampling(
                chunk_rays.origins,
                chunk_rays.viewdirs,
                sigma_fn=sigma_fn,
                near_plane=self.near_plane,
                far_plane=self.far_plane,
                render_step_size=self.render_step_size,
                stratified=self.radiance_field.training,
                cone_angle=self.cone_angle,
                alpha_thre=self.alpha_thre,
                early_stop_eps=self.early_stop_eps
            )
            rgb, opacity, depth, sem, extras = custom_rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=chunk_rays.origins.shape[0],
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=self.render_bkgd,
            )
            chunk_results = [rgb, opacity, depth, sem, len(t_starts), ray_indices, extras["weights"], extras["ts"]]
            results.append(chunk_results)
        colors, opacities, depths, sems, n_rendering_samples, ray_indices, weights, ts = [
            torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
            for r in zip(*results)
        ]
        return (
            colors.view((*rays_shape[:-1], -1)),
            opacities.view((*rays_shape[:-1], -1)),
            depths.view((*rays_shape[:-1], -1)),
            sems.view((*rays_shape[:-1], -1)),
            sum(n_rendering_samples),
            ray_indices,
            weights,
            ts
        )
    
    # @torch.cuda.amp.autocast(enabled=False)
    # def forward_occ(
    #         self, 
    #         representation, 
    #         metas=None,
    #         **kwargs):
        
    #     rays = self.ray_sampler() # HW, 2

    #     origin, direction = self.img2lidar(metas, rays)
    #     num_cams, num_rays = direction.shape[1:3]
    #     direction_norm = torch.norm(direction, dim=-1, keepdim=True)
    #     direction = direction / direction_norm
    #     origin, direction = origin.squeeze(0), direction.squeeze(0)
    #     origin = origin.unsqueeze(1).repeat(1, direction.shape[1], 1).flatten(0, 1)
    #     direction = direction.flatten(0, 1)

    #     # update occupancy grid
    #     self.radiance_field.pre_compute_density_color(representation)
    #     def occ_eval_fn(x):
    #         density = self.radiance_field.query_density(x)
    #         # logger.info(f'mean of density: {density.mean()}')
    #         return density * self.render_step_size
    #     self.estimator._update(
    #         step=0,
    #         occ_eval_fn=occ_eval_fn,
    #         occ_thre=1e-2,
    #         ema_decay=0.)
    #     # logger.info('number of nonzero grid: ' + str(self.estimator.binaries.sum()))
    #     if self.estimator.binaries.sum() == 0:
    #         logger.info(representation.mean())
    #         logger.info(self.radiance_field.density_net[-1].weight.mean())
        
    #     return {}

    @torch.cuda.amp.autocast(enabled=False)
    def forward_occ(
            self, 
            representation, 
            metas=None,
            **kwargs):
        if isinstance(representation, (tuple, list)):
            device = representation[0].device
        else:
            device = representation.device
        aabb = kwargs['aabb'] if 'aabb' in kwargs else self.aabb
        reso = kwargs['resolution'] if 'resolution' in kwargs else self.reso
        # update occupancy grid
        self.radiance_field.pre_compute_density_color(representation)
        
        if self.return_sem:
            sigma, sem, sem_logits, _ = self.get_uniform_sdf(aabb, reso, device=device)
            return {'sigma': sigma, 'rep': representation, 'sem': sem, 'logits': sem_logits}
        sigma, _ = self.get_uniform_sdf(aabb, reso, device=device)
        return {'sigma': sigma, 'rep': representation}

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

        sigma, sem = self.radiance_field.forward_geo(xyzs)
        
        if self.return_sem:
            sigma = sigma.reshape(H, W, D)
            sem_logits = sem.reshape(H, W, D, -1)
            sem = torch.argmax(sem, dim=-1).reshape(H, W, D)
            return sigma, sem, sem_logits, xyzs.reshape(H, W, D, -1)
        else:
            sigma = sigma.reshape(H, W, D)
            return sigma, xyzs.reshape(H, W, D, -1)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
            self, 
            representation, 
            metas=None,
            **kwargs):

        if os.environ.get('eval', 'false') == 'true':
            ray_sampler = self.ray_sampler_eval
        else:
            ray_sampler = self.ray_sampler 

        rays = ray_sampler() # HW, 2

        origin, direction = self.img2lidar(metas, rays)
        num_cams, num_rays = direction.shape[1:3]
        direction_norm = torch.norm(direction, dim=-1, keepdim=True)
        direction = direction / direction_norm
        origin, direction = origin.squeeze(0), direction.squeeze(0)
        origin = origin.unsqueeze(1).repeat(1, direction.shape[1], 1).flatten(0, 1)
        direction = direction.flatten(0, 1)

        # update occupancy grid
        self.radiance_field.pre_compute_density_color(representation)
        def occ_eval_fn(x):
            density = self.radiance_field.query_density(x)
            # logger.info(f'mean of density: {density.mean()}')
            return density * self.render_step_size
        self.estimator._update(
            step=0,
            occ_eval_fn=occ_eval_fn,
            occ_thre=OCC_THRE,
            ema_decay=0.)
        # logger.info('number of nonzero grid: ' + str(self.estimator.binaries.sum()))
        if self.estimator.binaries.sum() == 0:
            logger.info(representation.mean())
            logger.info(self.radiance_field.density_net[-1].weight.mean())

        # render
        rgb, acc, depth, sem, n_rendering_samples, ray_indices, weights, ts = self.render_image_with_occgrid(
            Rays(origins=origin.float(), viewdirs=direction.float()))
        
        # turn depth to camera coordinate
        depth = depth / direction_norm.reshape(-1, 1)

        weights_for_cams, ts_for_cams, ray_idx_for_cams = [], [], []
        for i in range(num_cams):
            curr_cam = torch.logical_and(ray_indices >= (i * num_rays), ray_indices < (i * num_rays + num_rays))
            ray_idx_for_cam = ray_indices[curr_cam] - i * num_rays
            ray_idx_for_cams.append(ray_idx_for_cam)
            weights_for_cams.append(weights[curr_cam])
            ts_for_curr_cam = ts[curr_cam]
            direction_norm_for_cam = direction_norm[0, i, ray_idx_for_cam, 0]
            ts_for_curr_cam = ts_for_curr_cam / direction_norm_for_cam
            ts_for_cams.append(ts_for_curr_cam)
        
        if self.two_split and self.img2lidar.two_split:
            depth = depth.reshape(1, -1, rays.shape[0])[:, :(num_cams//2), :]
            rgb = rgb.reshape(1, -1, rays.shape[0], 3)[:, (num_cams//2):, ...]
            sem = sem.reshape(1, num_cams, rays.shape[0], -1)[:, (num_cams//2):, ...]
            acc = acc.reshape(1, -1, rays.shape[0])[:, :(num_cams//2), :]
            ray_idx_for_cams = ray_idx_for_cams[:(num_cams // 2)]
            weights_for_cams = weights_for_cams[:(num_cams // 2)]
            ts_for_cams = ts_for_cams[:(num_cams // 2)]
        else:
            depth = depth.reshape(1, -1, rays.shape[0])
            rgb = rgb.reshape(1, -1, rays.shape[0], 3)
            sem = sem.reshape(1, num_cams, rays.shape[0], -1)
            acc = acc.reshape(1, -1, rays.shape[0])
        
        outputs = {
                'ms_depths': [depth],
                'ms_colors': [rgb],
                'sem': [sem],
                'ms_accs': [acc],
                'ms_rays': rays,
                'ray_indices': ray_idx_for_cams,
                'weights': weights_for_cams,
                'ts': ts_for_cams
        }
        return outputs