from nerfacc import OccGridEstimator
import torch
from torch import Tensor
from typing import Callable, Optional, Tuple
from nerfacc.grid import traverse_grids


class CustomOccGridEstimator(OccGridEstimator):

    @torch.no_grad()
    def sampling(
        self,
        # rays
        rays_o: Tensor,  # [n_rays, 3]
        rays_d: Tensor,  # [n_rays, 3]
        # sigma/alpha function for skipping invisible space
        sigma_fn: Optional[Callable] = None,
        alpha_fn: Optional[Callable] = None,
        near_plane: float = 0.0,
        far_plane: float = 1e10,
        t_min: Optional[Tensor] = None,  # [n_rays]
        t_max: Optional[Tensor] = None,  # [n_rays]
        # rendering options
        render_step_size: float = 1e-3,
        early_stop_eps: float = 1e-4,
        alpha_thre: float = 0.0,
        stratified: bool = False,
        cone_angle: float = 0.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sampling with spatial skipping.

        Note:
            This function is not differentiable to any inputs.

        Args:
            rays_o: Ray origins of shape (n_rays, 3).
            rays_d: Normalized ray directions of shape (n_rays, 3).
            sigma_fn: Optional. If provided, the marching will skip the invisible space
                by evaluating the density along the ray with `sigma_fn`. It should be a
                function that takes in samples {t_starts (N,), t_ends (N,),
                ray indices (N,)} and returns the post-activation density values (N,).
                You should only provide either `sigma_fn` or `alpha_fn`.
            alpha_fn: Optional. If provided, the marching will skip the invisible space
                by evaluating the density along the ray with `alpha_fn`. It should be a
                function that takes in samples {t_starts (N,), t_ends (N,),
                ray indices (N,)} and returns the post-activation opacity values (N,).
                You should only provide either `sigma_fn` or `alpha_fn`.
            near_plane: Optional. Near plane distance. Default: 0.0.
            far_plane: Optional. Far plane distance. Default: 1e10.
            t_min: Optional. Per-ray minimum distance. Tensor with shape (n_rays).
                If profided, the marching will start from maximum of t_min and near_plane.
            t_max: Optional. Per-ray maximum distance. Tensor with shape (n_rays).
                If profided, the marching will stop by minimum of t_max and far_plane.
            render_step_size: Step size for marching. Default: 1e-3.
            early_stop_eps: Early stop threshold for skipping invisible space. Default: 1e-4.
            alpha_thre: Alpha threshold for skipping empty space. Default: 0.0.
            stratified: Whether to use stratified sampling. Default: False.
            cone_angle: Cone angle for linearly-increased step size. 0. means
                constant step size. Default: 0.0.

        Returns:
            A tuple of {LongTensor, Tensor, Tensor}:

            - **ray_indices**: Ray index of each sample. IntTensor with shape (n_samples).
            - **t_starts**: Per-sample start distance. Tensor with shape (n_samples,).
            - **t_ends**: Per-sample end distance. Tensor with shape (n_samples,).

        Examples:

        .. code-block:: python

            >>> ray_indices, t_starts, t_ends = grid.sampling(
            >>>     rays_o, rays_d, render_step_size=1e-3)
            >>> t_mid = (t_starts + t_ends) / 2.0
            >>> sample_locs = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

        """

        near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
        far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

        if t_min is not None:
            near_planes = torch.clamp(near_planes, min=t_min)
        if t_max is not None:
            far_planes = torch.clamp(far_planes, max=t_max)

        if stratified:
            near_planes += torch.rand_like(near_planes) * render_step_size
        intervals, samples, _ = traverse_grids(
            rays_o,
            rays_d,
            self.binaries,
            self.aabbs,
            near_planes=near_planes,
            far_planes=far_planes,
            step_size=render_step_size,
            cone_angle=cone_angle,
        )
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = samples.ray_indices
        packed_info = samples.packed_info

        # # skip invisible space
        # if (alpha_thre > 0.0 or early_stop_eps > 0.0) and (
        #     sigma_fn is not None or alpha_fn is not None
        # ):
        #     alpha_thre = min(alpha_thre, self.occs.mean().item())

        #     # Compute visibility of the samples, and filter out invisible samples
        #     if sigma_fn is not None:
        #         if t_starts.shape[0] != 0:
        #             sigmas = sigma_fn(t_starts, t_ends, ray_indices)
        #         else:
        #             sigmas = torch.empty((0,), device=t_starts.device)
        #         assert (
        #             sigmas.shape == t_starts.shape
        #         ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
        #         masks = render_visibility_from_density(
        #             t_starts=t_starts,
        #             t_ends=t_ends,
        #             sigmas=sigmas,
        #             packed_info=packed_info,
        #             early_stop_eps=early_stop_eps,
        #             alpha_thre=alpha_thre,
        #         )
        #     elif alpha_fn is not None:
        #         if t_starts.shape[0] != 0:
        #             alphas = alpha_fn(t_starts, t_ends, ray_indices)
        #         else:
        #             alphas = torch.empty((0,), device=t_starts.device)
        #         assert (
        #             alphas.shape == t_starts.shape
        #         ), "alphas must have shape of (N,)! Got {}".format(alphas.shape)
        #         masks = render_visibility_from_alpha(
        #             alphas=alphas,
        #             packed_info=packed_info,
        #             early_stop_eps=early_stop_eps,
        #             alpha_thre=alpha_thre,
        #         )
        #     ray_indices, t_starts, t_ends = (
        #         ray_indices[masks],
        #         t_starts[masks],
        #         t_ends[masks],
        #     )
        return ray_indices, t_starts, t_ends
