try:
    from atlas_gsplat.project_gaussians_2d import project_gaussians_2d
    from atlas_gsplat.rasterize_sum import rasterize_gaussians_sum, rasterize_gaussians_sum_general
except ImportError as _kernel_import_error:
    raise ImportError(
        "ATLAS needs the CUDA kernel package `atlas-gsplat`, which is not installed.\n"
        "  pip install atlas-gsplat   (needs a CUDA toolkit matching your torch build;\n"
        "  run `atlas doctor` to check your environment)\n"
        f"original import error: {_kernel_import_error}"
    ) from _kernel_import_error


def _require_atlas_gsplat_kernel():
    """Fail clearly *now* if the atlas-gsplat CUDA extension can't load, instead of a
    cryptic ``'NoneType' object has no attribute ...`` deep in the forward pass.

    Cheap: uses ``find_spec`` + toolkit detection, so it does not trigger the (slow)
    JIT build itself — that still happens lazily on the first real kernel call.
    """
    import importlib.util

    # A prebuilt C extension (`pip install atlas-gsplat`) works without a toolkit.
    if importlib.util.find_spec("atlas_gsplat.csrc") is not None:
        return
    try:
        from atlas_gsplat.cuda._detect import cuda_toolkit_available
    except Exception:
        return  # older atlas-gsplat without _detect — let the first kernel call decide
    if not cuda_toolkit_available():
        raise ImportError(
            "atlas-gsplat is installed but its CUDA kernel cannot load: there is no "
            "prebuilt extension and no CUDA toolkit (nvcc) was found to compile it. "
            "Install a CUDA toolkit (the kernel then builds on first use), or reinstall "
            "atlas-gsplat from source on a CUDA machine. Run `atlas doctor` for details."
        )


_require_atlas_gsplat_kernel()
from .utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from .quantize import *
from .optimizer import Adan
from torch.utils.data import DataLoader, Dataset


class GaussianImage_Cholesky_EA(nn.Module):
    def __init__(self, loss_type="L2", **kwargs): #L2 SSIM Fusion1 Fusion2
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.max_num_points = kwargs.get("max_num_points", self.init_num_points)
        self.grow_interval = kwargs.get("grow_interval", 1000)
        self.grow_stop_iter = kwargs.get("grow_stop_iter", 5000)
        self.d3_enabled = self.max_num_points > self.init_num_points
        self.prune_interval = kwargs.get("prune_interval", 100)
        self.caf = kwargs.get("caf", True)
        self.delta_coding = kwargs.get("delta_coding", False)
        self.qat_enabled = False  # Toggled at QAT transition
        self.H, self.W, self.rank, self.C = kwargs["H"], kwargs["W"], kwargs["rank"], kwargs["C"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) 
        #torch.seed(1234)
        torch.cuda.synchronize()
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) # MB
        self.device = kwargs["device"]
        self.image = kwargs["GT"]#.to(torch.float32) # (1, C, H, W)
        
        self.endmember_lr = kwargs.get("endmember_lr", 0)
        self.sam_weight = kwargs.get("sam_weight", 0.0)
        E_tensor = torch.tensor(kwargs["E"]).to(self.device)
        if self.endmember_lr > 0:
            # Softplus parameterization: store raw, apply softplus to get non-negative E
            raw = torch.log(torch.exp(E_tensor) - 1)
            raw = torch.where(E_tensor > 20, E_tensor, raw)  # numerical stability
            raw = torch.where(E_tensor <= 1e-6, torch.tensor(-10.0, device=self.device), raw)
            self._endmember_raw = nn.Parameter(raw)
        else:
            self.endmember = nn.Parameter(E_tensor, requires_grad=False)
        
        torch.cuda.synchronize()
        E_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2) # MB
        print(f"E0 GPU memory usage: {E_gpu_memory - gpu_memory} MB")

        A0 = kwargs.get("A0", None)
        if A0 is not None:
            self._xyz, self._features_dc = self._entropy_init(A0, kwargs["E"])
        else:
            self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
            self._features_dc = nn.Parameter(0.5 * torch.rand(self.init_num_points, self.rank))
        self._cholesky = nn.Parameter(torch.zeros(self.init_num_points, 3))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self.coef = nn.Parameter(torch.tensor(0.0))

        self.last_size = (self.H, self.W)
        self.quantize = kwargs["quantize"]
        self.register_buffer('background', torch.ones(self.rank))
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        if self.caf:
            low_pass = min(self.H * self.W / (9 * math.pi * self.init_num_points), 300)
            self.register_buffer('cholesky_bound',
                torch.tensor([low_pass, 0, low_pass]).view(1, 3).repeat(self.init_num_points, 1))
        else:
            self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        
        self.abundance_bits = kwargs.get("abundance_bits", 6)
        self.cholesky_bits = kwargs.get("cholesky_bits", 10)
        self.endmember_quantizer = FakeQuantizationHalf.apply
        self.xyz_quantizer = FakeQuantizationHalf.apply  # FP16: position quantization hurts HSI quality
        self.features_dc_quantizer = UniformQuantizer(signed=False, bits=self.abundance_bits, learned=True, num_channels=self.rank)
        self.cholesky_quantizer = UniformQuantizer(signed=True, bits=self.cholesky_bits, learned=True, num_channels=3)

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self._gaussian_params(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self._gaussian_params(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.5)

        # Separate optimizer for learnable endmember (small LR, non-negative via softplus)
        if self.endmember_lr > 0:
            self.endmember_optimizer = torch.optim.Adam([self._endmember_raw], lr=self.endmember_lr)
            self.endmember_scheduler = torch.optim.lr_scheduler.StepLR(
                self.endmember_optimizer, step_size=3000, gamma=0.5)

    def _gaussian_params(self):
        """Gaussian params only — excludes endmember (which has its own optimizer)."""
        return [p for n, p in self.named_parameters() if n not in ('endmember', '_endmember_raw')]

    def _entropy_init(self, A0_np, E_np):
        """Initialize positions at high-error pixels, biased toward spectral edges.

        Blends NMF reconstruction error with Sobel gradient magnitude of abundance maps
        to place more Gaussians at material boundaries and spectral transitions.
        """
        I_recon = A0_np @ E_np  # [H*W, C]
        I_gt = self.image.squeeze(0).permute(1, 2, 0).reshape(-1, self.C).cpu().float().numpy()
        errors = np.sum(np.abs(I_gt - I_recon), axis=1)  # [H*W]

        # Compute Sobel gradient magnitude on abundance maps
        from scipy.ndimage import sobel
        A0_2d = A0_np.reshape(self.H, self.W, -1)  # [H, W, rank]
        grad_mag = np.zeros((self.H, self.W), dtype=np.float32)
        for ch in range(A0_2d.shape[2]):
            gx = sobel(A0_2d[:, :, ch], axis=1)
            gy = sobel(A0_2d[:, :, ch], axis=0)
            grad_mag += np.sqrt(gx**2 + gy**2)
        grad_flat = grad_mag.flatten()  # [H*W]

        # Normalize both to [0, 1] and blend
        err_norm = errors / (errors.max() + 1e-8)
        grad_norm = grad_flat / (grad_flat.max() + 1e-8)
        blended = 0.5 * err_norm + 0.5 * grad_norm

        errors_t = torch.from_numpy(blended).float() + 1e-8
        replace = self.init_num_points > len(errors_t)
        indices = torch.multinomial(errors_t, self.init_num_points, replacement=replace)

        ys = (indices // self.W).float()
        xs = (indices % self.W).float()
        norm_x = torch.clamp((xs / (self.W - 1)) * 2 - 1, -0.999, 0.999)
        norm_y = torch.clamp((ys / (self.H - 1)) * 2 - 1, -0.999, 0.999)
        xyz = nn.Parameter(torch.atanh(torch.stack([norm_x, norm_y], dim=-1)))

        A0_t = torch.from_numpy(A0_np).float()
        features = nn.Parameter(A0_t[indices])
        return xyz, features

    def _init_data(self):
        self.cholesky_quantizer._init_data(self._cholesky)
        self.features_dc_quantizer._init_data(self._features_dc)

    def enable_quantization(self, qat_ramp_iters=0):
        """Enable real quantization (QAT stage). Call _init_data to set scale/beta from current params.

        Args:
            qat_ramp_iters: If > 0, progressively tighten quantization over this many iters.
                           Scale starts at 2x (coarser) and linearly decays to 1x (target).
        """
        self.qat_enabled = True
        self.qat_ramp_iters = qat_ramp_iters
        self.qat_ramp_step = 0
        self.cholesky_quantizer._init_data(self._cholesky)
        self.features_dc_quantizer._init_data(self._features_dc)

    @torch.no_grad()
    def prune_degenerate(self):
        """Check for degenerate Gaussians (non-PSD after quantization) and remove them."""
        # Check quantized cholesky elements
        if self.qat_enabled:
            cholesky_q, _, _ = self.cholesky_quantizer(self._cholesky)
        else:
            cholesky_q = self._cholesky
        cholesky_elements = cholesky_q + self.cholesky_bound
        keep_mask = (cholesky_elements[:, 0] > 0) & (cholesky_elements[:, 2] > 0)

        num_pruned = (~keep_mask).sum().item()
        if num_pruned == 0:
            return 0
        if keep_mask.sum().item() == 0:
            print("[PRUNE] WARNING: All Gaussians degenerate! Skipping.")
            return 0

        # Save old optimizer state by param index (gaussian params only)
        old_N = self._xyz.shape[0]
        old_params = self._gaussian_params()
        old_states = {i: self.optimizer.state[p] for i, p in enumerate(old_params) if p in self.optimizer.state}
        old_param_groups = self.optimizer.param_groups
        last_epoch = self.scheduler.last_epoch

        # Prune parameters
        self._xyz = nn.Parameter(self._xyz.data[keep_mask])
        self._cholesky = nn.Parameter(self._cholesky.data[keep_mask])
        self._features_dc = nn.Parameter(self._features_dc.data[keep_mask])
        self._opacity = self._opacity[keep_mask]
        if self.caf:
            self.cholesky_bound = self.cholesky_bound[keep_mask]

        # Rebuild optimizer with pruned params (exclude endmember)
        lr = old_param_groups[0]['lr']
        self.optimizer = Adan(self._gaussian_params(), lr=lr)
        for key in ['betas', 'eps', 'weight_decay', 'max_grad_norm', 'no_prox']:
            if key in old_param_groups[0]:
                self.optimizer.param_groups[0][key] = old_param_groups[0][key]

        # Transplant momentum (sliced by keep_mask for per-Gaussian tensors)
        for i, new_p in enumerate(self._gaussian_params()):
            if i not in old_states:
                continue
            s = old_states[i]
            new_s = {'step': s.get('step', 0)}
            for state_key in ['exp_avg', 'exp_avg_sq', 'exp_avg_diff', 'neg_pre_grad']:
                if state_key in s:
                    old_tensor = s[state_key]
                    if old_tensor.dim() >= 1 and old_tensor.shape[0] == old_N:
                        new_s[state_key] = old_tensor[keep_mask]
                    else:
                        new_s[state_key] = old_tensor
            self.optimizer.state[new_p] = new_s

        # Recreate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.5)
        self.scheduler.last_epoch = last_epoch

        return num_pruned

    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        return self._opacity

    @property
    def endmember_activated(self):
        if self.endmember_lr > 0:
            return F.softplus(self._endmember_raw)
        return self.endmember

    @property
    def get_cholesky_elements(self):
        return self._cholesky+self.cholesky_bound

    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(self.get_xyz, self.get_cholesky_elements, self.H, self.W, self.tile_bounds)

        # UNIFIED KERNEL: Process all spectral channels in one pass
        out_img = rasterize_gaussians_sum_general(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            self.get_features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, return_alpha=False
        )

        out_img = torch.clamp(out_img, 0, 1) #(H, W, rank)
        out_img = out_img.view(self.H * self.W, self.rank).contiguous() #(H * W, rank)
        return {"render": out_img}

    def train_iter(self):
        render_pkg = self.forward()
        A = render_pkg["render"] # (H * W, rank)
        flatimage = (A @ self.endmember_activated) # image: (H * W, C)
        #image_cube = flatimage.view(self.H, self.W, self.C)
        image = flatimage.view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous() #[1, C, H, W]
        
        # update abundance
        loss =   loss_fn(image, self.image, self.loss_type, lambda_value=0.7) 
        loss.backward()
        '''0.15 * loss_fn(A, self.abundance) + 0.85 *0.05 * loss_fn(A, self.abundance) + 0.95 *
        # update endmember: multiplication rule
        with torch.no_grad():
            A_new = self.forward()["render"]
            image_new = (A_new @ self.endmember)
            self.endmember *=  (A_new.T @ self.image.view(-1, self.C)) / (A_new.T @ image_new)
        '''
        with torch.no_grad():
            I = (A @ self.endmember * self.coef_endmember).view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()
            mse_loss = F.mse_loss(I, self.image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
            
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.scheduler.step()
        return loss, psnr
    
    def _compute_sort_order(self, means):
        """Compute Morton Z-order sort for spatial delta coding."""
        if not self.delta_coding or self.training:
            return None
        # Morton order: interleave bits of quantized x,y coordinates
        xy = means.detach().cpu().numpy()  # [N, 2], range [-1, 1]
        # Quantize to 16-bit grid
        qx = ((xy[:, 0] + 1) * 32767.5).astype(np.uint32)
        qy = ((xy[:, 1] + 1) * 32767.5).astype(np.uint32)
        # Simple Morton code: interleave bits (use multiplication trick for speed)
        morton = np.zeros(len(qx), dtype=np.uint64)
        for bit in range(16):
            morton |= ((qx >> bit) & 1).astype(np.uint64) << (2 * bit)
            morton |= ((qy >> bit) & 1).astype(np.uint64) << (2 * bit + 1)
        return np.argsort(morton)

    def forward_quantize(self):
        # quantize plan1: "GaussianImage"
        N = self._xyz.shape[0]
        l_vqm, m_bit = 0, 16*N*2
        means = torch.tanh(FakeQuantizationHalf.apply(self._xyz))

        # Compute spatial sort order for delta coding (eval-time only)
        sort_order = self._compute_sort_order(means)

        # Quantization controlled by qat_enabled flag (set via enable_quantization())
        if self.qat_enabled:
            # Progressive quantization: temporarily widen scale during ramp
            if self.qat_ramp_iters > 0 and self.qat_ramp_step < self.qat_ramp_iters and self.training:
                progress = self.qat_ramp_step / self.qat_ramp_iters
                scale_mult = 2.0 - progress  # 2.0 -> 1.0
                self.qat_ramp_step += 1
                # Temporarily scale up (coarser quant), then restore
                with torch.no_grad():
                    self.cholesky_quantizer.scale.data *= scale_mult
                    self.features_dc_quantizer.scale.data *= scale_mult
                cholesky_elements, l_vqs, s_bit = self.cholesky_quantizer(self._cholesky)
                features, l_vqc, c_bit = self.features_dc_quantizer(self.get_features)
                with torch.no_grad():
                    self.cholesky_quantizer.scale.data /= scale_mult
                    self.features_dc_quantizer.scale.data /= scale_mult
            else:
                cholesky_elements, l_vqs, s_bit = self.cholesky_quantizer(self._cholesky, sort_order=sort_order)
                features, l_vqc, c_bit = self.features_dc_quantizer(self.get_features, sort_order=sort_order)
        else:
            cholesky_elements, l_vqs, s_bit = self._cholesky, 0, 32*N*3
            features, l_vqc, c_bit = self.get_features, 0, 32*N*self.rank
        cholesky_elements = cholesky_elements + self.cholesky_bound
        l_vqr, r_bit = 0, 0

        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d(means,
                cholesky_elements, self.H, self.W, self.tile_bounds)

        # UNIFIED KERNEL: Process all spectral channels in one pass
        out_img = rasterize_gaussians_sum_general(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            features, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, return_alpha=False
        )

        out_img = torch.clamp(out_img, 0, 1) #(H, W, rank)
        out_img = out_img.view(self.H * self.W, self.rank).contiguous() * torch.exp(self.coef)#(H * W, rank)

        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
        return {"render": out_img, "vq_loss": vq_loss, "unit_bit":[m_bit, s_bit, r_bit, c_bit]}

    def train_iter_quantize(self, current_iter=None):
        render_pkg = self.forward_quantize()
        A = render_pkg["render"] # (H * W, rank)
        E = FakeQuantizationHalf.apply(self.endmember_activated)
        flatimage = A @ E # image: (H * W, C)
        image = flatimage.view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous() #[1, C, H, W]

        # update abundance
        loss =  loss_fn(image, self.image, self.loss_type, lambda_value=0.7) + 0.05 * render_pkg['vq_loss']
        if self.sam_weight > 0:
            loss = loss + self.sam_weight * sam_loss_fn(image, self.image)
        loss.backward()
        #0.05 * loss_fn(A, self.abundance) + 0.95 *
        with torch.no_grad():
            I = (A @ self.endmember_activated).view(-1, self.H, self.W, self.C).permute(0, 3, 1, 2).contiguous()
            mse_per_channel = F.mse_loss(I, self.image, reduction='none')
            mse_per_channel_avg = mse_per_channel.mean(dim=(0, 2, 3))
            psnr_per_channel = 10 * torch.log10(1.0 / mse_per_channel_avg)
            psnr = psnr_per_channel.mean().item()

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        # Learnable endmember update (softplus ensures non-negativity)
        if self.endmember_lr > 0:
            self.endmember_optimizer.step()
            self.endmember_optimizer.zero_grad(set_to_none=True)
            self.endmember_scheduler.step()

        # D3 densification
        if current_iter is not None and self.d3_enabled:
            self.densify_step(current_iter)

        # Pruning during QAT
        if current_iter is not None and self.qat_enabled and self.prune_interval > 0:
            if current_iter % self.prune_interval == 0:
                num_pruned = self.prune_degenerate()
                if num_pruned > 0:
                    print(f"[PRUNE] Iter {current_iter}: pruned {num_pruned}, remaining {self._xyz.shape[0]}")

        return loss, psnr

    # ==================== D3 Densification ====================

    @torch.no_grad()
    def compute_pixel_errors(self):
        """Per-pixel spectral L1 error for D3 placement."""
        out = self.forward_quantize()
        A = out["render"]  # [H*W, rank]
        E = FakeQuantizationHalf.apply(self.endmember_activated.to(torch.float32))
        I_pred = (A @ E).view(self.H, self.W, self.C)
        I_gt = self.image.squeeze(0).permute(1, 2, 0).float()  # [H, W, C]
        errors = torch.abs(I_gt - I_pred).sum(dim=-1)  # [H, W]
        return errors, A.view(self.H, self.W, -1)

    @torch.no_grad()
    def add_gaussians(self, num_new):
        """Add num_new Gaussians at highest-error pixel locations."""
        errors, A_map = self.compute_pixel_errors()  # [H,W], [H,W,rank]

        # Find top-k error pixels
        flat_errors = errors.view(-1)
        num_new = min(num_new, self.max_num_points - self._xyz.shape[0])
        if num_new <= 0:
            return
        topk_vals, topk_indices = torch.topk(flat_errors, num_new)

        # Convert flat indices to (y, x) coordinates
        ys = (topk_indices // self.W).float()
        xs = (topk_indices % self.W).float()

        # Normalize to [-1, 1] range then atanh for _xyz space
        norm_x = (xs / (self.W - 1)) * 2 - 1
        norm_y = (ys / (self.H - 1)) * 2 - 1
        # Clamp to avoid atanh(±1) = ±inf
        norm_x = torch.clamp(norm_x, -0.999, 0.999)
        norm_y = torch.clamp(norm_y, -0.999, 0.999)
        new_xyz = torch.stack([norm_x, norm_y], dim=-1)  # [num_new, 2]
        new_xyz = torch.atanh(new_xyz)

        # Warm-start abundances from rendered A map at those pixels
        new_features = A_map[ys.long(), xs.long(), :]  # [num_new, rank]
        # Invert sigmoid to get raw parameter space (features go through sigmoid in forward)
        new_features = torch.clamp(new_features, 0.001, 0.999)

        # Cholesky init as zeros (cholesky_bound adds [0.5, 0, 0.5])
        new_cholesky = torch.zeros(num_new, 3, device=self.device)

        # Extend parameters
        old_num = self._xyz.shape[0]
        self._xyz = nn.Parameter(torch.cat([self._xyz.data, new_xyz.to(self._xyz.dtype)], dim=0))
        self._cholesky = nn.Parameter(torch.cat([self._cholesky.data, new_cholesky.to(self._cholesky.dtype)], dim=0))
        self._features_dc = nn.Parameter(torch.cat([self._features_dc.data, new_features.to(self._features_dc.dtype)], dim=0))

        # Extend opacity buffer
        new_opacity = torch.ones((num_new, 1), device=self.device)
        self._opacity = torch.cat([self._opacity, new_opacity], dim=0)

        if self.caf:
            new_n = self._xyz.shape[0]
            new_low_pass = min(self.H * self.W / (9 * math.pi * new_n), 300)
            new_bound = torch.tensor([new_low_pass, 0, new_low_pass], device=self.device).view(1, 3).repeat(num_new, 1)
            self.cholesky_bound = torch.cat([self.cholesky_bound, new_bound], dim=0)

        return old_num

    def rebuild_optimizer(self, old_num_points):
        """Rebuild Adan optimizer, transplanting momentum state for existing points."""
        new_num = self._xyz.shape[0]
        num_new = new_num - old_num_points

        # Save old optimizer state
        old_state = self.optimizer.state
        old_param_groups = self.optimizer.param_groups
        lr = old_param_groups[0]['lr']

        # Save scheduler state
        last_epoch = self.scheduler.last_epoch

        # Create new optimizer (exclude endmember — it has its own optimizer)
        self.optimizer = Adan(self._gaussian_params(), lr=lr)
        # Copy param group settings (betas, eps, weight_decay, etc.) from old
        for key in ['betas', 'eps', 'weight_decay', 'max_grad_norm', 'no_prox']:
            if key in old_param_groups[0]:
                self.optimizer.param_groups[0][key] = old_param_groups[0][key]

        # Transplant momentum states (gaussian params only)
        for old_p, new_p in zip(list(old_state.keys()), self._gaussian_params()):
            if old_p not in old_state:
                continue
            s = old_state[old_p]
            new_s = {}
            new_s['step'] = s.get('step', 0)
            for state_key in ['exp_avg', 'exp_avg_sq', 'exp_avg_diff', 'neg_pre_grad']:
                if state_key in s:
                    old_tensor = s[state_key]
                    if old_tensor.dim() >= 1 and old_tensor.shape[0] == old_num_points and new_p.dim() >= 1 and new_p.shape[0] == new_num:
                        # This is a per-point parameter that grew
                        padding = torch.zeros(num_new, *old_tensor.shape[1:], device=old_tensor.device, dtype=old_tensor.dtype)
                        new_s[state_key] = torch.cat([old_tensor, padding], dim=0)
                    else:
                        new_s[state_key] = old_tensor
            self.optimizer.state[new_p] = new_s

        # Recreate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.5)
        self.scheduler.last_epoch = last_epoch

    def densify_step(self, current_iter):
        """D3 orchestrator: check timing and grow if needed."""
        if current_iter % self.grow_interval != 0:
            return
        if current_iter > self.grow_stop_iter:
            return
        current_n = self._xyz.shape[0]
        if current_n >= self.max_num_points:
            return

        # Compute how many to add this step
        remaining_grow_steps = (self.grow_stop_iter - current_iter) // self.grow_interval + 1
        points_needed = self.max_num_points - current_n
        num_new = math.ceil(points_needed / remaining_grow_steps)
        num_new = min(num_new, self.max_num_points - current_n)

        if num_new <= 0:
            return

        old_num = self.add_gaussians(num_new)
        if old_num is not None:
            self.rebuild_optimizer(old_num)
            print(f"[D3] Iter {current_iter}: {old_num} -> {self._xyz.shape[0]} Gaussians (+{self._xyz.shape[0] - old_num})")
    