import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision import transforms
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from torchvision import utils as vutil
import lpips

# ==========================================
#  Helper Classes & Functions
# ==========================================

class MatrixOperator:
    def __init__(self, tensor):
        self.tensor = tensor

    def __mul__(self, other):
        return torch.matmul(self.tensor, other)

def load_images_as_tensors(dir_path, image_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image_paths = []
    supported_formats = ["*.jpg", "*.jpeg", "*.png"]
    for fmt in supported_formats:
        image_paths.extend(glob.glob(os.path.join(dir_path, fmt)))
    
    if not image_paths:
        print(f"Warning: No images found in {dir_path}")
        return torch.empty(0)

    tensors_list = []
    for path in tqdm(image_paths, desc="Loading Images"):
        try:
            img = Image.open(path).convert("RGB")
            tensors_list.append(transform(img))
        except Exception as e:
            print(f"Error loading {path}: {e}")

    return torch.stack(tensors_list, dim=0)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if verbose:
        if len(m) > 0: print("missing keys:", m)
        if len(u) > 0: print("unexpected keys:", u)
    model.cuda()
    model.eval()
    return model

def save_img_individually(img, path):
    if len(img.shape) == 3: img = img.unsqueeze(0)
    dirname = os.path.dirname(path)
    basename = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1]
    os.makedirs(dirname, exist_ok=True)
    for i in range(img.shape[0]):
        vutil.save_image(img[i], os.path.join(dirname, f"{basename}_{i}{ext}"))
    print(f"Saved images to {dirname}/")

def remove_png(path):
    for file in glob.glob(f'{path}/*.png'):
        try: os.remove(file)
        except: pass

# ==========================================
#  Mappers (Latent <-> MIMO Streams)
# ==========================================
def latent_to_mimo_streams(z_real, t_antennas):
    B, C, H, W = z_real.shape
    z_flat = z_real.view(B, -1)
    
    total_elements = z_flat.shape[1]
    L_complex = total_elements // (t_antennas * 2)
    cutoff = L_complex * t_antennas * 2
    z_used = z_flat[:, :cutoff]
    
    z_view = z_used.view(B, t_antennas, -1)
    real_part, imag_part = torch.chunk(z_view, 2, dim=2)
    s = torch.complex(real_part, imag_part)
    
    return s, (B, C, H, W)

def mimo_streams_to_latent(s, original_shape):
    real_part = s.real
    imag_part = s.imag
    z_view = torch.cat([real_part, imag_part], dim=2) # (B, t, 2L)
    z_flat = z_view.view(s.shape[0], -1)
    
    target_size = np.prod(original_shape[1:])
    current_size = z_flat.shape[1]
    
    if current_size < target_size:
        padding = torch.zeros(s.shape[0], target_size - current_size, device=s.device)
        z_flat = torch.cat([z_flat, padding], dim=1)
    
    return z_flat.view(original_shape)

# ==========================================
#  Main Script
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # MIMO Parameters
    t_mimo = 2 
    r_mimo = 2 
    N_pilot = 2 
    
    P_power = 1.0 
    Perfect_Estimate = False
    # python -m scripts.mimo_dps_proposed > output_dps_estimate.txt
    # Rename Experiment to "Proposed"
    base_experiment_name = f"MIMO_Proposed_LS/t={t_mimo}_r={r_mimo}"
    
    parser.add_argument("--input_path", type=str, default="input_img")
    parser.add_argument("--outdir", type=str, default=f"outputs/{base_experiment_name}")
    parser.add_argument("--nosample_outdir", type=str, default=f"outputs/{base_experiment_name}/nosample")
    parser.add_argument("--sentimgdir", type=str, default="./sentimg")
    
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--dps_scale", type=float, default=0.1) 
    
    opt = parser.parse_args()

    # Directory Setup
    suffix = "perfect" if Perfect_Estimate else "estimated"
    opt.outdir = os.path.join(opt.outdir, suffix)
    opt.nosample_outdir = os.path.join(opt.nosample_outdir, suffix)

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)
    remove_png(opt.outdir)

    # Load Model
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    # Load Images
    img = load_images_as_tensors(opt.input_path).to(device)
    batch_size = img.shape[0]
    save_img_individually(img, opt.sentimgdir + "/original.png")

    # Encode & Normalize
    z = model.encode_first_stage(img)
    z = model.get_first_stage_encoding(z).detach()
    
    z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
    z_var = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    eps = 1e-7
    z_norm = (z - z_mean) / (torch.sqrt(z_var) + eps)
    
    # ----------------------------------------------------------------
    # 1. Map Latent to MIMO Streams
    # ----------------------------------------------------------------
    s_0_real = z_norm / np.sqrt(2.0)
    s_0, latent_shape = latent_to_mimo_streams(s_0_real, t_mimo)
    s_0 = s_0.to(device)
    
    L_len = s_0.shape[2]
    print(f"MIMO Streams: {t_mimo}x{L_len} complex symbols")

    # ----------------------------------------------------------------
    # 2. Pilot Signal Setup
    # ----------------------------------------------------------------
    t_vec = torch.arange(t_mimo, device=device)
    N_vec = torch.arange(N_pilot, device=device)
    tt, NN = torch.meshgrid(t_vec, N_vec, indexing='ij')
    P = torch.sqrt(torch.tensor(P_power/(N_pilot*t_mimo))) * torch.exp(1j*2*torch.pi*tt*NN/N_pilot)
    P = P.to(device) 

    # Simulation Loop
    for snr in range(-5, 26, 3): 
        print(f"\n======== SNR = {snr} dB ========")
        
        noise_variance = t_mimo / (10**(snr/10))
        sigma_n = np.sqrt(noise_variance / 2.0)

        # A. Channel Generation
        H_real = torch.randn(batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
        H_imag = torch.randn(batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
        H = torch.complex(H_real, H_imag)

        # B. Pilot Transmission & Estimation
        V_real = torch.randn(batch_size, r_mimo, N_pilot, device=device) * np.sqrt(noise_variance/2)
        V_imag = torch.randn(batch_size, r_mimo, N_pilot, device=device) * np.sqrt(noise_variance/2)
        V = torch.complex(V_real, V_imag)
        
        S_pilot = torch.matmul(H, P) + V
        
        if Perfect_Estimate:
            H_hat = H
            sigma_e2 = 0.0
        else:
            P_herm = P.mH
            inv_PP = torch.inverse(torch.matmul(P, P_herm))
            H_hat = torch.matmul(S_pilot, torch.matmul(P_herm, inv_PP))
            sigma_e2 = noise_variance / (P_power/t_mimo)

        # C. Data Transmission
        W_real = torch.randn(batch_size, r_mimo, L_len, device=device) * sigma_n
        W_imag = torch.randn(batch_size, r_mimo, L_len, device=device) * sigma_n
        W = torch.complex(W_real, W_imag)
        
        Y = torch.matmul(H, s_0) + W
        
        # D. MMSE Initialization
        eff_noise = sigma_e2 + noise_variance
        
        H_hat_H = H_hat.mH
        Gram = torch.matmul(H_hat_H, H_hat) 
        Reg = eff_noise * torch.eye(t_mimo, device=device).unsqueeze(0)
        
        inv_mat = torch.inverse(Gram + Reg)
        W_mmse = torch.matmul(inv_mat, H_hat_H) # (B, t, r)
        
        # Equalization
        s_mmse = torch.matmul(W_mmse, Y) # (B, t, L)
        
        # Save MMSE Result
        z_init_real = mimo_streams_to_latent(s_mmse, latent_shape)
        z_init_mmse = z_init_real * np.sqrt(2.0)
        
        z_nosample = z_init_mmse * (torch.sqrt(z_var) + eps) + z_mean
        rec_nosample = model.decode_first_stage(z_nosample)
        save_img_individually(rec_nosample, f"{opt.nosample_outdir}/mmse_snr{snr}.png")
        
        # E. Prepare for Proposed Method (DPS)
        
        # [Fix 2] Calculate Post-MMSE Noise Variance
        # MMSEフィルタ通過後のノイズ共分散行列の対角成分(平均)を推定
        # R_post = W_mmse @ (eff_noise * I) @ W_mmse^H = eff_noise * (W_mmse @ W_mmse^H)
        W_W_H = torch.matmul(W_mmse, W_mmse.mH) # (B, t, t)
        noise_power_factor = W_W_H.diagonal(dim1=-2, dim2=-1).real.mean() # scalar (batch average handled in sampler if needed, but mean here is safe)
        post_mmse_noise_var = eff_noise * noise_power_factor
        
        # 通常のチャネルノイズ逆数 (Guidance用)
        eff_var_scalar = noise_variance + sigma_e2
        Sigma_inv = 1.0 / eff_var_scalar
        
        H_hat_wrapper = MatrixOperator(H_hat)
        
        def forward_mapper(z):
            return latent_to_mimo_streams(z / np.sqrt(2.0), t_mimo)
        
        def backward_mapper(s, shape):
            z = mimo_streams_to_latent(s, shape)
            return z * np.sqrt(2.0)

        # [Fix 1] Normalization
        actual_std = z_init_mmse.std(dim=(1, 2, 3), keepdim=True)
        z_init_normalized = z_init_mmse / (actual_std + 1e-8)
        
        cond = model.get_learned_conditioning(batch_size * [""])

        # [Fix 1] Adaptive Guidance Scale
        # 低SNR時はガイドを弱めて、モデルの事前分布（補完能力）を優先する
        current_zeta = opt.dps_scale
        if snr < 5:
            current_zeta *= 0.1
            print(f"[Info] Low SNR ({snr}dB): Reducing Zeta to {current_zeta:.4f}")
        
        print(f"Starting Proposed Sampling... Steps={opt.ddim_steps}, Zeta={current_zeta}")
        target_variance_for_timestep = eff_noise
        # Call Proposed Sampling (Renamed from method_c)
        samples = sampler.proposed_dps_sampling(
            S=opt.ddim_steps,
            batch_size=batch_size,
            shape=z.shape[1:4], 
            conditioning=cond,
            
            y=Y,                 
            H_hat=H_hat_wrapper, 
            Sigma_inv=torch.tensor(Sigma_inv, device=device),
            z_init=z_init_normalized, 
            zeta=current_zeta,
            
            mapper=forward_mapper,
            inv_mapper=backward_mapper,
            
            # [Fix 2] Pass correct noise variance for start timestep
            initial_noise_variance=post_mmse_noise_var,
            
            eta=0.0,
            verbose=False
        )
        
        # Denormalize & Decode
        z_restored = samples * (torch.sqrt(z_var) + eps) + z_mean
        rec_proposed = model.decode_first_stage(z_restored)
        
        save_img_individually(rec_proposed, f"{opt.outdir}/proposed_snr{snr}.png")
        print(f"Saved result for SNR {snr}")