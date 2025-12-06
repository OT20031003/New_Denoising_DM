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
#  Helper Functions
# ==========================================

def load_images_as_tensors(dir_path, image_size=(256, 256)):
    """
    指定されたディレクトリ内のすべての画像ファイルを読み込み、
    PyTorchテンソルのリストとして返す。
    """
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
    for t in trange(len(image_paths), desc="Loading Images"):
        path = image_paths[t]
        try:
            img = Image.open(path).convert("RGB")
            tensor_img = transform(img)
            tensors_list.append(tensor_img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    if not tensors_list:
        return torch.empty(0)

    return torch.stack(tensors_list, dim=0)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def save_img_individually(img, path):
    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    dirname = os.path.dirname(path)
    basename = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1]

    os.makedirs(dirname, exist_ok=True)

    batch_size = img.shape[0]
    for i in range(batch_size):
        individual_path = os.path.join(dirname, f"{basename}_{i}{ext}")
        vutil.save_image(img[i], individual_path)

    print(f"{batch_size} images are saved in {dirname}/")

def remove_png(path):
    png_files = glob.glob(f'{path}/*.png')
    for file in png_files:
        try:
            os.remove(f"{file}")
        except OSError:
            pass

# ==========================================
#  Mappers (Latent <-> MIMO Streams)
# ==========================================
def latent_to_mimo_streams(z_real, t_antennas):
    """
    (Batch, C, H, W) -> (Batch, t, L) Complex
    """
    B, C, H, W = z_real.shape
    z_flat = z_real.view(B, -1)
    
    # 複素数化するために要素数は t_antennas * 2 で割り切れる必要がある
    total_elements = z_flat.shape[1]
    L_complex = total_elements // (t_antennas * 2)
    cutoff = L_complex * t_antennas * 2
    z_used = z_flat[:, :cutoff]
    
    z_view = z_used.view(B, t_antennas, -1)
    real_part, imag_part = torch.chunk(z_view, 2, dim=2)
    s = torch.complex(real_part, imag_part)
    
    return s, (B, C, H, W)

def mimo_streams_to_latent(s, original_shape):
    """
    (Batch, t, L) Complex -> (Batch, C, H, W) Real
    """
    real_part = s.real
    imag_part = s.imag
    # Concatenate back
    z_view = torch.cat([real_part, imag_part], dim=2) # (B, t, 2L)
    z_flat = z_view.view(s.shape[0], -1)
    
    target_size = np.prod(original_shape[1:])
    current_size = z_flat.shape[1]
    
    if current_size < target_size:
        padding = torch.zeros(s.shape[0], target_size - current_size, device=s.device)
        z_flat = torch.cat([z_flat, padding], dim=1)
    
    return z_flat.view(original_shape)

# ==========================================
#  Main Script (Benchmark: MMSE + Standard Diffusion)
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # MIMO Parameters
    t_mimo = 2 # 送信アンテナ数 (Streams)
    r_mimo = 2 # 受信アンテナ数
    N_pilot = 2 # パイロット長
    
    P_power = 1.0 
    Perfect_Estimate = False
    # python -m scripts.bench_MMSE > output_bench_MMSE_estimate.txt
    # ベンチマーク用のディレクトリ設定
    base_experiment_name = f"MIMO_Benchmark_MMSE/t={t_mimo}_r={r_mimo}"
    
    parser.add_argument("--input_path", type=str, default="input_img", help="input image path")
    parser.add_argument("--outdir", type=str, nargs="?", default=f"outputs/{base_experiment_name}")
    parser.add_argument("--nosample_outdir", type=str, nargs="?", default=f"outputs/{base_experiment_name}/nosample")
    parser.add_argument("--sentimgdir", type=str, nargs='?', default="./sentimg")
    
    parser.add_argument("--ddim_steps", type=int, default=200, help="number of ddim sampling steps")
    parser.add_argument("--scale", type=float, default=5.0, help="unconditional guidance scale")
    # DPSのスケール引数はここでは使いません
    
    opt = parser.parse_args()

    # ディレクトリのsuffix設定
    suffix = "perfect" if Perfect_Estimate else "estimated"
    opt.outdir = os.path.join(opt.outdir, suffix)
    opt.nosample_outdir = os.path.join(opt.nosample_outdir, suffix)

    print(f"Output Directory: {opt.outdir}")
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
    print(f"Input image shape: {img.shape}")
    save_img_individually(img, opt.sentimgdir + "/original.png")

    # 1. Encode to Latent Space
    z = model.encode_first_stage(img)
    z = model.get_first_stage_encoding(z).detach()
    
    # Normalize Latent (Mean 0, Var 1 approx)
    z_mean = z.mean(dim=(1, 2, 3), keepdim=True)
    z_var_original = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    eps = 1e-7
    z_norm = (z - z_mean) / (torch.sqrt(z_var_original) + eps)
    
    # 2. Map to MIMO Streams (Complex)
    # img2img_benchと同様、信号電力を調整してマッピング
    s_0_real = z_norm / np.sqrt(2.0) 
    s_0, latent_shape = latent_to_mimo_streams(s_0_real, t_mimo)
    s_0 = s_0.to(device)
    
    L_len = s_0.shape[2]
    print(f"MIMO Streams: {t_mimo}x{L_len} complex symbols")

    # 3. Pilot Signal Setup (Orthogonal)
    t_vec = torch.arange(t_mimo, device=device)
    N_vec = torch.arange(N_pilot, device=device)
    tt, NN = torch.meshgrid(t_vec, N_vec, indexing='ij')
    # 直交パイロット行列 P (t x N)
    P = torch.sqrt(torch.tensor(P_power/(N_pilot*t_mimo))) * torch.exp(1j*2*torch.pi*tt*NN/N_pilot)
    P = P.to(device)

    # Simulation Loop over SNR
    for snr in range(-5, 26, 3): 
        print(f"\n======== SNR = {snr} dB (Benchmark: MMSE) ========")
        
        # ノイズ分散
        noise_variance = t_mimo / (10**(snr/10))
        sigma_n = np.sqrt(noise_variance / 2.0)

        # A. Channel Generation H (Batch, r, t)
        H_real = torch.randn(batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
        H_imag = torch.randn(batch_size, r_mimo, t_mimo, device=device) * np.sqrt(0.5)
        H = torch.complex(H_real, H_imag)

        # B. Pilot Transmission & Estimation (LS)
        # Pilot Noise V (Batch, r, N)
        V_real = torch.randn(batch_size, r_mimo, N_pilot, device=device) * np.sqrt(noise_variance/2)
        V_imag = torch.randn(batch_size, r_mimo, N_pilot, device=device) * np.sqrt(noise_variance/2)
        V = torch.complex(V_real, V_imag)
        
        # Received Pilot S_pilot = H P + V
        S_pilot = torch.matmul(H, P) + V
        
        if Perfect_Estimate:
            H_hat = H
            sigma_e2 = 0.0
        else:
            # LS Estimation: H_hat = S * P^H * (P * P^H)^-1
            # 直交パイロットなので P P^H は対角に近い
            P_herm = P.mH
            inv_PP = torch.inverse(torch.matmul(P, P_herm))
            H_hat = torch.matmul(S_pilot, torch.matmul(P_herm, inv_PP))
            
            # 推定誤差分散の理論値 (Standard LS)
            # sigma_e2 approx noise_var / P_pilot_total
            sigma_e2 = noise_variance / (P_power/t_mimo) 

        # C. Data Transmission
        # Data Noise W (Batch, r, L)
        W_real = torch.randn(batch_size, r_mimo, L_len, device=device) * sigma_n
        W_imag = torch.randn(batch_size, r_mimo, L_len, device=device) * sigma_n
        W = torch.complex(W_real, W_imag)
        
        # Received Signal Y = H s_0 + W
        Y = torch.matmul(H, s_0) + W
        
        # D. MMSE Equalization
        # Filter W_mmse = (H_hat^H H_hat + (sigma_n^2 + sigma_e2) I)^-1 H_hat^H
        # 注: Benchでは誤差分散を含めたRobust MMSE構成にします
        eff_noise = noise_variance + sigma_e2
        
        H_hat_H = H_hat.mH
        Gram = torch.matmul(H_hat_H, H_hat) 
        Reg = eff_noise * torch.eye(t_mimo, device=device).unsqueeze(0)
        
        inv_mat = torch.inverse(Gram + Reg)
        W_mmse = torch.matmul(inv_mat, H_hat_H) # (B, t, r)
        
        # Estimate s_hat = W_mmse * Y
        s_mmse = torch.matmul(W_mmse, Y) # (B, t, L)
        
        # E. Reconstruct Latent
        # s_mmse -> z_noisy (Real)
        z_init_real = mimo_streams_to_latent(s_mmse, latent_shape)
        
        # 元のスケールに戻す (s_0 = z_norm/sqrt(2) だったため)
        z_mmse_scaled = z_init_real * np.sqrt(2.0)
        
        # No-Sample Result (単純復号)
        z_nosample = z_mmse_scaled * (torch.sqrt(z_var_original) + eps) + z_mean
        rec_nosample = model.decode_first_stage(z_nosample)
        save_img_individually(rec_nosample, f"{opt.nosample_outdir}/mmse_snr{snr}.png")

        # F. Blind Diffusion Sampling (No DPS)
        # ここでは「ノイズの乗った潜在変数」を開始点として、単純なデノイズを行う
        # まず、入力データを標準正規分布(分散1)に強制正規化する (Robust Scaling)
        
        actual_std = z_mmse_scaled.std(dim=(1, 2, 3), keepdim=True)
        z_input_for_sampler = z_mmse_scaled / (actual_std + 1e-8)
        
        # ノイズレベルの推定と補正
        # MMSE後の残留ノイズ増幅率
        # Noise Covariance after MMSE approx W * W^H * noise_var
        # (簡易計算) noise_amplification = mean(diag(W W^H))
        noise_amplification = torch.mean(torch.diagonal(torch.matmul(W_mmse, W_mmse.mH).real, dim1=1, dim2=2), dim=1)
        
        current_noise_power = noise_variance * noise_amplification
        
        # 正規化したため、分散情報もスケーリングする
        actual_var_flat = (actual_std.flatten()) ** 2
        effective_noise_variance = current_noise_power / actual_var_flat
        
        cond = model.get_learned_conditioning(batch_size * [""])
        
        # DDIMサンプラーによるデノイズ (Start Timestepを自動決定して実行)
        # DPSガイダンス (y, H_hat) は渡さない
        samples = sampler.MIMO_decide_starttimestep_ddim_sampling(
            S=opt.ddim_steps,
            batch_size=batch_size,
            shape=z.shape[1:4],
            x_T=z_input_for_sampler, # 正規化済み入力
            conditioning=cond,
            noise_variance=effective_noise_variance, # 補正済みノイズ分散
            verbose=False
        )

        # 復元
        z_restored = samples * (torch.sqrt(z_var_original) + eps) + z_mean
        rec_bench = model.decode_first_stage(z_restored)
        
        save_img_individually(rec_bench, f"{opt.outdir}/bench_snr{snr}.png")
        print(f"Saved result for SNR {snr}")