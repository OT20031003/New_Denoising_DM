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
import os
import glob
import lpips

# ... (load_images_as_tensors, load_model_from_config, save_img_individually などの関数定義は変更なし。そのまま使ってください) ...
def load_images_as_tensors(dir_path, image_size=(256, 256)):
    """指定されたディレクトリ内のすべての画像ファイルを読み込む"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image_paths = []
    supported_formats = ["*.jpg", "*.jpeg", "*.png"]
    for fmt in supported_formats:
        image_paths.extend(glob.glob(os.path.join(dir_path, fmt)))
    if not image_paths:
        return []
    tensors_list = []
    for t in trange(len(image_paths), desc="Loading Image"):
        path = image_paths[t]
        try:
            img = Image.open(path).convert("RGB")
            tensor_img = transform(img)
            tensors_list.append(tensor_img)
        except Exception:
            pass
    if not tensors_list:
        return torch.empty(0)
    return torch.stack(tensors_list, dim=0)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
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
    for i in range(img.shape[0]):
        vutil.save_image(img[i], os.path.join(dirname, f"{basename}_{i}{ext}"))

def remove_png(path):
    png_files = glob.glob(f'{path}/*.png')
    for file in png_files:
        try:
            os.remove(f"{file}")
        except OSError:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    T = None 
    t = 2    
    N = t
    r = 2    
    ft = 100 
    P_power = 1.0
    Perfect_Estimate = True
    
    experiment_name = f"SU-MIMO_KnownNoise/t={t}_r={r}_ft={ft}"

    parser.add_argument("--prompt", type=str, nargs="?", default="known noise", help="the prompt to render")
    parser.add_argument("--outdir", type=str, nargs="?", default=f"outputs/{experiment_name}")
    parser.add_argument("--nosample_outdir", type=str, nargs="?", default=f"outputs/{experiment_name}/nosample")
    parser.add_argument("--sentimgdir", type=str, nargs='?', default="./sentimg")
    parser.add_argument("--ddim_steps", type=int, default=200, help="number of ddim sampling steps")
    parser.add_argument("--plms", action='store_true', help="use plms sampling")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta")
    parser.add_argument("--n_iter", type=int, default=1, help="sample this often")
    parser.add_argument("--H", type=int, default=256, help="image height")
    parser.add_argument("--W", type=int, default=256, help="image width")
    parser.add_argument("--n_samples", type=int, default=4, help="how many samples")
    parser.add_argument("--scale", type=float, default=5.0, help="guidance scale")
    parser.add_argument("--input_path", type=str, default="input_img", help="input image path")
    parser.add_argument("--intermediate_path", type=str, default=None, help="intermediate path")
    parser.add_argument("--intermediate_skip", type=int, default=1, help="intermediate path")
    opt = parser.parse_args()

    if opt.intermediate_path != None:
        os.makedirs(opt.intermediate_path, exist_ok=True)

    base_outdir = f"outputs/{experiment_name}"
    if Perfect_Estimate:
        opt.outdir = os.path.join(base_outdir, "perfect_estimate")
        opt.nosample_outdir = os.path.join(base_outdir, "nosample_perfect")
    else:
        opt.outdir = os.path.join(base_outdir, "estimated")
        opt.nosample_outdir = os.path.join(base_outdir, "nosample_estimated")

    print(f"Output Directory: {opt.outdir}")

    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)

    remove_png(opt.outdir)
    eps = 0.0000001
    img = load_images_as_tensors(opt.input_path)
    batch_size = img.shape[0]

    print(f"img shape = {img.shape}")
    save_img_individually(img, opt.sentimgdir + "/sentimg.png")
    img = img.to(device=device)
    
    # Encode & Normalize
    z = model.encode_first_stage(img)
    z = model.get_first_stage_encoding(z).detach()
    z_encode_mean = z.mean(dim=(1, 2, 3), keepdim=True)
    z_variances_original = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    z = (z - z_encode_mean) / (torch.sqrt(z_variances_original) + eps)
    
    # 既知ノイズの準備
    eps_known = torch.randn_like(z).to(device)
    alphas_cumprod = model.alphas_cumprod.to(device)
    alpha_bar_t = alphas_cumprod[ft]
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

    # 1. 既知ノイズ付加 (Proposed Method: Known Noise Injection)
    # ここが提案手法のキモ：送信側で既知ノイズを埋め込む
    z_noisy = sqrt_alpha_bar_t * z + sqrt_one_minus_alpha_bar_t * eps_known
    
    z_sender = z_noisy # すでに電力は1
    
    # マッピング
    q_real_data = z_sender / torch.sqrt(torch.tensor(2.0)).view(-1, 1, 1, 1).to(device)
    q_view = q_real_data.view(batch_size, t, -1)
    l = q_view.shape[2] // 2
    real_part, imag_part = torch.chunk(q_view, 2, dim=2)
    q = torch.complex(real_part, imag_part).to(device)

    # パイロット
    t_vec = torch.arange(t, device=device)
    N_vec = torch.arange(N, device=device)
    tt, NN = torch.meshgrid(t_vec, N_vec)
    P = torch.sqrt(torch.tensor(P_power/(N*t)))* torch.exp(1j*2*torch.pi*tt*NN/N)

    for snr in range(-5, 10, 1):
        print(f"--------SNR = {snr} (Known Noise)-----------")
        noise_variance = t/(10**(snr/10))
        
        # Channel Simulation
        X = q
        H_real = torch.randn(batch_size, r, t) * torch.sqrt(torch.tensor(0.5))
        H_imag = torch.randn(batch_size, r, t) * torch.sqrt(torch.tensor(0.5))
        H = H_real + H_imag * 1j
        H = H.to(device)
        
        V_real = torch.randn(batch_size, r, N) * torch.sqrt(torch.tensor(0.5*noise_variance))
        V_imag = torch.randn(batch_size, r, N) * torch.sqrt(torch.tensor(0.5*noise_variance))
        V = V_real + V_imag * 1j
        V = V.to(device)
        S = H @ P + V

        H_hat = S @ (P.mH @ torch.inverse(P@P.mH))
        H_tilde = H_hat - H

        W_real = torch.randn(batch_size, r, l) * torch.sqrt(torch.tensor(0.5*noise_variance))
        W_imag = torch.randn(batch_size, r, l) * torch.sqrt(torch.tensor(0.5*noise_variance))
        W = W_real + W_imag * 1j
        W = W.to(device)

        Y = H @ X + W
        
        # ZF Equalization
        if Perfect_Estimate:
            H_for_ZF = H
        else:
            H_for_ZF = H_hat
            
        inv_HH = torch.inverse(H_for_ZF.mH @ H_for_ZF)
        A = inv_HH @ H_for_ZF.mH
        AY = A @ Y
        
        # === 修正ポイント1: 有効雑音分散の計算 ===
        # ベンチマーク側と条件を合わせるため、ZFによるノイズ増幅を考慮する
        

        noise_amplification = torch.mean(torch.diagonal(inv_HH.real, dim1=1, dim2=2), dim=1) # Shape: (Batch_Size,)

        current_noise_variance = noise_variance * noise_amplification # Shape: (Batch_Size,)
        # 逆符号化 
        AY_real_imag = torch.view_as_real(AY)
        real_part_restored = AY_real_imag[..., 0]
        imag_part_restored = AY_real_imag[..., 1]
        q_view_restored = torch.cat([real_part_restored, imag_part_restored], dim=2)
        z_channel = z.shape[1]
        z_h_size = z.shape[2]
        z_w_size = z.shape[3]
        q_real_data_restored = q_view_restored.view(batch_size, z_channel, z_h_size, z_w_size)

        # 復元 (No Sample)
        z_nosample = q_real_data_restored * torch.sqrt(torch.tensor(2)).view(-1, 1, 1, 1).to(device)
        z_nosample_decoded = z_nosample * (torch.sqrt(z_variances_original)+eps) + z_encode_mean
        recoverd_img_no_samp = model.decode_first_stage(z_nosample_decoded)
        save_img_individually(recoverd_img_no_samp, f"{opt.nosample_outdir}/output_{snr}.png")

        # === 修正ポイント2: 実測値ベースの強制正規化 (Robust Scaling) ===
        # 理論値ではなく、実際のデータの分散を計測して正規化します。
        # これにより、ZFが不安定な場合でも拡散モデルへの入力は必ず分散1.0になります。
        
        # 各バッチごとの実際の標準偏差を計算 (Shape: [Batch_Size, 1, 1, 1])
        actual_std = q_real_data_restored.std(dim=(1, 2, 3), keepdim=True)
        
        # 強制的に分散1.0に正規化
        z_input_scaled = q_real_data_restored / (actual_std + 1e-8)
        
        # 確認用プリント
        print(f"z_input_scaled variance = {torch.var(z_input_scaled, dim=(1, 2, 3))}") 
        # -> これですべて 1.0 付近 (0.99~1.01) になるはずです
        
        # === 重要: サンプラーに渡すノイズ分散の補正 ===
        # 入力を actual_std で割ったため、中のノイズ成分も小さくなっています。
        # サンプラーに「現在のノイズ量はこれくらいだよ」と教える値も、同じ比率で小さくして伝える必要があります。
        # 分散なので、(1/std)^2 倍 します。
        
        # current_noise_variance は shape [Batch] なので、計算用に変形
        actual_var_flat = (actual_std.flatten()) ** 2
        effective_noise_variance = current_noise_variance / actual_var_flat

        cond = model.get_learned_conditioning(z.shape[0] * [""])
        
        # === 修正ポイント3: サンプラー呼び出し ===
        # noise_variance 引数には、補正後の effective_noise_variance を渡す
        samples = sampler.known_noise_guided_ddim_sampling(
            S=opt.ddim_steps, 
            batch_size=batch_size,
            shape=z.shape[1:4],
            x_T=z_input_scaled, # 正規化済み入力
            conditioning=cond,
            noise_variance=effective_noise_variance, # 補正済み有効雑音
            added_timestep=ft, 
            eps_known=eps_known
        )

        # デコード
        # samples は正規化空間にあるので、元のスケールに戻す
        z_restored = samples * (torch.sqrt(z_variances_original) + eps) + z_encode_mean
        recoverd_img = model.decode_first_stage(z_restored)
        
        save_img_individually(recoverd_img, f"{opt.outdir}/output_{snr}.png")
        print(f"Saved SNR {snr}")