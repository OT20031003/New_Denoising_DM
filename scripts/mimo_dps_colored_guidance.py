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
import torch

def print_debug_stats(tensor, name="Tensor"):
    """
    テンソルの統計情報を表示して、異常値（NaN/Inf/全ゼロ）を検知します。
    """
    if tensor is None:
        print(f"[{name}] is None!")
        return

    # GPUにある場合はCPUに移して計算
    t = tensor.detach().cpu().float()
    
    min_val = t.min().item()
    max_val = t.max().item()
    mean_val = t.mean().item()
    std_val = t.std().item()
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    
    print(f"--- DEBUG: {name} ---")
    print(f"    Shape: {t.shape}")
    print(f"    Range: [{min_val:.5f}, {max_val:.5f}]")
    print(f"    Mean:  {mean_val:.5f} | Std: {std_val:.5f}")
    
    if has_nan:
        print(f"    !!! WARNING: Contains NaN (Not a Number) !!!")
    if has_inf:
        print(f"    !!! WARNING: Contains Inf (Infinity) !!!")
    if min_val == 0.0 and max_val == 0.0:
        print(f"    !!! WARNING: All values are ZERO !!!")
    print("-------------------------")
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
        print(f"警告: ディレクトリ '{dir_path}' にサポートされている画像ファイルが見つかりません。")
        return []

    tensors_list = []
    for t in trange(len(image_paths), desc="Loading Image"):
        path = image_paths[t]
        try:
            img = Image.open(path).convert("RGB")
            tensor_img = transform(img)
            tensors_list.append(tensor_img)
        except Exception as e:
            print(f"エラー: ファイル '{path}' の読み込みに失敗しました。スキップします。エラー内容: {e}")
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
    """
    バッチ画像を個別のファイルとして1枚ずつ保存する関数。
    """
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
        except OSError as e:
            print(f"Error: {e.strerror} - {e.filename}")
    print(f"remove_png complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # MIMO Settings
    T = None 
    t = 2 # Transmit Antennas
    N = t # Pilot Length
    r = 2 # Receive Antennas
    P_power = 1.0
    Perfect_Estimate = False # 論文の主眼に合わせてFalse推奨 (推定誤差あり)
    # python -m scripts.mimo_dps_colored_guidance > output_guidance.txt
    # Diffusion & DPS Settings
    parser.add_argument("--dps_scale", type=float, default=100.0, help="Gradient scale for DPS guidance (zeta)")
    parser.add_argument("--ddim_steps", type=int, default=200, help="number of ddim sampling steps")
    
    # Path Settings
    # わかりやすいディレクトリ名に変更
    base_experiment_name = f"MIMO_DPS_Colored_MMSE/t={t}_r={r}_scale={parser.parse_args().dps_scale}"
    
    parser.add_argument("--prompt", type=str, nargs="?", default="benchmark", help="the prompt to render")
    parser.add_argument("--outdir", type=str, nargs="?", default=f"outputs/{base_experiment_name}")
    parser.add_argument("--nosample_outdir", type=str, nargs="?", default=f"outputs/{base_experiment_name}/nosample_mmse")
    parser.add_argument("--sentimgdir", type=str, nargs='?', default="./sentimg")
    parser.add_argument("--plms", action='store_true', help="use plms sampling")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta")
    parser.add_argument("--n_iter", type=int, default=1, help="sample this often")
    parser.add_argument("--H", type=int, default=256, help="image height")
    parser.add_argument("--W", type=int, default=256, help="image width")
    parser.add_argument("--n_samples", type=int, default=4, help="how many samples to produce")
    parser.add_argument("--scale", type=float, default=5.0, help="unconditional guidance scale")
    parser.add_argument("--input_path", type=str, default="input_img", help="input image path")
    parser.add_argument("--intermediate_path", type=str, default=None, help="intermediate path")
    parser.add_argument("--intermediate_skip", type=int, default=1, help="intermediate path")
    
    opt = parser.parse_args()
    
    # 出力ディレクトリの再設定 (Estimateの有無で分岐)
    base_path = f"outputs/MIMO_DPS_Colored_MMSE/scale_{opt.dps_scale}"
    if Perfect_Estimate:
        opt.outdir  = os.path.join(base_path, "perfect_estimate")
        opt.nosample_outdir = os.path.join(base_path, "nosample_mmse_perfect")
    else:
        opt.outdir  = os.path.join(base_path, "estimated")
        opt.nosample_outdir = os.path.join(base_path, "nosample_mmse_estimated")
        
    print(f"Output Directory: {opt.outdir}")
    print(f"DPS Scale: {opt.dps_scale}")

    # コンフィグ読み込み
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(opt.sentimgdir, exist_ok=True)
    os.makedirs(opt.nosample_outdir, exist_ok=True)

    remove_png(opt.outdir)
    eps = 0.0000001
    
    # 画像読み込み
    img = load_images_as_tensors(opt.input_path)
    batch_size = img.shape[0]
    print(f"Input image shape = {img.shape}")
    save_img_individually(img, opt.sentimgdir + "/sentimg.png")
    
    # エンコード (Latent表現へ)
    img = img.to(device=device)
    z = model.encode_first_stage(img)
    z = model.get_first_stage_encoding(z).detach()
    
    # 正規化パラメータの取得
    z_encode_mean = z.mean(dim=(1, 2, 3), keepdim=True)
    z_variances_original = torch.var(z, dim=(1, 2, 3)).view(-1, 1, 1, 1)
    
    # 送信用に正規化 (Mean 0, Var 1)
    z_norm = (z - z_encode_mean) / (torch.sqrt(z_variances_original) + eps)
    z_variance = torch.var(z_norm, dim=(1, 2, 3)) 
    
    # MIMO送信シンボルへのマッピング
    z_channel = z.shape[1]
    z_w_size = z.shape[3]
    z_h_size = z.shape[2]
    
    # 電力正規化 (Complex Symbol Power = 1)
    q_real_data = z_norm / torch.sqrt(2*(z_variance + eps)).view(-1, 1, 1, 1)
    
    q_view = q_real_data.view(batch_size, t, -1) 
    l = q_view.shape[2] // 2 
    real_part, imag_part = torch.chunk(q_view, 2, dim=2)
    q = torch.complex(real_part, imag_part).to(device)
    
    # パイロット信号生成
    t_vec = torch.arange(t, device=device)
    N_vec = torch.arange(N, device=device)
    tt, NN = torch.meshgrid(t_vec, N_vec)
    P = torch.sqrt(torch.tensor(P_power/(N*t)))* torch.exp(1j*2*torch.pi*tt*NN/N)

    # ------------------------------------------------------------------
    # Simulation Loop (SNR Sweep)
    # ------------------------------------------------------------------
    for snr in range(-5, 26, 3):
        print(f"--------SNR = {snr} (Method C: MMSE + DPS w/ Colored Noise)-----------")
        noise_variance = t/(10**(snr/10))
        
        # 1. チャネル生成
        X = q 
        H_real = torch.randn(batch_size, r, t) * torch.sqrt(torch.tensor(0.5))
        H_imag = torch.randn(batch_size, r, t) * torch.sqrt(torch.tensor(0.5))
        H = H_real + H_imag * 1j 
        H = H.to(device)
        
        # 2. パイロット送信とチャネル推定
        V_real = torch.randn(batch_size, r, N) * torch.sqrt(torch.tensor(0.5*noise_variance))
        V_imag = torch.randn(batch_size, r, N) * torch.sqrt(torch.tensor(0.5*noise_variance))
        V = V_real + V_imag * 1j 
        V = V.to(device)
        S = H @ P + V 

        H_hat = S @ (P.mH @ torch.inverse(P@P.mH))
        
        # 3. データ送信
        W_real = torch.randn(batch_size, r, l) * torch.sqrt(torch.tensor(0.5*noise_variance))
        W_imag = torch.randn(batch_size, r, l) * torch.sqrt(torch.tensor(0.5*noise_variance))
        W = W_real + W_imag * 1j 
        W = W.to(device)
        
        Y = H @ X + W
        
        # -------------------------------------------------------------
        # [NEW] A. MMSE Equalization
        # ZFではなくMMSEを使用し、ベースライン性能を向上させる
        # -------------------------------------------------------------
        if Perfect_Estimate:
            H_est = H
        else:
            H_est = H_hat

        # MMSE Filter Calculation: W = (H^H H + sigma^2 I)^-1 H^H
        Eye = torch.eye(t, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        Gram = H_est.mH @ H_est
        
        # ノイズ分散を正則化項として加える
        inv_part = torch.inverse(Gram + noise_variance * Eye)
        W_mmse = inv_part @ H_est.mH
        
        # MMSE等化 (Complex Domain)
        Z_mmse_complex = W_mmse @ Y
        
        # -------------------------------------------------------------
        # [NEW] B. Colored Noise Weight Matrix for DPS
        # ガイダンスで使用するマハラノビス距離の重み行列 (Sigmaの逆行列相当)
        # 受信信号ドメインでの整合性を評価するための重み: H^H H
        # -------------------------------------------------------------
        Measurement_Weight = H_est.mH @ H_est 

        # -------------------------------------------------------------
        # [NEW] C. Reconstruction (No Sample / Baseline)
        # MMSE出力をそのまま画像に戻したもの
        # -------------------------------------------------------------
        Z_mmse_real_part = Z_mmse_complex.real
        Z_mmse_imag_part = Z_mmse_complex.imag
        Z_mmse_concat = torch.cat([Z_mmse_real_part, Z_mmse_imag_part], dim=2)
        Z_mmse_reshaped = Z_mmse_concat.view(batch_size, z_channel, z_h_size, z_w_size)
        
        # スケーリング復元 (MMSEは信号電力を下げるため、分散を再調整)
        actual_std = Z_mmse_reshaped.std(dim=(1, 2, 3), keepdim=True)
        Z_mmse_normalized = Z_mmse_reshaped / (actual_std + 1e-8)
        
        # デコードして保存 (Method A: MMSE Only)
        z_nosample = Z_mmse_normalized * (torch.sqrt(z_variances_original) + eps) + z_encode_mean
        recoverd_img_no_samp = model.decode_first_stage(z_nosample)
        save_img_individually(recoverd_img_no_samp, f"{opt.nosample_outdir}/output_{snr}.png")

        # -------------------------------------------------------------
        # [NEW] D. Define Measurement Loss Function for DPS
        # -------------------------------------------------------------
        def measurement_loss_fn(pred_z0_latent):
            """
            DPS内部で呼ばれる損失関数。
            UNetが予測した潜在変数 x0 をシンボルに変換し、MMSE出力(Y)との整合性を測る。
            """
            # 1. 潜在変数 [B, C, H, W] -> シンボル [B, t, L]
            q_view_pred = pred_z0_latent.view(batch_size, t, -1)
            
            # 複素数化
            real_p, imag_p = torch.chunk(q_view_pred, 2, dim=2)
            X_pred = torch.complex(real_p, imag_p)
            
            # 2. 誤差計算 (Error Vector)
            # MMSE等化後の信号(Z_mmse_complex)と、予測信号(X_pred)の差
            diff = Z_mmse_complex - X_pred
            
            # 3. マハラノビス距離 (Colored Noise Guidance)
            # Loss = diff^H @ Measurement_Weight @ diff
            # 次元合わせ: [B, L, t]
            diff_perm = diff.permute(0, 2, 1).unsqueeze(-1) # [B, L, t, 1]
            weight_expanded = Measurement_Weight.unsqueeze(1) # [B, 1, t, t]
            
            # v^H M v
            temp = torch.matmul(weight_expanded, diff_perm) # M v
            val = torch.matmul(diff_perm.conj().transpose(-1, -2), temp) # v^H (M v)
            
            #loss_val = val.real.sum()
            loss_val = val.real.mean()
            return loss_val

        # -------------------------------------------------------------
        # [NEW] E. DPS Sampling
        # -------------------------------------------------------------
        
        # 条件付け (Unconditional or Prompt)
        cond = model.get_learned_conditioning(batch_size * [""])
        
        # MMSE出力を初期値として使用 (Robust Scaling済み)
        z_input_for_sampler = Z_mmse_normalized 

        # dps_sampling の実行 (ddim.py に実装済みと仮定)
        samples = sampler.dps_sampling(
            S=opt.ddim_steps,
            batch_size=batch_size,
            shape=z.shape[1:4],
            x_T=z_input_for_sampler,       
            conditioning=cond,
            measure_loss_fn=measurement_loss_fn,
            guide_scale=opt.dps_scale,
            
            # 【変更点】
            # starttimestep=... を削除し、noise_variance を渡す
            noise_variance=noise_variance, 
            
            #eta=1.0 # 必要に応じて確率的な揺らぎを入れるなら
        )

        # 画像の復元と保存
        
        z_restored = samples * (torch.sqrt(z_variances_original) + eps) + z_encode_mean
        recoverd_img = model.decode_first_stage(z_restored)
        print_debug_stats(recoverd_img, "Final_Image_Before_Save")
        save_img_individually(recoverd_img, f"{opt.outdir}/output_{snr}.png")
        print(f"Saved SNR {snr} (Method C)")