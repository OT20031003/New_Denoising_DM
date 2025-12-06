import os
import argparse
import re
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Imports for LPIPS ---
try:
    import torch
    import lpips
except ImportError:
    print("Warning: 'torch' or 'lpips' libraries not found.")
    print("To use the LPIPS metric, please install them: pip install torch lpips")
    torch = None
    lpips = None
# -------------------------------

def np_to_torch(img_np):
    img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor / 127.5) - 1.0
    return img_tensor

def compute_metric(x, y, metric='ssim', lpips_model=None, device=None):
    if metric == 'ssim':
        data_range = float(x.max() - x.min())
        if data_range == 0: return 1.0
        return ssim(x, y, channel_axis=-1, data_range=data_range)

    xd = x.astype(np.float64)
    yd = y.astype(np.float64)
    mse = float(np.mean((xd - yd) ** 2))

    if metric == 'mse':
        return mse
    elif metric == 'psnr':
        if mse == 0: return 100.0
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return float(psnr)
    elif metric == 'lpips':
        if lpips_model is None or device is None:
            raise ValueError("LPIPS model required.")
        tensor_x = np_to_torch(x).to(device)
        tensor_y = np_to_torch(y).to(device)
        with torch.no_grad():
            dist = lpips_model(tensor_x, tensor_y)
        return float(dist.item())
    else:
        raise ValueError("Invalid metric.")

def parse_filename_info(filename, is_sent=False):
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split('_')
    try:
        if is_sent:
            if len(parts) < 2: return None
            img_id = parts[-1]
            if not img_id.isdigit(): return None
            return {'id': img_id}
        else:
            # 想定形式: output_{snr}_{id}.png
            if len(parts) < 3: return None
            img_id = parts[-1]
            snr_str = parts[-2]
            if not img_id.isdigit(): return None
            float(snr_str) # check if float
            return {'id': img_id, 'snr': snr_str}
    except ValueError:
        return None

def calculate_snr_vs_metric(sent_path, received_path, metric='ssim', resize=(256,256), lpips_model=None, device=None):
    dic_sum = {}
    dic_num = {}

    if not os.path.isdir(sent_path):
        print(f"Error: Sent directory not found: {sent_path}")
        return [], []
    if not os.path.isdir(received_path):
        # ディレクトリが存在しない場合はエラーを出さずに空を返す（比較対象がない場合のため）
        # ここでパスが見つからない旨を出力するとデバッグしやすい
        print(f"Warning: Directory not found: {received_path}")
        return [], []

    print(f"Processing: {received_path} ...")

    sent_images = {}
    for sp in os.listdir(sent_path):
        if not sp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
        info = parse_filename_info(sp, is_sent=True)
        if info: sent_images[info['id']] = os.path.join(sent_path, sp)

    if not sent_images:
        print("Error: No valid images found in sent directory.")
        return [], []

    file_count = 0
    for rp in os.listdir(received_path):
        if not rp.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
        info = parse_filename_info(rp, is_sent=False)
        if not info: continue

        img_id = info['id']
        snr_str = info['snr']

        if img_id in sent_images:
            try:
                sentimg = Image.open(sent_images[img_id]).convert('RGB')
                recimg = Image.open(os.path.join(received_path, rp)).convert('RGB')
                if resize:
                    sentimg = sentimg.resize(resize)
                    recimg = recimg.resize(resize)

                val = compute_metric(np.array(sentimg), np.array(recimg), metric=metric, lpips_model=lpips_model, device=device)
                dic_sum[snr_str] = dic_sum.get(snr_str, 0.0) + val
                dic_num[snr_str] = dic_num.get(snr_str, 0) + 1
                file_count += 1
            except Exception:
                continue

    if not dic_sum:
        print(f"  -> No matched pairs found in {received_path}")
        return [], []

    print(f"  -> Successfully processed {file_count} images.")
    print("  -> Data counts per SNR:")
    
    sorted_snrs = sorted(dic_num.keys(), key=lambda x: float(x))
    for snr_key in sorted_snrs:
        count = dic_num[snr_key]
        print(f"     SNR {snr_key:>3} dB: {count} images averaged")
    print("-" * 40)

    xy = []
    for snr_key, total in dic_sum.items():
        try:
            xy.append((float(snr_key), total / dic_num[snr_key]))
        except ValueError: continue
    
    xy.sort()
    return [item[0] for item in xy], [item[1] for item in xy]

def plot_results(results, title_suffix="", output_filename="snr_vs_metric.png"):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(12, 8))
    for i, (x_vals, y_vals, label) in enumerate(results):
        if not x_vals: continue
        plt.plot(x_vals, y_vals, 
                 marker=markers[i%len(markers)], 
                 linestyle=linestyles[(i//len(colors))%len(linestyles)], 
                 label=label, 
                 color=colors[i%len(colors)], 
                 markersize=8, linewidth=2)
    
    plt.xlabel("SNR (dB)", fontsize=14)
    plt.ylabel(f"Metric: {title_suffix}", fontsize=14)
    plt.title(f"SNR vs {title_suffix} Comparison", fontsize=16)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6, which='both')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"\nPlot saved as '{output_filename}'.")

def main():
    parser = argparse.ArgumentParser(description="Plot SNR vs Metric for ZF, MMSE, and Projected Guidance")
    parser.add_argument("--root_dir", default="outputs", help="Root outputs directory")
    parser.add_argument("--sent", "-s", default="./sentimg", help="Directory for 'sent' images")
    parser.add_argument("--metric", "-m", choices=["ssim","mse","psnr","lpips","all"], default="ssim", help="Metric to use")
    
    # 実験パラメータ
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--ft", type=int, default=50) 
    parser.add_argument("--inj", type=float, default=1.0, help="Injection scale for Projected Guidance (default: 1.0)")

    # ターゲット選択
    parser.add_argument("--targets", nargs="+", default=["all"], 
                        help="Choose targets (e.g., 'prop', 'bench') or groups (e.g., 'all', 'zf', 'proj')")

    args = parser.parse_args()
    metrics_to_run = ["ssim", "mse", "psnr", "lpips"] if args.metric == "all" else [args.metric]

    # LPIPS Init
    lpips_model, device = None, None
    if "lpips" in metrics_to_run:
        if lpips is None or torch is None: return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    # --- パス構築 ---
    # パスが正しいか、ログの出力を見ながら確認してください
    
    # 1. ZF (従来/Standard Known Noise)
    kn_base_zf = f"SU-MIMO_KnownNoise/t={args.t}_r={args.r}_ft={args.ft}"
    # 2. Benchmark (Standard)
    bn_base_zf = f"SU-MIMO_Benchmark/t={args.t}_r={args.r}"
    
    # 3. MMSE (Known Noise MMSE)
    kn_base_mmse = f"SU-MIMO_KnownNoise_MMSE/t={args.t}_r={args.r}_ft={args.ft}"
    # 4. Benchmark MMSE
    bn_base_mmse = f"SU-MIMO_Benchmark_MMSE/t={args.t}_r={args.r}"

    # 5. Projected Guidance
    # ※重要: ユーザー環境のディレクトリ名に合わせています。もし "Projected_Guidance_Benchmark" なら修正してください。
    pg_base = f"Projected_Guidance/t={args.t}_r={args.r}_inj={args.inj}"
    # python eval_knownnoise.py --inj 1.0 -m all --targets proj proj_lin bench bench_lin
    # --- 利用可能なターゲットの定義 ---
    available_targets = {
        # === ZF: 推定あり (Estimated) ===
        "prop":        ("Proposed ZF (KnownNoise)",  os.path.join(kn_base_zf, "estimated")),
        "prop_lin":    ("Proposed ZF Linear",        os.path.join(kn_base_zf, "nosample_estimated")),
        "bench":       ("Benchmark ZF",              os.path.join(bn_base_zf, "estimated")),
        "bench_lin":   ("Benchmark ZF Linear",       os.path.join(bn_base_zf, "nosample_estimated")),
        
        # === ZF: 完全推定 (Perfect Estimate) ===
        "prop_perf":   ("Proposed ZF Perfect",       os.path.join(kn_base_zf, "perfect_estimate")),
        "prop_perf_lin":("Proposed ZF Perf Lin",     os.path.join(kn_base_zf, "nosample_perfect")),
        "bench_perf":  ("Benchmark ZF Perfect",      os.path.join(bn_base_zf, "perfect_estimate")),
        "bench_perf_lin":("Benchmark ZF Perf Lin",   os.path.join(bn_base_zf, "nosample_perfect")),

        # === MMSE: 推定あり (Estimated) ===
        "prop_mmse":     ("Proposed MMSE (KnownNoise)", os.path.join(kn_base_mmse, "estimated")),
        "prop_mmse_lin": ("Proposed MMSE Linear",       os.path.join(kn_base_mmse, "nosample_estimated")),
        "bench_mmse":    ("Benchmark MMSE",             os.path.join(bn_base_mmse, "estimated")),
        "bench_mmse_lin":("Benchmark MMSE Linear",      os.path.join(bn_base_mmse, "nosample_estimated")),

        # === Projected Guidance (New) ===
        "proj":          ("Projected Guidance",         os.path.join(pg_base, "estimated")),
        "proj_lin":      ("Projected Guidance Linear",  os.path.join(pg_base, "nosample_estimated")),
        "proj_perf":     ("Projected Guidance Perf",    os.path.join(pg_base, "perfect_estimate")),
        "proj_perf_lin": ("Projected Guidance Perf Lin",os.path.join(pg_base, "nosample_perfect")),
    }

    # === 修正: ターゲット選択ロジック ===
    selected_keys = []
    
    # 予約されたグループ名定義
    groups = {
        "all": list(available_targets.keys()),
        "estimated": [k for k in available_targets.keys() if "estimated" in available_targets[k][1]],
        "perfect": [k for k in available_targets.keys() if "perfect" in available_targets[k][1]],
        "zf": [k for k in available_targets.keys() if "ZF" in available_targets[k][0]],
        "mmse": [k for k in available_targets.keys() if "MMSE" in available_targets[k][0]],
        
        # 'proj' をグループとして指定したい場合
        "proj_group": [k for k in available_targets.keys() if "Projected" in available_targets[k][0]],
        
        "compare_bench": ["bench", "bench_mmse", "proj"],
    }

    for t in args.targets:
        # 1. グループ名に一致するか？
        if t in groups:
            selected_keys.extend(groups[t])
        
        # 2. 個別のキー名に一致するか？ (proj, bench など)
        elif t in available_targets:
            selected_keys.append(t)
            
        # 3. 特殊対応: "proj" と入力された場合、キーの "proj" (estimated) を指すのか、
        #    グループとしての "Projected全般" を指すのか曖昧になりやすいため、
        #    ここでは「個別のキー」として扱う (上のelifで処理済み)。
        #    もし "proj" で全projectedデータを出したいなら、コマンドライン引数で
        #    "proj_group" を指定するか、個別に列挙してください。
        
        else:
            print(f"Warning: Unknown target or group '{t}'. Skipping.")

    # リスト作成と重複排除
    unique_keys = sorted(list(set(selected_keys)))
    
    plot_list = []
    for key in unique_keys:
        if key in available_targets:
            plot_list.append(available_targets[key])

    if not plot_list:
        print("No valid targets selected.")
        return

    # 実行ループ
    for metric in metrics_to_run:
        print(f"\n=== METRIC: {metric.upper()} ===")
        results = [] 

        for label, subpath in plot_list:
            full_path = os.path.join(args.root_dir, subpath)
            
            x, y = calculate_snr_vs_metric(args.sent, full_path, metric=metric, 
                                           lpips_model=lpips_model, device=device)
            if x: results.append((x, y, label))

        if not results:
            print("No data found to plot. Check if output directories exist.")
            continue

        outname = f"comparison_t{args.t}_{metric}.png"
        plot_results(results, title_suffix=metric.upper(), output_filename=outname)

if __name__ == "__main__":
    main()