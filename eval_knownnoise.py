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
            if len(parts) < 3: return None
            img_id = parts[-1]
            snr_str = parts[-2]
            if not img_id.isdigit(): return None
            float(snr_str) 
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
        print(f"  -> No matched pairs found.")
        return [], []

    xy = []
    for snr_key, total in dic_sum.items():
        try:
            xy.append((float(snr_key), total / dic_num[snr_key]))
        except ValueError: continue
    
    xy.sort()
    return [item[0] for item in xy], [item[1] for item in xy]

def plot_results(results, title_suffix="", output_filename="snr_vs_metric.png"):
    # 色とマーカーの定義を増やして多くのラインに対応
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'X']
    linestyles = ['-', '--', '-.', ':']
    
    plt.figure(figsize=(12, 8)) # 凡例が増えるため少し大きく
    for i, (x_vals, y_vals, label) in enumerate(results):
        if not x_vals: continue
        plt.plot(x_vals, y_vals, marker=markers[i%len(markers)], linestyle=linestyles[(i//len(colors))%len(linestyles)], 
                 label=label, color=colors[i%len(colors)], markersize=8, linewidth=2)
    
    plt.xlabel("SNR (dB)", fontsize=14)
    plt.ylabel(f"Metric: {title_suffix}", fontsize=14)
    plt.title(f"SNR vs {title_suffix} Comparison", fontsize=16)
    
    # 凡例をグラフの外に出す（項目が多い場合のため）
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6, which='both')
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"\nPlot saved as '{output_filename}'.")

def main():
    parser = argparse.ArgumentParser(description="Plot SNR vs Metric with Perfect Estimate")
    parser.add_argument("--root_dir", default="outputs", help="Root outputs directory")
    parser.add_argument("--sent", "-s", default="./sentimg", help="Directory for 'sent' images")
    parser.add_argument("--metric", "-m", choices=["ssim","mse","psnr","lpips","all"], default="ssim", help="Metric to use")
    
    # 実験パラメータ
    parser.add_argument("--t", type=int, default=2)
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--ft", type=int, default=100)

    # ターゲット選択
    parser.add_argument("--targets", nargs="+", default=["all"], 
                        help="Choose: 'all', 'estimated', 'perfect', 'prop', 'bench', or individual keys like 'prop_perf'.")

    args = parser.parse_args()
    metrics_to_run = ["ssim", "mse", "psnr", "lpips"] if args.metric == "all" else [args.metric]

    # LPIPS Init
    lpips_model, device = None, None
    if "lpips" in metrics_to_run:
        if lpips is None or torch is None: return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    # パス構築
    kn_base = f"SU-MIMO_KnownNoise/t={args.t}_r={args.r}_ft={args.ft}"
    bn_base = f"SU-MIMO_Benchmark/t={args.t}_r={args.r}"

    # --- 利用可能なターゲットの定義 ---
    available_targets = {
        # === 推定あり (Estimated) ===
        "prop":        ("Proposed (Estimated)",      os.path.join(kn_base, "estimated")),
        "prop_zf":     ("Proposed (Estimated ZF)",   os.path.join(kn_base, "nosample_estimated")),
        "bench":       ("Benchmark (Estimated)",     os.path.join(bn_base, "estimated")),
        "bench_zf":    ("Benchmark (Estimated ZF)",  os.path.join(bn_base, "nosample_estimated")),
        
        # === 完全推定 (Perfect Estimate) ===
        "prop_perf":   ("Proposed (Perfect)",        os.path.join(kn_base, "perfect_estimate")),
        "prop_perf_zf":("Proposed (Perfect ZF)",     os.path.join(kn_base, "nosample_perfect")),
        "bench_perf":  ("Benchmark (Perfect)",       os.path.join(bn_base, "perfect_estimate")),
        "bench_perf_zf":("Benchmark (Perfect ZF)",   os.path.join(bn_base, "nosample_perfect")),
    }

    # 選択ロジック
    selected_keys = []
    
    if "all" in args.targets:
        # 全てを表示
        selected_keys = list(available_targets.keys())
        
    elif "estimated" in args.targets:
        # 推定ありのみ (従来と同じ)
        selected_keys = ["prop", "prop_zf", "bench", "bench_zf"]
        
    elif "perfect" in args.targets:
        # 完全推定のみ
        selected_keys = ["prop_perf", "prop_perf_zf", "bench_perf", "bench_perf_zf"]
        
    elif "proposed" in args.targets:
        # 提案手法の全て (推定・完全含む)
        selected_keys = [k for k in available_targets.keys() if "prop" in k]
        
    elif "benchmark" in args.targets:
        # ベンチマークの全て (推定・完全含む)
        selected_keys = [k for k in available_targets.keys() if "bench" in k]
        
    else:
        # 個別指定 (例: prop prop_perf bench)
        selected_keys = args.targets

    # リスト作成
    plot_list = []
    for key in selected_keys:
        if key in available_targets:
            plot_list.append(available_targets[key])
        else:
            print(f"Warning: Unknown target '{key}'. Skipping.")

    if not plot_list:
        print("No valid targets selected.")
        return

    # 実行ループ
    for metric in metrics_to_run:
        print(f"\n=== METRIC: {metric.upper()} ===")
        results = [] 

        for label, subpath in plot_list:
            full_path = os.path.join(args.root_dir, subpath)
            
            # ディレクトリの存在確認
            if not os.path.exists(full_path):
                # 存在しない場合は静かにスキップ（Perfectを計算していない場合などがあるため）
                # print(f"  [Skip] {label}: Path not found ({subpath})") 
                continue
                
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