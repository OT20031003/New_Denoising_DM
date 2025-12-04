import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import os
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
import torchvision.utils as vutil

# ==========================================
#  Added: Slerp (Spherical Linear Interpolation)
# ==========================================
def slerp_tensor(val, low, high):
    """
    Slerp for tensors.
    val: float or tensor, interpolation factor (0.0 = low (pred), 1.0 = high (known))
    low: tensor, starting vector (e.g., predicted noise)
    high: tensor, target vector (e.g., known noise)
    """
    dims = low.shape
    # Flatten for calculation
    low_flat = low.reshape(low.shape[0], -1)
    high_flat = high.reshape(high.shape[0], -1)

    # Normalize
    low_norm = low_flat / torch.norm(low_flat, dim=1, keepdim=True)
    high_norm = high_flat / torch.norm(high_flat, dim=1, keepdim=True)

    # Dot product (cos omega)
    dot = (low_norm * high_norm).sum(1)
    dot = torch.clamp(dot, -1.0, 1.0)
    
    omega = torch.acos(dot)
    so = torch.sin(omega)

    # Avoid zero division
    mask = so > 1e-6
    
    s1 = torch.zeros_like(dot)
    s2 = torch.zeros_like(dot)
    
    if isinstance(val, float):
        val = torch.tensor(val, device=low.device)
    
    s1[mask] = torch.sin((1.0 - val) * omega[mask]) / so[mask]
    s2[mask] = torch.sin(val * omega[mask]) / so[mask]
    
    # Linear interpolation fallback
    s1[~mask] = 1.0 - val
    s2[~mask] = val
    
    s1 = s1.view(-1, 1)
    s2 = s2.view(-1, 1)
    
    res = s1 * low_flat + s2 * high_flat
    return res.reshape(dims)


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def forward_diffusion(self,
               S, #ddim_num_steps 200
               batch_size,
               conditioning=None,
               x = None,
               eta=0.,
               verbose=True,
               timestep = 0,
               **kwargs
               ):
        assert(x != None)
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)

        alpha_t_bar = self.alphas_cumprod[timestep]

        # 1. torch.randn_like を使い、x と同じ形状・デバイスでガウスノイズを生成
        epsilon = torch.randn_like(x)

        # 2. 係数を変形
        sqrt_alpha_t_bar = torch.sqrt(alpha_t_bar).view(1, 1, 1, 1)
        if timestep == 0:
            #print(f"ddim.py ===========   sqrt_alpha_t_bar = {sqrt_alpha_t_bar}")
            sqrt_alpha_t_bar = torch.full_like(sqrt_alpha_t_bar, 1.0)
            return x
        sqrt_one_minus_alpha_t_bar = torch.sqrt(1.0 - alpha_t_bar).view(1, 1, 1, 1)

        # 3. ノイズを加える
        return sqrt_alpha_t_bar * x + sqrt_one_minus_alpha_t_bar * epsilon


    @torch.no_grad()
    def sample(self,
               S, #ddim_num_steps
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                #impainting用
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        # c: cond
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0



    @torch.no_grad
    def my_ddim_sampling(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,
               noise_sigma,
               noise_sigma_predict,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               intermediate_path = None,
               intermediate_skip = 1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        alpha_bar_u = 1/(1 + noise_sigma_predict)
        print(f"ddim.py, alpha_bar_u = {alpha_bar_u}")
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps *= 1 # ここでタイムステップをいじくる
        #torch.clamp(start_timesteps, 0, S)

        # alphas
        #indiceからタイムステップ
        print(f"ddim.py start_timesteps after clamp S = {S} : {start_timesteps}")
        results = []
        img = x_T.to(device)
        maxind = start_timesteps.max().item()
        time_range = reversed(range(0, maxind))
        iterator = tqdm(time_range, desc='My Sampling')
        for i, step in enumerate(iterator):
            # 現在のステップが、バッチ内の各画像の開始タイムステップ以下であるかどうかのマスクを作成
            # これにより、まだサンプリングが始まっていない画像をスキップできる
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)

            # ts (タイムステップ) はバッチ全体分を用意
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # indexはスケジュール配列を参照するためのもの。現在のループ回数から計算
            # self.ddim_timestepsはmake_scheduleで作成される長さSの配列
            # stepに対応するindexを探す
            time_idx_tensor = torch.where(torch.from_numpy(self.ddim_timesteps).to(device) == step)[0]
            if time_idx_tensor.numel() == 0:
                # もし現在のstepがスケジュールになければスキップ (補間なども可能)
                continue
            index = time_idx_tensor.item()


            # p_sample_ddimをバッチ全体に対して一度だけ呼び出す
            outs = self.p_sample_ddim(img, conditioning, ts, index, # 修正: indexを追加
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning)
            img_prev, pred_x0 = outs

            # active_maskがTrueの画像だけを、計算結果で更新する
            img = torch.where(active_mask, img_prev, img)
            #print(f"step = {step}, index = {index}")

            if intermediate_path != None and index % intermediate_skip == 0:
                #TODO 途中結果を保存

                decoded_img = self.model.decode_first_stage(pred_x0)

                # VAEのデコーダ出力は [-1, 1] の範囲なので、[0, 1] にスケーリングする
                # decoded_img = (decoded_img + 1.0) / 2.0
                # decoded_img = torch.clamp(decoded_img, min=0.0, max=1.0)

                # バッチ内の各画像を個別に保存
                batch_size = decoded_img.shape[0]
                for i in range(batch_size):
                    # ファイル名をステップ番号と画像インデックスで一意に決定
                    # 例: /path/to/intermediate/step_0180_img_00.png
                    target_dir = os.path.join(intermediate_path, str(snr), str(i))

                    # 2. ディレクトリが存在しなければ再帰的に作成 (exist_ok=True)
                    os.makedirs(target_dir, exist_ok=True)

                    # 3. ファイル名を決定 (ステップ番号)
                    #    例: step_0436.png
                    file_name = f"step_{step:04d}.png"

                    # 4. 最終的なファイルパス
                    img_path = os.path.join(target_dir, file_name)

                    # torchvision.utils.save_image を使って保存
                    vutil.save_image(decoded_img[i], img_path)
                #print(f"save figure")
        return img


    

    @torch.no_grad()
    def known_noise_guided_ddim_sampling(self,
               S,
               batch_size,
               shape,
               noise_variance, 
               conditioning=None,
               eta=0.,
               x_T=None,
               verbose=True,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               added_timestep=0,
               eps_known=None,
               h=1.0,
               guide_strength_start=0.2,
               guide_strength_end=0.8,
               **kwargs
               ):
        
        # 1. スケジュールの作成
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        device = self.model.betas.device # Device定義
        
        # noise_variance のテンソル変換
        if not torch.is_tensor(noise_variance):
            noise_variance = torch.tensor(noise_variance, device=device)

        # 2. 開始タイムステップ u (start_timesteps) の計算
        alpha_bar_t = self.alphas_cumprod[added_timestep]
        # h=1.0 (等化済みと仮定)
        term1 = (torch.sqrt(1 - alpha_bar_t) + torch.sqrt(noise_variance) / h) / torch.sqrt(alpha_bar_t)
        alpha_bar_u = 1 / (term1**2 + 1)
        
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps = start_timesteps.to(device)
        start_timesteps = torch.clamp(start_timesteps, min=0, max=999) # DDPMの最大ステップ内に収める

        print(f"ddim.py Computed start_timestep: {start_timesteps}")

        # 3. ループ設定 (絶対時刻 0~999 ベースで回す)
        maxind = start_timesteps.max().item()
        iterator = tqdm(reversed(range(0, maxind + 1)), desc='Known-Noise Guided DDIM', total=maxind+1)

        img = x_T.clone()

        # 事前にDDIMタイムステップをテンソル化しておく（検索用）
        ddim_timesteps_tensor = torch.from_numpy(self.ddim_timesteps).to(device)

        for i, step in enumerate(iterator):
            # -----------------------------------------------------------
            # A. スケジュール判定 (ユーザー様の修正部分)
            # -----------------------------------------------------------
            # 現在の絶対時刻 'step' が、DDIMのサンプリング時刻に含まれているか確認
            time_idx_tensor = torch.where(ddim_timesteps_tensor == step)[0]
            
            # DDIMスケジュールに存在しない時刻ならスキップ (Continue)
            if time_idx_tensor.numel() == 0:
                continue
            
            # DDIM用のインデックス (0 ~ S-1) を取得
            index = time_idx_tensor.item()

            # -----------------------------------------------------------
            # B. Active Mask 作成
            # -----------------------------------------------------------
            # 個々の画像について、まだデノイジング開始時刻に達していない場合は更新しない
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)

            # -----------------------------------------------------------
            # C. ノイズ予測とSlerp (重要: p_sample_ddimは使わず展開して書く)
            # -----------------------------------------------------------
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # (1) モデルによるノイズ予測
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(img, ts, conditioning)
            else:
                x_in = torch.cat([img] * 2)
                t_in = torch.cat([ts] * 2)
                c_in = torch.cat([unconditional_conditioning, conditioning])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            # (2) 既知ノイズによるガイド (Slerp)
            if eps_known is not None:
                # 進捗度合い (0.0 -> 1.0)
                current_progress = i / (maxind + 1e-6)
                lambda_guide = guide_strength_start + (guide_strength_end - guide_strength_start) * current_progress
                
                # ガイド適用
                e_t = slerp_tensor(lambda_guide, e_t, eps_known)

            # -----------------------------------------------------------
            # D. 状態更新 (Manual DDIM Step)
            # -----------------------------------------------------------
            alphas = self.ddim_alphas
            alphas_prev = self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas

            a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # x_0 推定
            pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
            
            # x_{t-1} 方向ベクトル
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            
            # ノイズ項
            noise = sigma_t * noise_like(img.shape, device, False) * eta

            # 更新後の画像候補
            img_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            # -----------------------------------------------------------
            # E. マスク適用 (開始時刻を過ぎたものだけ更新)
            # -----------------------------------------------------------
            img = torch.where(active_mask, img_prev, img)

        return img

    @torch.no_grad
    def jointdiffusion_ddim_sampling(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,

               noise_sigma_predict,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               intermediate_path = None,
               intermediate_skip = 1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               added_timestep = 0,
               h = torch.tensor(1 + 0j, dtype=torch.complex64),
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)
        alpha_bar_t = self.alphas_cumprod[added_timestep]
        if added_timestep == 0:
            alpha_bar_t = torch.full_like(alpha_bar_t, 1.0)
            print(f"ddim.py alpha_bar_t = {alpha_bar_t}")
        device = self.model.betas.device
        term1 = (torch.sqrt(1 - alpha_bar_t) + torch.sqrt(noise_sigma_predict) / h) / torch.sqrt(alpha_bar_t)
        term1 = term1.abs()
        alpha_bar_u = 1/(term1 * term1 + 1)

        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps *= 1 # ここでタイムステップをいじくる
        #torch.clamp(start_timesteps, 0, S)

        # alphas
        #indiceからタイムステップ
        print(f"ddim.py start_timesteps after clamp S = {S} : {start_timesteps}")
        results = []
        img = x_T.to(device)
        maxind = start_timesteps.max().item()
        time_range = reversed(range(0, maxind))
        iterator = tqdm(time_range, desc='My Sampling')
        for i, step in enumerate(iterator):
            # 現在のステップが、バッチ内の各画像の開始タイムステップ以下であるかどうかのマスクを作成
            # これにより、まだサンプリングが始まっていない画像をスキップできる
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)

            # ts (タイムステップ) はバッチ全体分を用意
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # indexはスケジュール配列を参照するためのもの。現在のループ回数から計算
            # self.ddim_timestepsはmake_scheduleで作成される長さSの配列
            # stepに対応するindexを探す
            time_idx_tensor = torch.where(torch.from_numpy(self.ddim_timesteps).to(device) == step)[0]
            if time_idx_tensor.numel() == 0:
                # もし現在のstepがスケジュールになければスキップ (補間なども可能)
                continue
            index = time_idx_tensor.item()


            # p_sample_ddimをバッチ全体に対して一度だけ呼び出す
            outs = self.p_sample_ddim(img, conditioning, ts, index, # 修正: indexを追加
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning)
            img_prev, pred_x0 = outs

            # active_maskがTrueの画像だけを、計算結果で更新する
            img = torch.where(active_mask, img_prev, img)
            #print(f"step = {step}, index = {index}")

            if intermediate_path != None and index % intermediate_skip == 0:
                #TODO 途中結果を保存

                decoded_img = self.model.decode_first_stage(pred_x0)

                # VAEのデコーダ出力は [-1, 1] の範囲なので、[0, 1] にスケーリングする
                # decoded_img = (decoded_img + 1.0) / 2.0
                # decoded_img = torch.clamp(decoded_img, min=0.0, max=1.0)

                # バッチ内の各画像を個別に保存
                batch_size = decoded_img.shape[0]
                for i in range(batch_size):
                    # ファイル名をステップ番号と画像インデックスで一意に決定
                    # 例: /path/to/intermediate/step_0180_img_00.png
                    target_dir = os.path.join(intermediate_path, str(snr), str(i))

                    # 2. ディレクトリが存在しなければ再帰的に作成 (exist_ok=True)
                    os.makedirs(target_dir, exist_ok=True)

                    # 3. ファイル名を決定 (ステップ番号)
                    #    例: step_0436.png
                    file_name = f"step_{step:04d}.png"

                    # 4. 最終的なファイルパス
                    img_path = os.path.join(target_dir, file_name)

                    # torchvision.utils.save_image を使って保存
                    vutil.save_image(decoded_img[i], img_path)
                #print(f"save figure")
        return img

    @torch.no_grad
    def onestep_sampling(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,
               noise_sigma,
               noise_sigma_predict,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,

               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               starttimestep = None,
               breakstep = 1,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        alpha_bar_u = 1/(1 + noise_sigma_predict)
        print(f"ddim.py, alpha_bar_u = {alpha_bar_u}")
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps *= 1 # ここでタイムステップをいじくる
        #torch.clamp(start_timesteps, 0, S)
        if starttimestep != None:
            start_timesteps = torch.full_like(start_timesteps, starttimestep)
        # alphas
        #indiceからタイムステップ
        print(f"ddim.py start_timesteps after clamp S = {S} : {start_timesteps}")
        results = []
        img = x_T.to(device)
        maxind = start_timesteps.max().item()
        time_range = reversed(range(0, maxind))
        iterator = tqdm(time_range, desc='My Sampling')
        cnt = 0
        for i, step in enumerate(iterator):

            # 現在のステップが、バッチ内の各画像の開始タイムステップ以下であるかどうかのマスクを作成
            # これにより、まだサンプリングが始まっていない画像をスキップできる
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)

            # ts (タイムステップ) はバッチ全体分を用意
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # indexはスケジュール配列を参照するためのもの。現在のループ回数から計算
            # self.ddim_timestepsはmake_scheduleで作成される長さSの配列
            # stepに対応するindexを探す
            time_idx_tensor = torch.where(torch.from_numpy(self.ddim_timesteps).to(device) == step)[0]
            if time_idx_tensor.numel() == 0:
                # もし現在のstepがスケジュールになければスキップ (補間なども可能)
                continue
            index = time_idx_tensor.item()


            # p_sample_ddimをバッチ全体に対して一度だけ呼び出す
            outs = self.p_sample_ddim(img, conditioning, ts, index, # 修正: indexを追加
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning)
            img_prev, pred_x0 = outs
            cnt += 1
            # active_maskがTrueの画像だけを、計算結果で更新する
            img = torch.where(active_mask, img_prev, img)
            print(f"ddim.py , onestep sampling complete ,step = {step}, index = {index}, cnt = {cnt}, breakstep = {breakstep}")
            if cnt == breakstep:
                break

        return img

    @torch.no_grad
    def MIMO_decide_starttimestep_ddim_sampling(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               intermediate_path = None,
               intermediate_skip = 1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               starttimestep = 200,
               noise_variance = None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device

        #torch.clamp(start_timesteps, 0, S)
        if noise_variance == None:
            noise_variance = 1.0
        alpha_bar_u = 1/(1 + noise_variance)
        print(f"ddim.py, alpha_bar_u = {alpha_bar_u}")
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        if starttimestep != None and starttimestep>=0:
            start_timesteps = torch.zeros(batch_size, dtype=torch.long).to(device)
            start_timesteps = torch.full_like(start_timesteps, starttimestep).to(device)

        # alphas
        #indiceからタイムステップ
        print(f"ddim.py start_timesteps after clamp S = {S} : {start_timesteps}")
        results = []
        img = x_T.to(device)
        maxind = start_timesteps.max().item()
        time_range = reversed(range(0, maxind))
        iterator = tqdm(time_range, desc='My Sampling')
        for i, step in enumerate(iterator):
            # 現在のステップが、バッチ内の各画像の開始タイムステップ以下であるかどうかのマスクを作成
            # これにより、まだサンプリングが始まっていない画像をスキップできる
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1).to(device)

            # ts (タイムステップ) はバッチ全体分を用意
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # indexはスケジュール配列を参照するためのもの。現在のループ回数から計算
            # self.ddim_timestepsはmake_scheduleで作成される長さSの配列
            # stepに対応するindexを探す
            time_idx_tensor = torch.where(torch.from_numpy(self.ddim_timesteps).to(device) == step)[0]
            if time_idx_tensor.numel() == 0:
                # もし現在のstepがスケジュールになければスキップ (補間なども可能)
                continue
            index = time_idx_tensor.item()


            # p_sample_ddimをバッチ全体に対して一度だけ呼び出す
            outs = self.p_sample_ddim(img, conditioning, ts, index, # 修正: indexを追加
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning)
            img_prev, pred_x0 = outs

            # active_maskがTrueの画像だけを、計算結果で更新する
            img = torch.where(active_mask, img_prev, img)
            #print(f"step = {step}, index = {index}")

            if intermediate_path != None and index % intermediate_skip == 0:
                #TODO 途中結果を保存

                decoded_img = self.model.decode_first_stage(pred_x0)

                # VAEのデコーダ出力は [-1, 1] の範囲なので、[0, 1] にスケーリングする
                # decoded_img = (decoded_img + 1.0) / 2.0
                # decoded_img = torch.clamp(decoded_img, min=0.0, max=1.0)

                # バッチ内の各画像を個別に保存
                batch_size = decoded_img.shape[0]
                for i in range(batch_size):
                    # ファイル名をステップ番号と画像インデックスで一意に決定
                    # 例: /path/to/intermediate/step_0180_img_00.png
                    target_dir = os.path.join(intermediate_path, str(snr), str(i), str(starttimestep))

                    # 2. ディレクトリが存在しなければ再帰的に作成 (exist_ok=True)
                    os.makedirs(target_dir, exist_ok=True)

                    # 3. ファイル名を決定 (ステップ番号)
                    #    例: step_0436.png
                    file_name = f"step_{step:04d}.png"

                    # 4. 最終的なファイルパス
                    img_path = os.path.join(target_dir, file_name)

                    # torchvision.utils.save_image を使って保存
                    vutil.save_image(decoded_img[i], img_path)
                #print(f"save figure")

        return img


    @torch.no_grad
    def decide_starttimestep_ddim_sampling(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,
               noise_sigma,
               noise_sigma_predict,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               intermediate_path = None,
               intermediate_skip = 1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               starttimestep = None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        alpha_bar_u = 1/(1 + noise_sigma_predict)
        print(f"ddim.py, alpha_bar_u = {alpha_bar_u}")
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps *= 1 # ここでタイムステップをいじくる
        #torch.clamp(start_timesteps, 0, S)
        if starttimestep != None:
            start_timesteps = torch.full_like(start_timesteps, starttimestep)

        # alphas
        #indiceからタイムステップ
        print(f"ddim.py start_timesteps after clamp S = {S} : {start_timesteps}")
        results = []
        img = x_T.to(device)
        maxind = start_timesteps.max().item()
        time_range = reversed(range(0, maxind))
        iterator = tqdm(time_range, desc='My Sampling')
        for i, step in enumerate(iterator):
            # 現在のステップが、バッチ内の各画像の開始タイムステップ以下であるかどうかのマスクを作成
            # これにより、まだサンプリングが始まっていない画像をスキップできる
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)

            # ts (タイムステップ) はバッチ全体分を用意
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # indexはスケジュール配列を参照するためのもの。現在のループ回数から計算
            # self.ddim_timestepsはmake_scheduleで作成される長さSの配列
            # stepに対応するindexを探す
            time_idx_tensor = torch.where(torch.from_numpy(self.ddim_timesteps).to(device) == step)[0]
            if time_idx_tensor.numel() == 0:
                # もし現在のstepがスケジュールになければスキップ (補間なども可能)
                continue
            index = time_idx_tensor.item()


            # p_sample_ddimをバッチ全体に対して一度だけ呼び出す
            outs = self.p_sample_ddim(img, conditioning, ts, index, # 修正: indexを追加
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning)
            img_prev, pred_x0 = outs

            # active_maskがTrueの画像だけを、計算結果で更新する
            img = torch.where(active_mask, img_prev, img)
            #print(f"step = {step}, index = {index}")

            if intermediate_path != None and index % intermediate_skip == 0:
                #TODO 途中結果を保存

                decoded_img = self.model.decode_first_stage(pred_x0)

                # VAEのデコーダ出力は [-1, 1] の範囲なので、[0, 1] にスケーリングする
                # decoded_img = (decoded_img + 1.0) / 2.0
                # decoded_img = torch.clamp(decoded_img, min=0.0, max=1.0)

                # バッチ内の各画像を個別に保存
                batch_size = decoded_img.shape[0]
                for i in range(batch_size):
                    # ファイル名をステップ番号と画像インデックスで一意に決定
                    # 例: /path/to/intermediate/step_0180_img_00.png
                    target_dir = os.path.join(intermediate_path, str(snr), str(i), str(starttimestep))

                    # 2. ディレクトリが存在しなければ再帰的に作成 (exist_ok=True)
                    os.makedirs(target_dir, exist_ok=True)

                    # 3. ファイル名を決定 (ステップ番号)
                    #    例: step_0436.png
                    file_name = f"step_{step:04d}.png"

                    # 4. 最終的なファイルパス
                    img_path = os.path.join(target_dir, file_name)

                    # torchvision.utils.save_image を使って保存
                    vutil.save_image(decoded_img[i], img_path)
                #print(f"save figure")

        return img

    @torch.no_grad
    def observe_ddim(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,
               noise_sigma,
               noise_sigma_predict,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               intermediate_path = None,
               intermediate_skip = 1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               starttimestep = None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        alpha_bar_u = 1/(1 + noise_sigma_predict)
        print(f"ddim.py, alpha_bar_u = {alpha_bar_u}")
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps *= 1 # ここでタイムステップをいじくる
        #torch.clamp(start_timesteps, 0, S)
        if starttimestep != None:
            start_timesteps = torch.full_like(start_timesteps, starttimestep)

        # alphas
        #indiceからタイムステップ
        print(f"ddim.py start_timesteps after clamp S = {S} : {start_timesteps}")
        results = []
        img = x_T.to(device)
        maxind = start_timesteps.max().item()
        time_range = reversed(range(0, maxind))
        iterator = tqdm(time_range, desc='My Sampling')
        past_img = {}
        for i, step in enumerate(iterator):
            # 現在のステップが、バッチ内の各画像の開始タイムステップ以下であるかどうかのマスクを作成
            # これにより、まだサンプリングが始まっていない画像をスキップできる
            active_mask = (start_timesteps >= step).view(-1, 1, 1, 1)

            # ts (タイムステップ) はバッチ全体分を用意
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # indexはスケジュール配列を参照するためのもの。現在のループ回数から計算
            # self.ddim_timestepsはmake_scheduleで作成される長さSの配列
            # stepに対応するindexを探す
            time_idx_tensor = torch.where(torch.from_numpy(self.ddim_timesteps).to(device) == step)[0]
            if time_idx_tensor.numel() == 0:
                # もし現在のstepがスケジュールになければスキップ (補間なども可能)
                continue
            index = time_idx_tensor.item()


            # p_sample_ddimをバッチ全体に対して一度だけ呼び出す
            outs = self.p_sample_ddim(img, conditioning, ts, index, # 修正: indexを追加
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning)
            img_prev, pred_x0 = outs

            # active_maskがTrueの画像だけを、計算結果で更新する
            img = torch.where(active_mask, img_prev, img)
            #print(f"step = {step}, index = {index}")
            past_img[(step, index)] = img


        return past_img

    @torch.no_grad
    def search_timestep(self,
               S, #ddim_num_steps 200
               batch_size,
               shape,
               noise_sigma,
               noise_sigma_predict,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               intermediate_path = None,
               intermediate_skip = 1,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               snr = None,
               starttimestep = None,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        print(f"ddim.py alphas_cumprod = {self.alphas_cumprod.shape}") #1000
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.model.betas.device
        alpha_bar_u = 1/(1 + noise_sigma_predict)
        print(f"ddim.py, alpha_bar_u = {alpha_bar_u}")
        alpha_minus = -self.alphas_cumprod
        start_timesteps = torch.searchsorted(alpha_minus, -alpha_bar_u)
        start_timesteps *= 1 # ここでタイムステップをいじくる
        #torch.clamp(start_timesteps, 0, S)
        if starttimestep != None:
            start_timesteps = torch.full_like(start_timesteps, starttimestep)


        return start_timesteps