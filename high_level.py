#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Macro‑level DP for optimal ρ(t) compensation
Author : <your‑name>
Date   : 2025‑08‑03
"""
import zipfile, argparse, io, os, math
import numpy as np
import pandas as pd

# -------------------------------------------------
# 1. Data loader
# -------------------------------------------------
def load_dataset(zip_paths):
    dfs = []
    for path in zip_paths:
        with zipfile.ZipFile(path, 'r') as zf:
            for name in zf.namelist():
                raw = io.StringIO(zf.read(name).decode('utf‑8'))
                df  = pd.read_csv(raw, header=None,
                                  names=['veh_id','frame','s','lane','clip'])
                df['src_zip'] = os.path.basename(path)
                dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    return data

# -------------------------------------------------
# 2. compute headway & density
# -------------------------------------------------
def compute_headway_density(df, veh_len=4.5):
    """
    对同一 zip 内的数据按 frame+lane 排序，
    计算相邻纵向车辆间距 (含车长) -> h_t；
    密度 k = N / seg_len
    """
    results = []
    for (frame,lane), grp in df.groupby(['frame','lane']):
        grp = grp.sort_values('s')
        s_vals = grp['s'].values
        # headway: 前车尾到后车头间距
        delta_s = np.diff(s_vals) - veh_len
        # 末车头距路段末端可略过；只记后车角度
        for idx, h in enumerate(delta_s):
            n_id = grp.iloc[idx+1]['veh_id']
            results.append({'frame':frame,'veh_id':n_id,'lane':lane,'h_t':h})
        # density
        seg_len = s_vals.max() - s_vals.min() + veh_len
        k = len(s_vals) / seg_len if seg_len>0 else np.nan
        results.append({'frame':frame,'veh_id':np.nan,
                        'lane':lane,'h_t':np.nan,'k':k})
    res_df = pd.DataFrame(results)
    return res_df

# -------------------------------------------------
# 3. Steady‑state headway from S3 diagram
# -------------------------------------------------
def steady_state_headway(k, veh_len=4.5, d0=2.0):
    """h_ss ≈ 1/k  – 车长 + 安全距; 也可替换为论文中的解析式"""
    return 1.0/np.maximum(k,1e-5) - veh_len + d0

def s3_speed(k, vf, kc, m):
    return vf / (1.0 + (k/kc)**m)**(2.0/m)

# -------------------------------------------------
# 4. DP core
# -------------------------------------------------
def s3_macro_dp(k_series, h_t_series, params):
    # unpack
    vf,kc,m                = params['vf'],params['kc'],params['m']
    dt,Nk,Nrho             = params['dt'],params['Nk'],params['Nrho']
    w_v,w_k,w_rho          = params['w_v'],params['w_k'],params['w_rho']
    alpha,beta,theta,ΔH_t  = params['alpha'],params['beta'],params['theta'],params['ΔH_t']
    k_min,k_max            = np.nanmin(k_series)*0.5, np.nanmax(k_series)*2
    # grids
    K_grid  = np.linspace(k_min, k_max, Nk)
    R_grid  = np.linspace(0, 1, Nrho)
    T       = len(k_series)
    J       = np.full((T+1, Nk), np.inf)
    Pi      = np.full((T,   Nk), -1, dtype=int)

    # terminal cost
    h_ss_T  = steady_state_headway(k_series[-1])
    v_T     = s3_speed(k_series[-1], vf,kc,m)
    for i,k in enumerate(K_grid):
        J[-1,i] = w_v*(s3_speed(k,vf,kc,m)-v_T)**2 + \
                  w_k*(k-k_series[-1])**2

    # backward DP
    for t in range(T-1,-1,-1):
        k_ref = k_series[t]; h_t = h_t_series[t]
        h_ss  = steady_state_headway(k_ref)
        v_ss  = s3_speed(k_ref,vf,kc,m)
        for i,k in enumerate(K_grid):
            v_k = s3_speed(k,vf,kc,m)
            for r_idx,ρ in enumerate(R_grid):
                # immediate cost
                l_cost = w_v*(v_k-v_ss)**2 + w_k*(k-k_ref)**2 + w_rho*ρ**2
                # density transition
                Δk_lc  = beta*ρ*ΔH_t*theta
                q      = k*v_k
                q_eq   = (kc*vf)/(2**(2/m))
                Δk_rel = alpha*(q-q_eq)
                k_next = np.clip(k + (Δk_lc+Δk_rel)*dt, k_min, k_max)
                jnxt   = np.interp(k_next, K_grid, J[t+1])
                cost   = l_cost + jnxt
                if cost < J[t,i]:
                    J[t,i]  = cost
                    Pi[t,i] = r_idx
    # forward pass
    ρ_star = []
    k_curr = k_series[0]
    for t in range(T):
        i = int(np.argmin(abs(K_grid-k_curr)))
        r_idx = Pi[t,i]
        ρ = R_grid[r_idx]
        ρ_star.append(ρ)
        # state update
        Δk_lc  = beta*ρ*ΔH_t*theta
        v_k    = s3_speed(k_curr,vf,kc,m)
        q      = k_curr*v_k
        q_eq   = (kc*vf)/(2**(2/m))
        Δk_rel = alpha*(q-q_eq)
        k_curr = np.clip(k_curr + (Δk_lc+Δk_rel)*dt, k_min, k_max)
    return np.array(ρ_star)

# -------------------------------------------------
# 5. Main
# -------------------------------------------------
def main(args):
    raw_df = load_dataset(args.zip_paths)
    hd_df  = compute_headway_density(raw_df)
    # 这里只做单车道示例；多车道可分 lane group 后循环
    k_series = hd_df.loc[hd_df['k'].notna(),'k'].values
    h_t_series = hd_df.loc[hd_df['h_t'].notna(),'h_t'].values
    params = dict(vf=args.vf,kc=args.kc,m=args.m,dt=0.1,Nk=41,Nrho=21,
                  w_v=1.0,w_k=1.0,w_rho=0.1,alpha=‑0.05,beta=0.02,
                  theta=0.3,ΔH_t=np.mean(h_t_series))
    ρ_star = s3_macro_dp(k_series,h_t_series,params)
    out = pd.DataFrame({'t':np.arange(len(ρ_star)),'rho_star':ρ_star})
    save_path = '/mnt/data/rho_star.csv'
    out.to_csv(save_path,index=False)
    print(f'[DONE] ρ*(t) saved to {save_path}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip_paths', nargs='+', required=True)
    ap.add_argument('--vf',  type=float, default=33.33) # m/s ≈120 km/h
    ap.add_argument('--kc',  type=float, default=0.04)  # veh/m
    ap.add_argument('--m',   type=float, default=2.5)
    args = ap.parse_args()
    main(args)
