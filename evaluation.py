#!/usr/bin/env python3
# evaluation.py  ——  reproduce Fig.4 (q‑k & v‑t) for DP + RL result
#
# 使用条件：
#   1. 已存在 macro_trajectory.npz （由 macro_dp.py 保存）
#   2. 已训练并保存 cav_acc_control_model.zip （由 micro_ddpg.py 保存）
#
# 主要输出：
#   Figure 4(a)  q‑k 曲线，标出 q_ss, q_p, q_lc 三点
#   Figure 4(b)  v‑t 曲线，标出 v_SS, v_p, v_LC 与 t_LC/t_SS
#
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import argparse
from stable_baselines3 import TD3

# -------------------- 参数 --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--lanes', type=int, default=1,
                    help='车道数 n（用于密度 k = veh/nL）')
parser.add_argument('--road_len', type=float, default=1000.0,
                    help='评估路段长度 L (m)')
args = parser.parse_args()
n_lane = args.lanes
road_len = args.road_len

# -------------------- 数据载入 -----------------
data = np.load('macro_trajectory.npz')
t      = data['t']
lead_y = data['veh_lead_pos']
lead_v = data['veh_lead_speed']
foll_y = data['veh_follow_pos']
foll_v = data['veh_follow_speed']
dt     = float(data['dt'])

# ----------------- 回放 RL 策略 ---------------
from micro_ddpg import CarFollowingEnv  # 直接复用已定义的环境
# 只用 leader 轨迹驱动环境
env = CarFollowingEnv(lead_y, lead_v, len(t))
model = TD3.load('cav_acc_control_model', env=env)

obs = env.reset()
rl_headway = []
rl_speed   = []
for _ in range(len(t)-1):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)
    rl_headway.append(obs[0])
    rl_speed.append(obs[1])
    if done:
        break
rl_headway = np.array(rl_headway)
rl_speed   = np.array(rl_speed)
rl_t       = t[:len(rl_speed)]

# ----------- 计算 k‑q 点（SS / p / LC）----------
# 选取：
#   * SS：t=0 附近的稳态
#   * LC：RL 轨迹中头距第一次降至 < d0+0.5 m 的时间
#   * p：t_LC 与 t_SS 中点
d0  = 2.0
idx_ss = 5                        # 前 0.5 s 当作 steady‑state 起点
idx_lc = np.where(rl_headway < d0 + 0.5)[0][0]   # 进入并道扰动
idx_p  = (idx_ss + idx_lc) // 2

def flow_density(pos, speed, idx):
    k = 1.0 / (rl_headway[idx] / n_lane)            # veh / m / lane
    q = k * speed[idx]                              # veh / s / lane
    return k, q

k_ss, q_ss = flow_density(foll_y, rl_speed, idx_ss)
k_lc, q_lc = flow_density(foll_y, rl_speed, idx_lc)
k_p , q_p  = flow_density(foll_y, rl_speed, idx_p )

# ------------- 绘制 Fig. 4(a)  q‑k ---------------
k_curve = np.linspace(0.0001, 0.5, 500)            # 简单 S3 曲线示例
# S3：  v = v_f / (1 + (k/k_c)^m)^(2/m)
v_f = 16.67; k_c = 0.08; m = 4
v_curve = v_f / (1 + (k_curve/k_c)**m)**(2/m)
q_curve = k_curve * v_curve

plt.figure(figsize=(6,5))
plt.plot(k_curve, q_curve, 'k', label='q‑k curve')
plt.scatter([k_ss, k_p, k_lc], [q_ss, q_p, q_lc],
            color=['green', 'blue', 'red'])
plt.annotate('q_SS', (k_ss, q_ss), textcoords='offset points', xytext=(-25,5))
plt.annotate('q_p',  (k_p,  q_p),  textcoords='offset points', xytext=(0,5))
plt.annotate('q_LC', (k_lc, q_lc), textcoords='offset points', xytext=(5,-15))
plt.xlabel('k  (veh/m/lane)')
plt.ylabel('q  (veh/s/lane)')
plt.title('Fig 4(a)  Proportional headway adjustment on q‑k')
plt.grid(alpha=0.3)

# ------------- 绘制 Fig. 4(b)  v‑t ----------------
v_ss = rl_speed[idx_ss]
v_lc = rl_speed[idx_lc]
v_p  = rl_speed[idx_p]
t_lc = rl_t[idx_lc]
t_ss = rl_t[-1]

plt.figure(figsize=(6,5))
plt.plot(rl_t, rl_speed, 'k', linewidth=1.5, label='ego speed')
plt.hlines([v_ss, v_p, v_lc], xmin=0, xmax=[t_ss, t_lc, t_lc],
           linestyles=['dotted','dashed','dotted'],
           colors=['green','blue','red'])
plt.vlines([t_lc, t_ss], ymin=0, ymax=[v_lc, v_ss],
           linestyles=['dotted','dotted'], colors=['red','green'])
plt.legend()
plt.xlabel('t  (s)')
plt.ylabel('v  (m/s)')
plt.title('Fig 4(b)  Speed transition (v‑t)')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
