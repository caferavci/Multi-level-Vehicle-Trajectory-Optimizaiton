# ---------- 依赖 ----------
import numpy as np
from scipy.io import loadmat, savemat

# ---------- 加载原始场景 ----------
MAT = 'lc_data_20s_withpoints.mat'   # 请确保已上传
IDX = 0
raw   = loadmat(MAT, struct_as_record=False, squeeze_me=True)
scene = raw['lc_data'][IDX]
pts   = float(raw['points'][IDX,1])

# ---------- 基本参数 ----------
L, dt = 200, 0.1
sub_v = scene.veh_s.v[:L]; sub_y = scene.veh_s.y[:L]
front_v = scene.veh_ft.v[:L]; front_y = scene.veh_ft.y[:L]
rear_y  = scene.veh_rt.y[:L]

h_ss, v_ss = pts/10.0, 6.0

# ---------- DP 离散空间 ----------
v_min,v_max,dv = 0.,15.,0.25
h_min,h_max,dh = 0.,48.,0.5
vg=np.arange(v_min,v_max+1e-9,dv); hg=np.arange(h_min,h_max+1e-9,dh)
Nv,Nh=len(vg),len(hg); rho_set=np.linspace(0,1,11)

Kh=Kv=1.0
wa=wv=wh=1.0; wρ=0.1
a_min,a_max = -3.5,3.5

# ---------- DP 向后递推 ----------
J=np.full((L,Nv,Nh),np.inf); π=np.full((L-1,Nv,Nh),-1,int)
for vi,v in enumerate(vg):
    for hi,h in enumerate(hg):
        J[-1,vi,hi] = wh*(h_ss-h)**2 + wv*(v_ss-v)**2

front_gap = front_y - rear_y
for k in range(L-2,-1,-1):
    for vi,v in enumerate(vg):
        for hi,h in enumerate(hg):
            best=np.inf;bu=-1
            for ui,rho in enumerate(rho_set):
                a=rho*Kh*(h-h_ss)+(1-rho)*Kv*(v_ss-v)
                a = np.clip(a,a_min,a_max)
                v2=np.clip(v+a*dt,v_min,v_max)
                h2=np.clip(h+(front_v[k]-v)*dt,h_min,h_max)
                if h2>front_gap[k]: continue
                vi2=int(round((v2-v_min)/dv)); hi2=int(round((h2-h_min)/dh))
                cost=wa*a*a+wv*(v_ss-v)**2+wh*(h_ss-h)**2+wρ*rho*rho
                tot = cost + J[k+1,vi2,hi2]
                if tot<best: best,bu=tot,ui
            J[k,vi,hi]=best; π[k,vi,hi]=bu

# ---------- rollout 得到优化轨迹 ----------
v,h = sub_v[0], front_y[0]-sub_y[0]
opt_v,opt_a=[],[]
for k in range(L-1):
    vi=int(round((v-v_min)/dv)); hi=int(round((h-h_min)/dh))
    rho=rho_set[π[k,vi,hi]]
    a=np.clip(rho*Kh*(h-h_ss)+(1-rho)*Kv*(v_ss-v),a_min,a_max)
    v=np.clip(v+a*dt,v_min,v_max); opt_v.append(v); opt_a.append(a)
    h=np.clip(h+(front_v[k]-v)*dt,h_min,h_max)
opt_v=np.array([sub_v[0],*opt_v])
x_opt = np.cumsum(opt_v*dt)+sub_y[0]

# ---------- 回写 ego 轨迹并保存 MAT ----------
scene.veh_s.y[:L] = x_opt
scene.veh_s.v[:L] = opt_v
scene.veh_s.a[:L] = np.r_[opt_a,[opt_a[-1]]]

savemat('scene_opt.mat', {'scene_opt': scene})
print('✅ scene_opt.mat 已保存（含 DP 优化后的 ego 轨迹）')
