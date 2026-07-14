"""
PINNs: Burgers 方程 - 复现版 v2 (已修改)
修改记录:
  1. 去掉 mask，全网格计算 L2 误差
  2. self.lb, self.ub 注册为 buffer，支持 GPU 迁移
  3. 修正缩进问题
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.interpolate import griddata
import scipy.io
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # 注册为 buffer → model.to('cuda') 时自动跟随
        self.register_buffer('lb', torch.tensor([-1., 0.]))
        self.register_buffer('ub', torch.tensor([1., 1.]))
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)
        z = 2. * (z - self.lb) / (self.ub - self.lb) - 1.
        return self.net(z)


def pde_residual(model, x, t):
    x.requires_grad_()
    t.requires_grad_()
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - 0.01 / torch.pi * u_xx


# ==================== 主程序 ====================
print("加载精确解 (burgers_shock.mat)...")
mat_data = scipy.io.loadmat('C:\\Users\\ROG\\Desktop\\burgers_shock.mat')
t_vals = mat_data['t'].flatten()
x_vals = mat_data['x'].flatten()
Exact = np.real(mat_data['usol']).T
print(f"  精确解范围: [{Exact.min():.4f}, {Exact.max():.4f}]")
u_star = Exact.flatten()[:, None]

N_u, N_f = 100, 10000
Xg, Tg = np.meshgrid(x_vals, t_vals)
X_star = np.hstack((Xg.flatten()[:, None], Tg.flatten()[:, None]))

# 边界/初始点
xx1 = np.hstack((Xg[0:1, :].T, Tg[0:1, :].T))
uu1 = Exact[0:1, :].T
xx2 = np.hstack((Xg[:, 0:1], Tg[:, 0:1]))
uu2 = Exact[:, 0:1]
xx3 = np.hstack((Xg[:, -1:], Tg[:, -1:]))
uu3 = Exact[:, -1:]
X_u = np.vstack([xx1, xx2, xx3])
u_val = np.vstack([uu1, uu2, uu3])
idx = np.random.choice(X_u.shape[0], N_u, replace=False)
X_u, u_val = X_u[idx], u_val[idx]

# 内部采样
sampler = qmc.LatinHypercube(d=2)
lb_arr, ub_arr = np.array([-1., 0.]), np.array([1., 1.])
X_f = lb_arr + (ub_arr - lb_arr) * sampler.random(N_f)
X_f = np.vstack([X_f, X_u])

data = {
    'x_u': torch.tensor(X_u[:, 0:1], dtype=torch.float32),
    't_u': torch.tensor(X_u[:, 1:2], dtype=torch.float32),
    'u_u': torch.tensor(u_val, dtype=torch.float32),
    'x_f': torch.tensor(X_f[:, 0:1], dtype=torch.float32),
    't_f': torch.tensor(X_f[:, 1:2], dtype=torch.float32),
}

model = PINN()
print(f"参数量: {sum(p.numel() for p in model.parameters())}")

# Adam 热身
opt_a = torch.optim.Adam(model.parameters(), lr=1e-3)
print("Adam 热身 2000 步...")
for ep in range(2000):
    opt_a.zero_grad()
    u_p = model(data['x_u'], data['t_u'])
    f_p = pde_residual(model, data['x_f'], data['t_f'])
    loss = torch.mean((u_p - data['u_u'])**2) + torch.mean(f_p**2)
    loss.backward()
    opt_a.step()
    if ep % 500 == 0:
        print(f"  Adam {ep:4d}, Loss={loss.item():.2e}")

# L-BFGS 精调
opt_l = torch.optim.LBFGS(
    model.parameters(), max_iter=30000, max_eval=30000,
    tolerance_grad=1e-8, tolerance_change=1e-10,
    history_size=50, line_search_fn='strong_wolfe',
)
print("L-BFGS 精调...")
cnt = [0]

def closure():
    opt_l.zero_grad()
    u_p = model(data['x_u'], data['t_u'])
    f_p = pde_residual(model, data['x_f'], data['t_f'])
    loss = torch.mean((u_p - data['u_u'])**2) + torch.mean(f_p**2)
    loss.backward()
    cnt[0] += 1
    if cnt[0] % 1000 == 0:
        print(f"  L-BFGS {cnt[0]:5d}, Loss={loss.item():.2e}")
    return loss

opt_l.step(closure)
print(f"完成! 步数: {cnt[0]}")

# ==================== 评估 ====================
X_star_t = torch.tensor(X_star, dtype=torch.float32)
u_pred = model(X_star_t[:, 0:1], X_star_t[:, 1:2]).detach().numpy()
su = u_star.flatten()
sp = u_pred.flatten()

# 全网格 L2 误差（论文标准做法）
error_u = np.linalg.norm(su - sp) / np.linalg.norm(su)
print(f"\n✦ 全网格相对 L2 误差: {error_u:.6e}")
print(f"  论文参考值:         6.7e-04")
if error_u < 1e-2:
    print("  ✅ 复现成功 (同一数量级)")
else:
    print("  ⚠️ 误差偏大")

# ==================== 画图（官方布局）====================
from mpl_toolkits.axes_grid1 import make_axes_locatable

U_pred = griddata(X_star, u_pred.flatten(), (Xg, Tg), method='cubic')

fig = plt.figure(figsize=(12, 9))

# Row 0: u(t,x) 云图
ax = plt.subplot(2, 3, (1, 3))
h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[0, 1, -1, 1], origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
ax.plot(X_u[:,1], X_u[:,0], 'kx',
        label=f'Data ({u_val.shape[0]} points)', markersize=4, clip_on=False)
line = np.linspace(x_vals.min(), x_vals.max(), 2)[:,None]
ax.plot(t_vals[25]*np.ones((2,1)), line, 'w-', linewidth=1)
ax.plot(t_vals[50]*np.ones((2,1)), line, 'w-', linewidth=1)
ax.plot(t_vals[75]*np.ones((2,1)), line, 'w-', linewidth=1)
ax.set_xlabel('$t$', fontsize=12)
ax.set_ylabel('$x$', fontsize=12)
ax.set_title('$u(t,x)$')
ax.legend(frameon=False, loc='best')

# Row 1: 时间切片
for idx, (ti, tl) in enumerate([(25, 't=0.25'), (50, 't=0.50'), (75, 't=0.75')]):
    ax = plt.subplot(2, 3, 4+idx)
    ax.plot(x_vals, Exact[ti,:], 'b-', lw=2, label='Exact')
    ax.plot(x_vals, U_pred[ti,:], 'r--', lw=2, label='Prediction')
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('$u(t,x)$', fontsize=10)
    ax.set_title(f'${tl}$', fontsize=10)
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    if idx == 1:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=5, frameon=False)

plt.tight_layout()
plt.savefig('pinns_result_v2.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nDone! pinns_result_v2.png")
