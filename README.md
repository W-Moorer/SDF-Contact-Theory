# SDF-Mesh 接触力学理论验证框架

这是一个面向 **SDF-Mesh 接触区域划分与接触力分布验证** 的 Python 项目。它覆盖以下完整链路：

```text
解析几何闭合网格生成 → SDF 生成 → 三线性/三次插值 → 接触区域划分 → 接触力积分 → 可视化与指标报告
```

框架包含 5 类接触工况：

1. 环状接触：圆环 Torus 与立方体；
2. 多点接触：空心圆柱与立方体；
3. 多点/多区域接触：锥体与立方体；
4. 多区域接触：波浪底面立方体与平面；
5. 多区域接触：波浪底面立方体与有限立方体。

默认生成的闭合网格面片数均可超过 2000。`--quick` 模式会降低 SDF 网格分辨率以便快速验证代码路径，但仍保留高细分网格生成能力。

---

## 1. 环境安装

### CPU / 可视化基础环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### RTX 3070 GPU 推荐环境

RTX 3070 属于 Ampere 架构，建议使用 CUDA 12.x。GPU 路径支持两种后端：

```bash
# Warp BVH 最近点查询后端，推荐用于 Mesh→SDF
pip install warp-lang

# CuPy RawKernel 后端，提供直接 CUDA brute-force 备选
pip install cupy-cuda12x
```

运行时用：

```bash
python scripts/run_all.py --sdf-source mesh --backend warp --resolution 128 --device cuda:0 --out outputs_gpu
```

如果没有安装 GPU 依赖，程序会自动回退到 NumPy/Numba CPU 路径。

---

## 2. 快速运行

为了验证项目是否安装成功，可先跑低分辨率 smoke test：

```bash
python scripts/run_all.py --quick --sdf-source analytic --out outputs_quick
```

生成结果在：

```text
outputs_quick/
├── index.html
├── summary.json
├── torus_cube_ring/
├── hollow_cylinder_cube_four_point/
├── cone_cube_four_patch/
├── wavy_cube_plane_multi_region/
└── wavy_cube_cube_clipped_multi_region/
```

每个 case 会包含：

```text
mesh_quality.json
sdf_metrics.json
contact_linear.json
contact_cubic.json
force_linear.json
force_cubic.json
sdf_slices.png
contact_linear.html
contact_cubic.html
force_linear.html
force_cubic.html
```

---

## 3. 主运行模式

### 解析 SDF 模式

适合理论验证，因为被动物体 SDF 是解析真值：

```bash
python scripts/run_all.py --sdf-source analytic --resolution 96 --out outputs_analytic
```

### Mesh SDF 模式：Warp 推荐

适合验证 Mesh→SDF 生成链路：

```bash
python scripts/run_all.py --sdf-source mesh --backend warp --resolution 128 --device cuda:0 --out outputs_mesh_warp
```

### Mesh SDF 模式：CuPy 备选

CuPy 后端当前为 brute-force triangle distance，适合中小分辨率验证：

```bash
python scripts/run_all.py --sdf-source mesh --backend cupy --resolution 64 --device cuda:0 --out outputs_mesh_cupy
```

### Mesh SDF 模式：CPU/Numba 备选

CPU brute-force 只建议用于小分辨率：

```bash
python scripts/run_all.py --sdf-source mesh --backend cpu --resolution 32 --out outputs_mesh_cpu
```

---

## 4. 理论验证指标

框架自动输出：

- SDF 值域与近零比例；
- 解析 SDF 与采样 SDF 的 RMS / L∞ 误差；
- Eikonal 误差：`| ||∇φ|| - 1 |`；
- 三线性与三次插值的接触面积差；
- 接触区域连通分量数量；
- 每个接触分量的面积、平均深度、最大深度、合力；
- 接触总力、总力矩、线性/三次插值差异。

---

## 5. 代码结构

```text
sdf_contact/
├── geometry/      # 解析几何网格生成、网格质量检查
├── sdf/           # SDFGrid、解析SDF、三线性/三次插值、Mesh→SDF
├── contact/       # 接触区域划分、连通域、接触力
├── validation/    # SDF/接触/力学指标
├── visualization/ # SDF切片、接触区域、力箭头、HTML报告
├── experiments/   # 5个标准工况定义
└── backends/      # Warp/CuPy/Numba 后端
```

---

## 6. 当前实现边界

- `analytic` SDF 模式用于高可信理论验证；
- `warp` 后端用于真正 Mesh→SDF 最近点查询，依赖 `warp-lang`；
- `cupy` 与 `cpu` 后端提供 fallback，但未实现高性能 BVH，因此高分辨率时速度不如 Warp；
- 接触力模型默认是无摩擦法向 penalty：`p = k_n * penetration`；
- 三次插值采用 Catmull-Rom tricubic convolution，并提供局部采样值夹紧以降低过冲风险。

---

## 7. 推荐实验命令

RTX 3070 上建议从 128³ 开始：

```bash
python scripts/run_all.py --sdf-source mesh --backend warp --resolution 128 --device cuda:0 --kn 1e5 --out outputs_rtx3070_r128
```

精度对比：

```bash
python scripts/run_all.py --sdf-source analytic --resolution 64  --out outputs_r64
python scripts/run_all.py --sdf-source analytic --resolution 128 --out outputs_r128
python scripts/run_all.py --sdf-source analytic --resolution 192 --out outputs_r192
```

比较 `summary.json` 中的 SDF、接触面积、力分布收敛趋势。

---

## 8. 显式动力学 GPU/Numba 验证阶段

第二阶段新增了 `sdf_contact/dynamics/`，用于验证上一阶段的 SDF-Mesh 接触流形与 penalty 接触力是否能在显式动力学中产生合理的位移、速度、接触力和能量响应。

### 8.1 设计原则

上一阶段的部分静态算例是为了制造困难的接触拓扑，例如人为穿插的空心圆柱-立方体或锥体-立方体。这些算例适合测试接触区域划分，但并不都适合作为真实动力学场景。动力学阶段因此新增自由落体算例：

1. `dyn_cube_plane_freefall_baseline`：细分立方体自由落体到解析平面，作为解析撞击时间基准；
2. `dyn_torus_cube_freefall_ring`：圆环自由落体到固定立方体支撑面，形成环状压力带；
3. `dyn_wavy_cube_plane_freefall_multi_region`：波浪底面立方体自由落体到平面，多个波谷形成多区域接触；
4. `dyn_hollow_cylinder_cube_freefall_annular`：空心圆柱自由落体到立方体支撑面，验证空心圆柱几何的环状底面接触；
5. `dyn_cone_cube_freefall_apex`：尖端向下的圆锥自由落体到立方体支撑面，验证尖锐局部接触的时程力峰。

这些动力学算例沿用同一批解析几何模型、`SDFGrid.sample()` 双插值接口、SDF 符号约定与 penalty 接触力逻辑，但不再把非物理的静态穿插姿态作为动力学基准。

### 8.2 运行自由落体动力学验证

CPU/NumPy smoke test：

```bash
python scripts/run_dynamics.py \
  --quick \
  --resolution 32 \
  --steps 300 \
  --dynamics-backend numpy \
  --method both \
  --out outputs_dynamics_quick
```

RTX 3070 推荐命令：

```bash
python scripts/run_dynamics.py \
  --quick \
  --resolution 64 \
  --steps 600 \
  --dynamics-backend cuda \
  --method both \
  --out outputs_dynamics_cuda
```

更高质量 Mesh→SDF + CUDA 动力学：

```bash
python scripts/run_dynamics.py \
  --sdf-source mesh \
  --sdf-backend warp \
  --resolution 128 \
  --device cuda:0 \
  --dynamics-backend cuda \
  --method both \
  --out outputs_dynamics_warp_cuda
```

### 8.3 输出内容

每个动力学 case/method 会输出：

```text
time_history.npz          # 位移、速度、接触力、能量等时程序列
dynamics_result.json      # 关键事件、峰值、能量漂移、运行性能
response_curves.png       # 位移/速度/接触力/接触状态曲线
energy.png                # 动能、重力势能、接触弹性能、总能量
state_animation.html      # 接触状态时序动画
report.html               # 单 case HTML 报告
```

总报告为：

```text
outputs_dynamics_quick/index.html
```

### 8.4 GPU 实现说明

动力学层的快速接触求解器是 `SurfaceContactEvaluator`：

```text
active triangle surface quadrature points
        ↓
SDFGrid 三线性/三次插值采样
        ↓
penetration depth d=max(0,-phi)
        ↓
pressure p = kn*d^alpha + cn*max(-v·n,0)
        ↓
parallel force accumulation
        ↓
symplectic Euler explicit update
```

CUDA 路径使用 Numba CUDA：

- 每个 active surface quadrature point 对应一个 CUDA thread；
- 支持三线性和 Catmull-Rom 三次插值；
- 输出每个采样点的 `phi/depth/normal/pressure/force/contact energy`；
- 当前版本采用 host-side reduction，优先保证稳定性和可调试性；
- CPU fallback 支持 NumPy 与 Numba CPU。

### 8.5 动力学验证指标

动力学报告自动标注：

- 首次接触时间；
- 解析自由落体首次接触时间；
- 首次接触误差；
- 最大压缩时刻；
- 离地时刻；
- 最大接触力；
- 最大穿透深度；
- 最大接触面积；
- 动能、重力势能、接触弹性能和总能量漂移。

注意：默认加入法向阻尼 `cn` 以稳定快速验证，因此总机械能会下降；如果需要检查无阻尼 penalty 系统的能量守恒，可设置 `--cn 0` 并减小 `--dt`。
