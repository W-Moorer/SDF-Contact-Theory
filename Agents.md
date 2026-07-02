# Agents.md — SDF-Mesh 接触力学验证框架接管说明

## 0. 当前阶段结论

本仓库已经完成两个阶段：

1. **静态 SDF-Mesh 接触流形验证阶段**：解析几何闭合网格、SDF 生成、三线性/三次插值、接触区域划分、接触力积分、SDF/接触/力分布可视化。
2. **显式动力学验证阶段**：新增自由落体动力学算例、显式积分器、NumPy/Numba/CUDA 接触力并行求解器、位移/速度/接触力时程曲线、能量图与接触状态动画。

重要说明：上一阶段的若干静态工况是为了测试复杂接触拓扑而人为构造的穿插姿态，未必是物理合理的动力学场景。动力学阶段已经改用自由落体场景验证方法在时域中的正确性。

---

## 1. 不要修改的核心理论接口

后续 Agent 需要严格保持以下接口兼容：

```python
SDFGrid.sample(points, method="linear" | "cubic", return_grad=True)
extract_contact_region(active_mesh, passive_sdf_grid, method="linear" | "cubic", ...)
compute_contact_forces(contact_triangles, kn, alpha, center_of_mass)
```

符号约定仍为：

```text
phi > 0: passive object outside
phi = 0: passive object boundary
phi < 0: passive object interior
penetration depth d = max(0, -phi)
normal n = normalize(grad(phi))
pressure p = kn * d^alpha
```

动力学层只新增更快的 surface quadrature evaluator，不替代静态阶段的精确 clipped manifold extractor。

---

## 2. 新增动力学模块

新增目录：

```text
sdf_contact/dynamics/
├── scenarios.py        # 自由落体动力学工况定义
├── force_evaluator.py  # NumPy/Numba/CUDA surface quadrature 接触力求解器
├── integrator.py       # 显式积分器、事件检测、结果保存/加载
├── runner.py           # 批量运行入口
├── visualization.py    # 响应曲线、能量图、状态动画、HTML 报告
└── __init__.py
```

新增脚本：

```text
scripts/run_dynamics.py             # 运行显式动力学验证
scripts/render_dynamics_outputs.py  # 从已保存的 npz/json 重新渲染图和报告
```

新增测试：

```text
tests/test_dynamics_freefall.py
```

---

## 3. 动力学自由落体算例

`build_dynamics_cases(quick=True|False)` 生成 5 个物理上更合理的自由落体工况：

| case | 目的 | 几何来源 |
|---|---|---|
| `dyn_cube_plane_freefall_baseline` | 解析撞击时间基准 | subdivided box + plane slab |
| `dyn_torus_cube_freefall_ring` | 环状接触压力带 | torus + cube support |
| `dyn_wavy_cube_plane_freefall_multi_region` | 多波谷/多区域接触 | wavy-bottom box + plane slab |
| `dyn_hollow_cylinder_cube_freefall_annular` | 空心圆柱底部环状接触 | hollow cylinder + cube support |
| `dyn_cone_cube_freefall_apex` | 尖锐局部接触 | cone apex-down + cube support |

`quick=True` 下所有 active/passive 网格面片数仍然不低于 2000。

---

## 4. 显式积分器

当前积分器位于：

```python
sdf_contact.dynamics.integrator.simulate_case
```

形式为 symplectic Euler：

```text
v_{n+1} = v_n + dt * (F_contact + m*g) / m
x_{n+1} = x_n + dt * v_{n+1}
```

目前只开放平动自由度，旋转冻结。这样做是为了隔离验证：

```text
SDF sampling → penetration → normal → pressure → contact force → z response
```

下一阶段如要加入转动，建议保留本平动版本作为回归基准，再引入四元数姿态、惯量张量和角速度更新。

---

## 5. GPU/Numba 接触力实现细节

`SurfaceContactEvaluator` 是动力学阶段的快速接触力求解器。它使用四点三角面面积求积：

```text
triangle vertices + triangle centroid, each with area/4
```

原因：

- 仅用三角形重心会漏掉尖端和波谷的首次接触；
- 加入三个顶点后，圆锥尖端、波浪底面波谷、立方体边角的接触事件能在正确时间附近出现；
- 仍保持简单并行结构，便于 CUDA thread-per-sample 实现。

CUDA 路径位于 `force_evaluator.py`：

```python
_surface_forces_cuda_kernel
```

特点：

- 每个 quadrature point 对应一个 CUDA thread；
- 支持 `linear` 和 `cubic` 两种 SDF 插值；
- cubic 使用 Catmull-Rom 权重，局部 clamp 与静态插值逻辑一致；
- 输出每点 `phi, depth, normal, pressure, force, contact elastic energy`；
- 当前版本在 host 端 reduction，便于调试；RTX 3070 上后续可改为 CUDA block reduction。

后端选择：

```text
--dynamics-backend cuda       # 强制 Numba CUDA，适合 RTX 3070
--dynamics-backend auto       # 有 CUDA 则 CUDA，否则 NumPy/Numba fallback
--dynamics-backend numpy      # 本沙箱/无 GPU smoke test 推荐
--dynamics-backend numba      # CPU Numba fallback，当前对小批量循环不一定比 NumPy 快
```

---

## 6. 推荐运行命令

CPU smoke test：

```bash
PYTHONPATH=. python scripts/run_dynamics.py \
  --quick \
  --resolution 32 \
  --steps 300 \
  --dynamics-backend numpy \
  --method both \
  --out outputs_dynamics_quick
```

RTX 3070 动力学验证：

```bash
PYTHONPATH=. python scripts/run_dynamics.py \
  --quick \
  --resolution 64 \
  --steps 600 \
  --dynamics-backend cuda \
  --method both \
  --out outputs_dynamics_cuda
```

Mesh→SDF + CUDA 动力学验证：

```bash
PYTHONPATH=. python scripts/run_dynamics.py \
  --sdf-source mesh \
  --sdf-backend warp \
  --resolution 128 \
  --device cuda:0 \
  --dynamics-backend cuda \
  --method both \
  --out outputs_dynamics_warp_cuda
```

如果已经有 `time_history.npz` 与 `dynamics_result.json`，可重新生成图和 HTML：

```bash
PYTHONPATH=. python scripts/render_dynamics_outputs.py \
  --quick \
  --out outputs_dynamics_quick \
  --methods linear,cubic
```

---

## 7. 本次沙箱基准结果

当前执行环境没有 RTX 3070/CUDA，因此实际运行的是 NumPy fallback。命令等价于：

```bash
PYTHONPATH=. python scripts/run_dynamics.py \
  --quick --resolution 32 --steps 300 \
  --dynamics-backend numpy --method both
```

结果摘要：

| case | method | first contact [s] | analytic [s] | max Fz [N] | max depth [m] | energy drift |
|---|---|---:|---:|---:|---:|---:|
| cube-plane | linear | 0.1424 | 0.142784 | 278.336 | 0.004461 | -0.1563 |
| cube-plane | cubic | 0.1424 | 0.142784 | 278.336 | 0.004461 | -0.1563 |
| torus-cube | linear | 0.1424 | 0.142784 | 165.972 | 0.007553 | -0.1540 |
| torus-cube | cubic | 0.1424 | 0.142784 | 165.972 | 0.007553 | -0.1540 |
| wavy-plane | linear | 0.1656 | 0.165900 | 225.040 | 0.022620 | -0.07645 |
| wavy-plane | cubic | 0.1656 | 0.165900 | 225.040 | 0.022620 | -0.07645 |
| hollow-cylinder-cube | linear | 0.1424 | 0.142784 | 275.983 | 0.003497 | -0.1627 |
| hollow-cylinder-cube | cubic | 0.1424 | 0.142784 | 278.511 | 0.003535 | -0.1600 |
| cone-cube | linear | 0.1424 | 0.142784 | 40.022 | 0.086482 | -0.02640 |
| cone-cube | cubic | 0.1424 | 0.142784 | 40.397 | 0.089581 | -0.02546 |

解释：

- 首次接触误差约为一个时间步 `dt=0.0008 s` 以内，说明四点 surface quadrature 已能捕获自由落体撞击事件。
- linear/cubic 在这些解析平面/盒体支撑 case 中结果非常接近；这符合预期，因为被动 SDF 在接触附近接近平面或盒面，插值误差较小。
- 能量漂移为负，主要来自默认法向阻尼 `cn=120`。若要做无阻尼能量守恒检验，应使用 `--cn 0 --dt 2e-4` 或更小时间步。
- 当前未在沙箱中获得 RTX 3070 CUDA 性能数据；后续 Agent 在真实机器上应追加 `outputs_dynamics_cuda/summary.json` 中的 `runtime_seconds` 与 `steps_per_second`。

---

## 8. 调试要点

1. **首次接触晚于解析时间很多**：检查 quadrature 是否只用了 centroid。当前代码已使用 vertices+centroid 四点求积。
2. **能量下降**：默认有阻尼 `cn`，不是能量守恒错误。设 `--cn 0` 再检查。
3. **最大穿透过大**：减小 `dt` 或增大 `kn`；显式 penalty 接触稳定性对 `dt*sqrt(kn/m)` 敏感。
4. **CUDA 后端报错**：先确认 `numba.cuda.is_available()`，再确认 NVIDIA driver/CUDA runtime 与 Numba 版本兼容。
5. **Numba CPU 比 NumPy 慢**：小规模 quick case 中可能出现，因 Python 每步调用 kernel 的 overhead 较明显。RTX 上应使用 `--dynamics-backend cuda`。
6. **状态动画 HTML 过大**：`save_dynamic_animation_html` 默认限制 10 帧、每帧最多 120 个接触采样点；可继续调低。
7. **静态精确流形与动态快速力略有差异**：动态阶段为显式积分速度采用 surface quadrature，静态阶段的 `extract_contact_region` 仍是精确 clipped manifold 验证路径。

---

## 9. 后续优化建议

1. CUDA block-level reduction，避免每步把所有 per-point forces 拷回 host。
2. 加入旋转自由度、惯量张量、四元数姿态更新和力矩耦合。
3. 将 `extract_contact_region` 的候选筛选和局部裁剪迁移到 CUDA/Warp，形成真正的动态 clipped manifold。
4. 对高刚度 penalty 引入自适应时间步或显式稳定性估计。
5. 增加 friction/tangential damping，但保持当前无摩擦法向版本作为回归基准。
