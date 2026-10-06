# MP2 Part 3：Colab 操作与报告指南

本指南对应仓库根目录 `machine_project_2_part1.pdf` 第 1–4 页，重点是第 3–4 页的 Part 3。只涵盖 GPU Part 3。

## 1. 课程要求与本仓库对应关系

| 课程要求 | 实现/实验方式 |
| --- | --- |
| 手动 im2col，禁止调用 fold/unfold | `myconv.py` 使用 pad、切片、stack、reshape、permute；matmul 按 64 个输出位置分块 |
| PyTorch interpreter 在 GPU 运行 | benchmark 的 `eager`；原始 `myconv.py` 的独立示例仍是 CPU 正确性示例 |
| Torch Inductor 编译并在 GPU 运行 | benchmark 的 `inductor`，`torch.compile`，`fullgraph=True`，固定 shape |
| JAX JIT 编译并在 GPU 运行 | benchmark 的 `jax`，JIT lowering + XLA compilation；强制检查 backend 是 gpu |
| CUDA shared-memory tiled convolution | `myconv_kernel.cu`，8×8 输出 tile，reduction tile 为 16，shared memory 为 1024 bytes |
| 与内置 conv2d 核对正确性 | 每个配置与 `F.conv2d` 比较；CUDA 另测边界 tile、stride、padding、矩形 filter 和非默认 stream |
| 导出 traces 并用 Perfetto 查看 | 每组保存 PyTorch Profiler JSON；JAX 另保存原生 JSON.GZ |
| 四种版本的 GPU kernel 时间、host compilation 时间 | `summary.csv`、各组 `result.json` 与 `kernels.csv` |
| 改变 input/filter size，画 trace 中的 walltimes | 默认 3×3 个配置；自动生成输入大小和 filter 大小变化的 GPU span 图 |
| 根据低层优化解释，最后用单段总结 shape 对不同 compiler 的影响 | 见下面的证据检查与报告提纲；结论须基于你的实际 traces |

CUDA 接口没有 bias，故四种版本统一使用零 bias。输入、权重使用相同 seed=0；N=2、C_in=3、C_out=8、stride=1、padding=1、float32，不启用 AMP/TF32。内置 conv2d 只作为正确性 reference，不算第五个被测版本。

## 2. 把修改后的代码上传 Colab

建议直接打开本目录的 `part3_colab.ipynb`（Colab：File → Upload notebook）。Notebook 已包含以下所有关键步骤。

1. Runtime → Change runtime type → Hardware accelerator → **T4 GPU**。课程明确按免费版 T4 设计和评分。
2. 将整个更新后的 `gpu/` 目录传进去。可在本机仓库根目录执行 `zip -r /tmp/mp2_gpu.zip gpu`，然后在 notebook 的上传单元选择这个 ZIP；也可以 clone **你自己的、包含这些修改的仓库**。不要只 clone 课程 starter 后就开始测，它没有这些修复和脚本。
3. `%cd /content/gpu`；如果是 clone，改为实际的仓库 `gpu/` 路径。
4. 检查 GPU 和框架。**遵循 PDF，不升级、不重新安装 PyTorch/JAX/cuDNN。**

```python
!nvidia-smi
!nvcc --version
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # 在 import jax 前设置
import torch, jax
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('JAX:', jax.__version__, 'backend:', jax.default_backend(), 'devices:', jax.devices())
assert torch.cuda.is_available()
assert jax.default_backend() == 'gpu'
assert 'T4' in torch.cuda.get_device_name(0), '本次作业应选择 T4'
```

如果 CUDA/JAX 检查失败，先确认 runtime 类型，再重新连接干净的 T4 runtime。不要拿 CPU 数据当 GPU 数据，也不要为了修环境盲目 pip 升级课程框架。这里的 `nvcc --version` 是实际编译 toolkit 版本；`nvidia-smi` 中的 CUDA Version 是 driver 支持的版本，二者含义不同。extension 编译若报 toolkit 与 PyTorch 版本不匹配，保留错误信息，再处理具体版本问题。

## 3. 先跑小实验，再跑完整实验

先验证四种实现都能编译、校验并生成 GPU trace：

```python
!python benchmark_colab.py --output results_smoke --sizes 19 --filters 3 --warmup 3 --repeats 5 --profile-calls 3
```

必须看到 `All 4 experiments passed`。如果失败，查看 `results_smoke/failures.json` 和对应 `.log`；不要忽略失败直接写报告。

正式实验：

```python
!python benchmark_colab.py --output results_full
```

默认 H=W∈{19,33,65}、KH=KW∈{3,5,7}，共 9 个 shape × 4 种实现 = **36 组**。每组预热 10 次，端到端测量 30 次，trace 捕获 5 次。这些尺寸取自/扩展自 starter 的示例，**是本脚本的实验选择，不是课程规定的唯一尺寸**。

为测冷编译，每组使用新的 Python 进程及专属 cache；因此首次运行需要较长时间，CUDA 同一源码也会重新编译 9 次。不要在测试过程中同时运行其他 GPU cell。想测得更稳可以执行三次完整实验，分别使用 `results_run1`、`results_run2`、`results_run3`；这不是必需步骤。

输出目录若已存在，脚本会拒绝覆盖。重跑时换一个 `--output` 名称，保留前一轮原始数据。你也可以改尺寸：

```python
!python benchmark_colab.py --output results_larger --sizes 33 65 97 --filters 3 5 7
```

更大的输入会把更多 64-position matmul tiles 展开进编译图，从而可能显著增加编译时间；先完成默认实验再决定是否扩展。

## 4. 输出与计时含义

| 文件 | 用途 |
| --- | --- |
| `summary.csv` / `summary.json` | 所有成功配置的实测结果、计时 scope 与 trace 来源 |
| `experiment.json` / `failures.json` | 运行参数与失败实验清单 |
| `eager_h19_k3/result.json` 等 | 具体配置、软硬件版本、误差、原始 wall 样本和 GPU span 样本 |
| `*/torch_trace.json` | 四种版本都尝试用 PyTorch Profiler 捕获 CPU+CUDA 活动 |
| `jax_*/jax_native.json.gz`、`jax_*/jax_profile/` | JAX 原生 trace / XProf 数据 |
| `*/kernels.csv` | 每个 GPU kernel 的名字、次数、总时长和平均时长 |
| `inductor_*/fx_graph.py`、`inductor_*/cache/` | Dynamo FX 图与编译产物，辅助定位生成代码和 fusion |
| `inductor_*/dynamo_compile_times.txt` | Dynamo/Inductor 各编译阶段的诊断时间；**嵌套阶段不能直接相加** |
| `jax_*/stablehlo.txt`、`optimized_hlo.txt` | JAX 编译前后的 IR，可检查 fusion、layout 和 dot/GEMM lowering |
| `gpu_span_by_input.png` / `gpu_span_by_filter.png` | 来自 trace 的 GPU 执行跨度图，覆盖 input/filter 两种变化 |
| `gpu_kernels_by_input.png` | kernel duration 总和图 |
| `wall_by_input.png` | profiler 之外的同步端到端 walltime 图 |
| `compile_by_input.png` | host 编译/构建时间图；eager 为 N/A |
| `report_table.tex` | 可复制进 LaTeX 报告的实测表格（列较多，必要时缩放） |

### GPU 的三种时间不能混称

- `gpu_kernel_sum_us`：trace 中所有实际 GPU kernel duration 加总，再除以 inference 次数。不包含 host launch 间隙、H2D/D2H 拷贝。单 kernel 的时间见 `kernels.csv`。
- `gpu_span_mean_us`：每次 inference 从第一个 GPU kernel 开始到最后一个 GPU kernel 结束的跨度，含 kernel 间的空隙。这是图中 **GPU span / trace walltime** 的定义。多 stream 的 kernel 若重叠，kernel duration 之和可能大于 span；不要把这两列当成同一指标。
- `steady_wall_median_us`：预热后，Python 调用到同步完成的 `perf_counter` walltime 中位数，包括 dispatch/launch 和等待开销；不包含输入上传、正确性检查和编译。JAX 使用 `block_until_ready()`，PyTorch/CUDA 使用 `torch.cuda.synchronize()`。

Profiler 会扰动执行，尤其是这里的小 tensor。报告的 trace span 由 profiler 测得；不带 profiler 的同步 walltime 单独保存，不能假设两者相等。5 次 trace 调用逐次同步，脚本在每个 host annotation 的时间窗口内求 GPU span，不把不同 inference 之间的等待空隙算进去。

### Host compilation 的口径

| 版本 | `host_compile_s` 的含义 |
| --- | --- |
| PyTorch eager | N/A：没有模型编译阶段。框架启动、CUDA 初始化不算模型 compilation |
| Inductor | 计时 backend 编译函数本身；不含 Dynamo graph capture、wrapper 建立及首次执行。首次同步调用的总时长另见 `first_call_s`，含 graph capture + compile + execution；wrapper 耗时另记 |
| JAX JIT | `lower()` 的 tracing/lowering 时间 + `compile()` 的 XLA compile 时间；两个子阶段分别保存。不包含 compiled executable 的首次执行 |
| CUDA | 冷 cache 下 extension `load()` 的 build、link、load walltime，不含 kernel 执行；同一源码不随 tensor shape 专门重新生成代码 |

这些 scope 不完全相同，图和报告应注明。**不要只计时 `torch.compile(model)` 或 `jax.jit(fn)` 的返回速度来声称测到编译时间**；真正工作会延迟发生。也不要把 Inductor `first_call_s` 当成纯 compilation time，或把 JAX/CUDA 的首次执行时间误算成 compilation。

### JAX 与 PyTorch Profiler 的课程表述

PDF 要求三种框架均经 PyTorch Profiler 导出 traces，本脚本确实对 JAX 调用也套了 PyTorch Profiler。不过 JAX 使用独立 runtime，当前 Kineto/CUPTI 组合未必能记录它的 GPU kernel。脚本同时保存 JAX 原生 trace：若 PyTorch trace 中没有 GPU kernels，就使用原生 trace 计算，且在 `trace_source` 标明来源。

如果任一 trace 都没有 GPU kernels，或者无法定位每次 inference 的 GPU 时间窗口，实验会失败而不是给出虚假的 GPU 数字。提交时保留两种 JAX traces，并在报告说明实际来源；如果老师严格要求只能使用 PyTorch Profiler，需要向课程方说明此 runtime 的捕获限制。

## 5. 在 Perfetto 中检查证据

1. 下载某个代表配置的 `torch_trace.json`。JAX 如走原生 fallback，就下载对应 `jax_native.json.gz`。
2. 打开 https://ui.perfetto.dev/ → Open trace file。选择同一 shape 的四种 trace，例如 `*_h33_k5`。
3. 搜索 `eager_conv_0`、`inductor_conv_0`、`jax_conv_0` 或 `cuda_conv_0`；放大一次 inference 的 CPU/GPU tracks。不要把首次编译与 steady-state trace 混在一起：这里保存的运行 trace 都是预热后的。
4. 点 GPU kernel，查看 Duration 和 kernel name；结合 `kernels.csv` 核对每次调用的 kernel 数量和主要耗时来源。脚本保留 CPU 活动，以便查看 aten、launch、Torch-Compiled Region 等 host 调用。
5. 截图至少覆盖一个小配置和一个较大配置；截图标注形状、版本、GPU 时间单位。报告可引用 kernel name、数量和 generated code/IR 作为优化证据。

建议检查以下现象，**它们是待验证的解释，不是已测得的结论**：

| 版本 | 在 traces / IR / 生成代码中检查什么 | 可以据实讨论的影响 |
| --- | --- | --- |
| eager | pad、stack、copy/transpose、多个 tile matmul、cat、bias add 的调用与 GPU launches | 显式中间张量、内存流量、launch 数量；小输入可能主要受 launch 开销影响 |
| Inductor | fused/generated Triton kernels、extern/cuBLAS GEMM、kernel 数量、cache 下生成 `.py` 源码 | elementwise/data-movement fusion、shape specialization、布局处理和 buffer reuse 是否实际发生；不能假定所有 matmul 都融合成一个 kernel |
| JAX | optimized HLO 的 fusion/dot/custom-call，GPU kernels 的数量和名字 | XLA fusion、静态 shape/循环展开、layout 与 GEMM lowering；IR 有 fusion 不等于整个卷积只有一次 GPU launch |
| CUDA | `gemm_gpu_o4_kernel` 是否每次调用一次；shared-memory tile 的协同加载和 barriers | implicit im2col 省去显式 patch tensor，tile 重用降低重复 global loads；手写标量 FP32 kernel 没有显式使用 Tensor Cores，未必比 compiler+库 GEMM 快 |

本实现每个 CUDA block 有 64 threads；8×8 输出 tile 沿 reduction 维度以 16 为单位加载 input/weight，共用 shared memory。kernel 最终写回一次输出。边界元素置零并让所有线程经过 barriers。CUDA 实现中的 tile 与 PyTorch/JAX 的 64-position matmul tile 不是同一个硬件调度方案，比较的是本次四种实现，而不是语言的普遍优劣。

## 6. Part III 报告怎么写

建议按以下顺序写进 `report/main.tex` 的 Part III（运行前无法填真实数字）：

1. **Implementation**：解释 patches 从 `[N,C,KH*KW,out_h,out_w]` 变为 `[N,L,C*KH*KW]`，weight 变为 `[C_out,C*KH*KW]`，按 64 个 output positions 做 matmul，再恢复 NCHW；说明没有 fold/unfold。CUDA 说明 implicit im2col 与 shared-memory tile。
2. **Setup/correctness**：填写实际 T4、Python/PyTorch/JAX/CUDA 版本；说明测试的 N/C/stride/pad/dtype、零 bias、seed、预热与采样次数，列正确性误差和 atol=rtol=1e-4。不要复制 starter 中 JAX 的较松 atol=0.1 作为唯一 correctness 证据。
3. **GPU time plots**：放 `gpu_span_by_input.png` 和 `gpu_span_by_filter.png`；给 caption 明确 span 定义及平均次数。再结合 kernel sum 与 kernel count 解释 span 中的 launch gaps。
4. **Host compilation**：放编译图或实测表格；写清四种 scope、eager N/A、Inductor 首次调用与 backend compilation 的区别、CUDA 相同源码的重复冷构建。
5. **Optimization evidence**：至少引用同一 shape 的四份 traces，给出关键 kernel 名字/数量、IR 或生成代码中确实出现的优化；讨论哪个操作减少了，如何影响时间。
6. **Single-paragraph summary**：用一段话总结小/大 input 和不同 filter 下性能排名如何变化；只用本次数据，不预设 Inductor/JAX/CUDA 必然最快。

英文段落框架（替换方括号；不可把占位符当最终结论）：

> On the Colab T4, we compared four FP32 convolution implementations using identical inputs and weights, zero bias, stride 1, and padding 1. For small inputs, [measured ranking and time] coincided with [observed kernel counts or launch gaps]. As input size increased, [measured trend] was consistent with [specific trace/IR evidence]. Increasing the filter size changed the reduction dimension from C_in·3² to C_in·7² and resulted in [measured trend]. Inductor and JAX incurred [measured compilation scopes/times] while [measured steady-state benefit]; the CUDA implementation showed [measured performance] with one shared-memory kernel per convolution. These results indicate [conclusion limited to the tested shapes and implementations].

若想补一句编译是否值得：在同一 shape 的 steady-state walltime 上，用 `host_compile_s / (eager_wall_s - compiled_wall_s)` 粗估摊销调用次数；仅当 compiled 更快且 scope 已解释清楚时才有意义。

## 7. 下载与提交

Colab 会清空临时文件。完成后打包结果和源码，并从 notebook 下载：

```python
!zip -r /content/mp2_part3_results.zip results_full -x '*/cache/*'
!zip -r /content/mp2_part3_sources.zip . -x 'results_*/*' '*/__pycache__/*' '.git/*'
from google.colab import files
files.download('/content/mp2_part3_results.zip')
files.download('/content/mp2_part3_sources.zip')
```

结果包保留 traces、图、IR、编译诊断、日志和原始 JSON，不包含体积较大的编译 cache。如果你在报告中引用 Inductor 生成代码，先把有关 `.py` 文件复制到结果目录的单独文件夹，再打包，或另外下载对应 cache 文件。

把选定图复制到本地 `report/fig/`，将真实数据和观察填写到 Part III。PDF 的 hand-in 说明表示具体提交方式会另行通知；最后按课程公告提交代码和报告。

## 官方参考

- 作业 PDF 是运行环境与提交要求的依据；其中明确要求保留 Colab 自带框架版本。
- PyTorch Profiler：https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Inductor profiling/compile diagnostics：https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html
- JAX lowering/compilation：https://docs.jax.dev/en/latest/aot.html
- JAX asynchronous benchmarking：https://docs.jax.dev/en/latest/benchmarking.html
- JAX profiling：https://docs.jax.dev/en/latest/profiling.html
- Perfetto UI：https://ui.perfetto.dev/
