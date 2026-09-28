<div align="center">

# 🔥 PyTorch Compilation Deep Dive

### How PyTorch *Actually* Compiles Your Code

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-7c4dff?style=for-the-badge)](LICENSE)

> **Eager Mode → Compiled Graphs → Kernel Fusion**  
> Understand the full journey from `model.forward(x)` to blazing-fast GPU kernels — explained simply.

🌐 **[View Interactive Version (Animated)](https://shivanshinigam.github.io/pytorch-compilation-guide/README.html)**

</div>

---

## 📚 Table of Contents

| # | Topic | TL;DR |
|---|-------|-------|
| [01](#-section-01--eager-mode) | 🐢 Eager Mode | Python calls CUDA one op at a time |
| [02](#-section-02--torchcompile) | ⚡ `torch.compile` | One line → 2-4× speedup |
| [03](#-section-03--torchdynamo) | 🦆 TorchDynamo | Traces Python bytecode into an FX Graph |
| [04](#-section-04--aot-autograd) | 🔬 AOT Autograd | Compiles forward + backward before training |
| [05](#-section-05--forward-computation-graph) | 📊 Forward Graph | DAG of ops from input → logits |
| [06](#-section-06--backward-graph--model-correction) | 🔄 Backward Graph | Gradients flow in reverse for learning |
| [07](#-section-07--kernel-fusion) | 🔀 Kernel Fusion | Multiple GPU ops → one kernel, zero intermediate memory |
| [08](#-section-08--cheat-sheet) | ✅ Cheat Sheet | Full comparison table |

---

## 🐢 Section 01 — Eager Mode

> **The Default PyTorch — "Run it now, no planning"**

When you write normal PyTorch code, you're in **eager mode**. Python runs line by line and every operation executes **immediately** on the GPU the moment Python hits that line.

```python
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 256)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        # ① Python → CUDA matmul fires RIGHT NOW
        x = self.linear1(x)
        # ② Python → CUDA relu fires RIGHT NOW
        x = self.relu(x)
        # ③ Python → CUDA matmul fires RIGHT NOW
        x = self.linear2(x)
        return x

model  = TinyNet().cuda()
x      = torch.randn(32, 512).cuda()
output = model(x)   # Python → CUDA → Python → CUDA → Python → CUDA ...
```

### How Eager Mode Executes

```
Python Interpreter
      │
      ▼
  PyTorch Op (linear1)  ──►  CUDA Kernel runs on GPU
      │                              │
      ◄──────────────────────────────┘  (returns result)
      │
  PyTorch Op (relu)     ──►  CUDA Kernel runs on GPU
      │                              │
      ◄──────────────────────────────┘
      │
  PyTorch Op (linear2)  ──►  CUDA Kernel runs on GPU
```

### Why This Becomes a Problem

| Property | Value |
|---|---|
| Python calls per GPU op | **1:1 — every single op** |
| Python overhead per op | **~microseconds (adds up fast)** |
| Cross-op optimization | **❌ Zero — GPU cannot see ahead** |
| Debuggability | **✅ Easy — print/inspect anything** |

> ⚠️ **The bottleneck:** For every operation, Python wakes up, schedules the CUDA call, and waits. In a 100-layer model that's 100+ round-trips. The GPU sits idle between calls.

---

## ⚡ Section 02 — `torch.compile`

> **One line of code. Massive speedup.**

Introduced in PyTorch 2.0, `torch.compile(model)` wraps your model so that instead of running Python eagerly, PyTorch **traces** what your code does and compiles it into an optimized computation graph — automatically.

```python
import torch

model  = TinyNet().cuda()
x      = torch.randn(32, 512).cuda()

# ──────────────────────────────────────────────────────────────
# THE MAGIC LINE
# ──────────────────────────────────────────────────────────────
compiled_model = torch.compile(model)

# First call: PyTorch TRACES & COMPILES (one-time warm-up cost)
output = compiled_model(x)

# All subsequent calls: runs the OPTIMIZED compiled graph ⚡
output = compiled_model(x)   # blazing fast!

# Choose your backend / optimization level:
compiled_model = torch.compile(model, backend="inductor")       # default
compiled_model = torch.compile(model, mode="reduce-overhead")   # minimize launch overhead
compiled_model = torch.compile(model, mode="max-autotune")      # maximum performance
```

### The 3-Stage Compilation Pipeline

```
Your Python Code
      │
      ▼
┌─────────────────┐
│  1. TorchDynamo │  ← Intercepts Python bytecode, captures FX Graph
└────────┬────────┘
         │  FX Graph
         ▼
┌──────────────────────┐
│  2. AOT Autograd     │  ← Derives backward graph ahead-of-time
└────────┬─────────────┘
         │  Joint fwd+bwd graph
         ▼
┌──────────────────────┐
│  3. TorchInductor    │  ← Generates fused Triton/CUDA kernels
└────────┬─────────────┘
         │
         ▼
     GPU 🚀  (Optimized execution)
```

---

## 🦆 Section 03 — TorchDynamo

> **Tracing Python Safely — The Hardest Part**

The hardest problem in compiling PyTorch wasn't making fast kernels — it was **tracing arbitrary Python code**. TorchDynamo solves this brilliantly.

**How it works:**
- Installs itself as a **CPython frame evaluation hook** — intercepts at the bytecode level
- Instead of running ops, it **records** them as nodes in an FX Graph
- Non-tensor Python code (math, prints) runs normally
- Records **guards** (assumptions about shapes/dtypes) for smart recompilation

```python
# ── Your original Python forward ──────────────────────────────
def forward(x, weight1, bias1, weight2, bias2):
    x = x @ weight1 + bias1     # linear
    x = torch.relu(x)           # activation
    x = x @ weight2 + bias2     # linear
    return x

# ── What TorchDynamo captures (FX Graph) ─────────────────────
#
#   graph(x, weight1, bias1, weight2, bias2):
#     mm_0   = aten.mm(x, weight1)       # matmul node
#     add_0  = aten.add(mm_0, bias1)     # add node
#     relu_0 = aten.relu(add_0)          # relu node
#     mm_1   = aten.mm(relu_0, weight2)  # matmul node
#     add_1  = aten.add(mm_1, bias2)     # add node
#     return add_1
#
# This FX Graph is a clean, compilable Python object ✓
```

> 💡 **Graph Breaks:** If Dynamo hits something untraceable, it *breaks the graph* at that point, runs that part eagerly, then starts a new graph. Your code **always works** — just might not be fully optimized at break points.

---

## 🔬 Section 04 — AOT Autograd

> **Ahead-of-Time: The Joint Forward + Backward Graph**

After Dynamo captures the forward graph, **AOT Autograd** expands it to include the backward pass — **before** the first training step even runs.

| | Eager Autograd | AOT Autograd |
|---|---|---|
| When backward is built | At runtime during `loss.backward()` | **At compile time, before training** |
| Recomputed every step? | **Yes, every step** | **No, compiled once** |
| Can be fused with forward? | ❌ No | ✅ Yes |
| Speedup | Baseline | **2-3× faster backward** |

```python
from functorch.compile import aot_function
import torch

def forward(x, w):
    return torch.relu(x @ w)

# AOT Autograd captures BOTH forward and backward
compiled_fn = aot_function(
    forward,
    fw_compiler=my_backend,   # receives compiled forward graph
    bw_compiler=my_backend,   # receives compiled backward graph
)

x = torch.randn(4, 4, requires_grad=True)
w = torch.randn(4, 4, requires_grad=True)

loss = compiled_fn(x, w).sum()
loss.backward()   # ← uses the COMPILED backward graph!
```

---

## 📊 Section 05 — Forward Computation Graph

> **The "Recipe" for Computing Predictions**

The forward graph is a **DAG (Directed Acyclic Graph)** — data flows in one direction only: input → transformations → output.

```
INPUT
  x : [32, 512]
      │
      ▼
  weight₁:[512,256] ──► MatMul ◄── bias₁:[256]
                            │
                            ▼
                    Add  →  x @ W₁ + b₁  →  [32, 256]
                            │
                            ▼
                    ReLU  →  max(0, x)   →  [32, 256]
                            │
                            ▼
  weight₂:[256,10] ──► MatMul ◄── bias₂:[10]
                            │
                            ▼
                    Add  →  x @ W₂ + b₂  →  [32, 10]
                            │
                            ▼
OUTPUT
  ŷ : Predictions [32, 10]
```

```python
# Inspect the captured FX Graph yourself:
exported = torch.export.export(model, (x,))
print(exported.graph)

# Or explain what compile will do:
explanation = torch._dynamo.explain(model)(x)
print(explanation.graph_count)   # 1 = ideal, >1 = graph breaks exist
```

> 🔑 Because it's a DAG (no cycles), the compiler can reason about dependencies, safely reorder operations, and fuse adjacent ops.

---

## 🔄 Section 06 — Backward Graph & Model Correction

> **How the Model Actually Learns**

The backward graph flows gradients *backwards* through the same operations — computing how much each weight contributed to the error.

### The Learning Loop

```
Input x  ──►  Forward Pass  ──►  Loss (ŷ vs y)
                                       │
                                       ▼
              Optimizer Step  ◄──  Backward Pass
              (w -= lr · ∂L/∂w)    (∂L/∂w for all w)
```

### Backward Graph — Gradients Flow Upstream

```
Loss (scalar)
      │  ∂L/∂logits
      ▼
  Add backward  →  passes gradient through
      │  ∂L/∂(xW₂)
      ├──────────────────────────────┐
      ▼                              ▼
  MatMul bwd                  ∂L/∂W₂ = xᵀ @ grad  ✓
  ∂L/∂x = grad @ W₂ᵀ
      │
      ▼
  ReLU backward  →  grad * (x > 0)  [zero out negatives]
      │
      ├──────────────────────────────┐
      ▼                              ▼
  MatMul bwd                  ∂L/∂W₁ = xᵀ @ grad  ✓
  ∂L/∂x = grad @ W₁ᵀ

All ∂L/∂w computed ✅  →  Optimizer updates all weights
```

```python
import torch, torch.nn as nn

model     = TinyNet().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

x      = torch.randn(32, 512).cuda()
labels = torch.randint(0, 10, (32,)).cuda()

optimizer.zero_grad()                    # 1. Clear old gradients
logits = model(x)                        # 2. FORWARD  — compute predictions
loss   = criterion(logits, labels)       # 3. LOSS     — how wrong?
loss.backward()                          # 4. BACKWARD — compute ∂L/∂w
optimizer.step()                         # 5. UPDATE   — w -= lr · ∂L/∂w

# With torch.compile, steps 2-4 become ONE compiled unit:
compiled_model = torch.compile(model)    # fwd + bwd both compiled ⚡
```

> ✅ In eager mode, the backward graph is **rebuilt every training step**. With `torch.compile`, it's compiled **once** and reused — delivering **2-4× speedup** on the backward pass alone.

---

## 🔀 Section 07 — Kernel Fusion

> **The Secret Weapon — Keeping Data in Registers**

Kernel fusion combines multiple GPU ops into a **single kernel** — eliminating expensive memory round-trips between ops.

### The Memory Bandwidth Problem

```
GPUs are fast at computing (FLOPS).
But they're often memory-bandwidth limited.

Every kernel launch:
  READ from HBM (slow) → compute → WRITE back to HBM (slow)
              ↑
      This round-trip is the bottleneck!
```

### Eager vs Fused — Side by Side

**❌ Eager Mode — 3 kernels, 6 memory accesses:**
```
Step 1:  [Read x]  →  x = x * 2.0  →  [Write x] 💾
Step 2:  [Read x]  →  x = relu(x)  →  [Write x] 💾
Step 3:  [Read x]  →  x = x + bias →  [Write x] 💾

Memory round-trips: ████████████████████████████████ 6 (slow 😓)
```

**✅ Fused — 1 kernel, 2 memory accesses:**
```
Step 1:  [Read x]  →  x*2 → relu → +bias (ALL IN REGISTERS)  →  [Write x] ✅

Memory round-trips: ████ 2 (fast 🚀)
```

### What TorchInductor Generates (Triton pseudocode)

```python
# What YOU write:
def fused_op(x, bias):
    return torch.relu(x * 2.0 + bias)

# What TorchInductor GENERATES:
#
# @triton.jit
# def fused_kernel(x_ptr, bias_ptr, out_ptr, n_elements):
#     pid     = tl.program_id(0)
#     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask    = offsets < n_elements
#
#     x    = tl.load(x_ptr + offsets, mask=mask)      # ONE load
#     bias = tl.load(bias_ptr + offsets, mask=mask)    # ONE load
#
#     # ALL OPS IN REGISTERS — no memory writes between!
#     result = tl.maximum(x * 2.0 + bias, 0.0)        # fused ✓
#
#     tl.store(out_ptr + offsets, result, mask=mask)   # ONE write
#
# Result: 3 ops → 1 kernel  |  6 memory accesses → 2  ✓

# See what Inductor generates yourself:
import torch._inductor.config as cfg
cfg.trace.enabled = True   # dumps Triton code to /tmp/torchinductor_*
compiled = torch.compile(fused_op)
compiled(x, bias)
```

### Types of Fusion TorchInductor Performs

| Fusion Type | Examples | Benefit |
|---|---|---|
| **Pointwise** | `relu`, `gelu`, `add bias`, `multiply scalar` | Multiple element-wise ops → 1 kernel |
| **Reduction** | `softmax`, `layer_norm`, `mean` | Avoids writing partial results to HBM |
| **Fwd + Bwd** | Activation + its gradient | Cross-boundary fusion via AOT Autograd |
| **Attention** | Flash Attention style | ~10× less memory, tile-by-tile computation |

### Impact Numbers

| Metric | Result |
|---|---|
| Typical inference speedup | **2–4×** |
| Memory savings (Flash Attention style) | **~10×** |
| Kernel launches | **1 (was N)** |
| Intermediate memory allocations | **~0** |

---

## ✅ Section 08 — Cheat Sheet

### Full Comparison Table

| Concept | What It Is | Analogy | Speed Impact |
|---|---|---|---|
| **Eager Mode** | Python runs each op immediately, 1 at a time | Reading recipe 1 sentence at a time | Baseline |
| **`torch.compile`** | Traces + compiles model into an optimized graph | Reading whole recipe, then cooking optimally | **2–4× faster** |
| **TorchDynamo** | Intercepts Python bytecode → captures FX Graph | Stenographer recording what you cook, not how you think | Enables compilation |
| **AOT Autograd** | Pre-computes fwd + bwd graphs before training | Planning recipe AND cleanup before starting | **Bwd 2–3× faster** |
| **Forward Graph** | DAG of ops: input → output | Flowchart from raw ingredients to finished dish | Compiled once |
| **Backward Graph** | Reverse graph: ∂loss/∂weight for every param | Finding which ingredient made the dish too salty | Compiled once |
| **Kernel Fusion** | Merges GPU ops; data stays in registers | Chopping + sautéing in one pan, no pan-switching | **Massive BW saving** |
| **TorchInductor** | Backend compiler → generates Triton/CUDA code | Master chef writing most efficient instructions | Enables fusion |

### The One Mental Model

```
Eager mode     = Python is the boss. Calls CUDA one op at a time.
                 Easy to debug. Hard to optimize at scale.

torch.compile  = Python steps aside. Compiler sees the whole
                 forward + backward graph and optimizes holistically.

Forward graph  = The "recipe" for computing predictions from inputs.

Backward graph = The "recipe" for figuring out how wrong each weight
                 was (chain rule, gradients flow backwards).

Kernel fusion  = Keep data in fast on-chip registers across multiple
                 ops, instead of bouncing to slow global memory between ops.

The pipeline:
  Dynamo (capture graph)
    → AOT Autograd (derive backward)
      → Inductor (fuse & generate kernels)
        → GPU 🚀
```

---

## 📖 Official Resources

| Resource | Link |
|---|---|
| `torch.compile` Tutorial | [pytorch.org/tutorials](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) |
| TorchDynamo Paper (2022) | [arxiv.org/abs/2206.01978](https://arxiv.org/abs/2206.01978) |
| TorchInductor Deep Dive | [dev-discuss.pytorch.org](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler/893) |
| FlashAttention Paper | [arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135) |

---

<div align="center">

*Built for learning — keep building, keep shipping 🚀*

</div>
