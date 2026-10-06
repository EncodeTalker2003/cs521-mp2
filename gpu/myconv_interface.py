import torch
from pathlib import Path
from torch.utils.cpp_extension import load

# Compile and load CUDA extension
conv_module = load(name="myconv",
                     sources=[str(Path(__file__).with_name("myconv_kernel.cu"))],
                     verbose=True)

# Input parameters
N, C_in, H, W = 4, 3, 25, 25
C_out, KH, KW = 4, 6, 6
stride, pad = 1, 1

# Allocate tensors
x = torch.randn(N, C_in, H, W, device="cuda", dtype=torch.float32)
w = torch.randn(C_out, C_in, KH, KW, device="cuda", dtype=torch.float32)

# Run o4 kernel
out_custom = conv_module.conv_cuda(x, w, stride, pad)

# Reference solution (PyTorch)
out_ref = torch.nn.functional.conv2d(x, w, stride=stride, padding=pad)

# Test shape and correctness
print("CUDA --- shape check:", out_custom.shape == out_ref.shape)
print("CUDA --- correctness check:", torch.allclose(out_custom, out_ref, atol=1e-4))
