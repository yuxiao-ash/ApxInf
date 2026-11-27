"""Independent torch references for the 910B native library (requires torch_npu).

Usage: python test_fusions.py /path/to/libapxinf_910b_fusions.so
Runs on NPU:0; do not overlap other profiling or benchmark jobs.
"""
import ctypes
import json
import sys

import torch
import torch_npu  # noqa: F401

lib = ctypes.CDLL(sys.argv[1])
ptr, uint = ctypes.c_void_p, ctypes.c_uint32
lib.launch_pointwise.argtypes = [ptr] * 5 + [uint] * 3
lib.launch_rope.argtypes = [ptr] * 7 + [uint] * 5
lib.launch_adarms.argtypes = [ptr] * 4 + [uint, uint, ctypes.c_float]
for name in ("launch_pointwise", "launch_rope", "launch_adarms"):
    getattr(lib, name).restype = ctypes.c_int
stream = ptr(torch.npu.current_stream().npu_stream)

def p(t):
    return ptr(t.data_ptr())

def check(actual, reference, tolerance=0.001):
    torch.npu.synchronize()
    a, b = actual.float().cpu(), reference.float().cpu()
    relative = ((a-b).norm() / b.norm().clamp_min(1e-20)).item()
    assert torch.isfinite(a).all() and relative <= tolerance, relative
    return {"relative_l2": relative, "max_abs": (a-b).abs().max().item()}

results = []
for seed in range(3):
    torch.manual_seed(77+seed)
    for rows in (1, 50, 183):
        x = torch.randn(rows, 1024, device="npu", dtype=torch.bfloat16)
        style = torch.randn(3072, device="npu", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        out = torch.empty_like(x)
        ref = x * style[2048:] + residual
        torch.npu.synchronize()
        assert lib.launch_pointwise(stream, p(x), p(style), p(residual), p(out), rows, 1024, 1) == 0
        gate = check(out, ref, 0)
        norm = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)+1e-6)).bfloat16()
        ref = (norm + norm*style[:1024]) + style[1024:2048]
        torch.npu.synchronize()
        assert lib.launch_adarms(stream, p(x), p(style), p(out), rows, 1024, 1e-6) == 0
        ada = check(out, ref)
        for qh, kh in ((8, 1), (8, 4)):
            dim = 256
            qkv = torch.randn(rows, qh+2*kh, dim, device="npu", dtype=torch.bfloat16)
            angle = torch.randn(rows, 1, dim//2, device="npu")
            angle = torch.cat([angle, angle], -1)
            cos, sin = angle.cos().bfloat16(), angle.sin().bfloat16()
            q = torch.empty(rows, qh, dim, device="npu", dtype=torch.bfloat16)
            # Exercise shifted K/V destinations without corrupting prefix rows.
            k = torch.zeros(rows+1, kh, dim, device="npu", dtype=torch.bfloat16)
            v = torch.zeros_like(k)
            torch.npu.synchronize()
            assert lib.launch_rope(stream, p(qkv), p(cos), p(sin), p(q), p(k[1:]), p(v[1:]), rows, qh, kh, dim, 1) == 0
            for output, src in ((q, qkv[:, :qh]), (k[1:], qkv[:, qh:qh+kh])):
                rotated = torch.cat([-src[:, :, dim//2:].float(), src[:, :, :dim//2].float()], -1)
                check(output, (src.float()*cos.float()+rotated*sin.float()).bfloat16())
            check(v[1:], qkv[:, qh+kh:], 0)
            assert not k[0].count_nonzero() and not v[0].count_nonzero()
        results.append({"seed": seed, "rows": rows, "gate": gate, "adarms": ada})
print(json.dumps(results, indent=2))

# GeGLU writes only the final result. Reference keeps CANN's tanh formula;
# BF16 rounding can differ on rare boundary values, so compare independently.
lib.launch_geglu.argtypes = [ptr] * 3 + [uint] * 3
lib.launch_geglu.restype = ctypes.c_int
for seed in range(3):
    torch.manual_seed(211 + seed)
    for rows, cols in ((1, 4096), (50, 4096), (183, 16384), (695, 16384), (951, 16384)):
        x = torch.randn(rows, cols * 2, device="npu", dtype=torch.bfloat16) * 3
        # Include zero, saturation and values around GELU's nonlinear region.
        values = torch.tensor([-100., -20., -8., -3., -1., 0., 1., 3., 8., 20., 100.], device="npu", dtype=torch.bfloat16)
        x[:, cols:cols + len(values)] = values
        out = torch.empty(rows, cols, device="npu", dtype=torch.bfloat16)
        reference, _ = torch_npu.npu_geglu(x, dim=-1, approximate=1)
        torch.npu.synchronize()
        assert lib.launch_geglu(ptr(torch.npu.current_stream().npu_stream), p(x), p(out), rows, cols, 40) == 0
        result = check(out, reference, 0.001)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            assert lib.launch_geglu(ptr(torch.npu.current_stream().npu_stream), p(x), p(out), rows, cols, 40) == 0
        graph.replay()
        check(out, reference, 0.001)
        x.copy_(torch.randn_like(x))
        reference, _ = torch_npu.npu_geglu(x, dim=-1, approximate=1)
        graph.replay()
        check(out, reference, 0.001)
        print(json.dumps({"geglu_seed": seed, "rows": rows, "cols": cols, **result}))
