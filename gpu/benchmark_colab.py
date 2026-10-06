#!/usr/bin/env python3
"""Colab T4 experiment. Each (shape, implementation) gets a fresh process/cache."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

VARIANTS = ('eager', 'inductor', 'jax', 'cuda')
HERE = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='results')
    p.add_argument('--sizes', type=int, nargs='+', default=[19, 33, 65])
    p.add_argument('--filters', type=int, nargs='+', default=[3, 5, 7])
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--in-channels', type=int, default=3)
    p.add_argument('--out-channels', type=int, default=8)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--padding', type=int, default=1)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--profile-calls', type=int, default=5)
    p.add_argument('--worker', choices=VARIANTS, help=argparse.SUPPRESS)
    p.add_argument('--size', type=int, help=argparse.SUPPRESS)
    p.add_argument('--filter', type=int, help=argparse.SUPPRESS)
    return p


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str) + '\n')


def worker(a):
    # Set before framework imports. Torch and JAX otherwise compete for GPU RAM.
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ.setdefault('MAX_JOBS', '2')
    directory = Path(a.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    cache = directory / 'cache'
    cache.mkdir()
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache / 'inductor')
    os.environ['TRITON_CACHE_DIR'] = str(cache / 'triton')
    os.environ['TORCH_EXTENSIONS_DIR'] = str(cache / 'extensions')
    os.environ['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '1'
    import numpy as np
    import torch
    import torch.nn.functional as F
    from myconv import ConvModel
    from trace_utils import summarize_trace

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Choose a T4 GPU runtime in Colab.')
    # Match FP32 precision across frameworks (no TF32/AMP).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.init()
    torch.cuda.synchronize()
    rng = np.random.default_rng(0)
    shape = (a.batch, a.in_channels, a.size, a.size)
    weight_shape = (a.out_channels, a.in_channels, a.filter, a.filter)
    x_np = rng.standard_normal(shape).astype(np.float32)
    w_np = rng.standard_normal(weight_shape).astype(np.float32)
    # CUDA starter's API has no bias, so ALL implementations use zero bias.
    b_np = np.zeros(a.out_channels, dtype=np.float32)
    x = torch.from_numpy(x_np).cuda()
    w = torch.from_numpy(w_np).cuda()
    model = ConvModel(a.size, a.size, a.in_channels, a.out_channels,
                      a.filter, a.stride, a.padding).cuda().eval()
    with torch.no_grad():
        model.weight.copy_(w)
        model.bias.zero_()
        reference = F.conv2d(x, w, None, stride=a.stride, padding=a.padding)
    torch.cuda.synchronize()
    result = {
        'variant': a.worker, 'N': a.batch, 'C_in': a.in_channels,
        'H': a.size, 'W': a.size, 'C_out': a.out_channels,
        'KH': a.filter, 'KW': a.filter, 'stride': a.stride, 'padding': a.padding,
        'dtype': 'float32', 'bias': 'zero', 'seed': 0,
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'torch_cuda': torch.version.cuda, 'python': sys.version,
        'host_compile_s': None, 'first_call_s': None,
        'warmup': a.warmup, 'repeats': a.repeats, 'profile_calls': a.profile_calls,
    }
    try:
        result['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
        result['nvcc'] = subprocess.check_output(['nvcc', '--version'], text=True)
    except (OSError, subprocess.CalledProcessError):
        pass

    with torch.no_grad():
        if a.worker == 'eager':
            run = lambda: model(x)
            sync = torch.cuda.synchronize
            result['compile_scope'] = 'N/A: eager mode does not compile this model'
        elif a.worker == 'cuda':
            from torch.utils.cpp_extension import load
            start = time.perf_counter()
            extension = load(name='mp2_conv', sources=[str(HERE / 'myconv_kernel.cu')], verbose=True)
            result['host_compile_s'] = time.perf_counter() - start
            result['compile_scope'] = 'cold extension build + link + load; independent of input shape'
            run = lambda: extension.conv_cuda(x, w, a.stride, a.padding)
            sync = torch.cuda.synchronize
            # Exercise batch/channel tile tails, K tails, stride, padding, rectangular filters.
            for n, ci, h, width, co, kh, kw, s, p in [
                    (2, 3, 19, 23, 11, 3, 5, 2, 1), (1, 4, 13, 17, 3, 6, 4, 1, 0)]:
                xx = torch.randn(n, ci, h, width, device='cuda')
                ww = torch.randn(co, ci, kh, kw, device='cuda')
                torch.testing.assert_close(extension.conv_cuda(xx, ww, s, p),
                                           F.conv2d(xx, ww, stride=s, padding=p),
                                           atol=1e-4, rtol=1e-4)
            # Confirm the extension respects PyTorch's current stream.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                xx = x + 0.25
                yy = extension.conv_cuda(xx, w, a.stride, a.padding)
                expected = F.conv2d(xx, w, stride=a.stride, padding=a.padding)
            stream.synchronize()
            torch.testing.assert_close(yy, expected, atol=1e-4, rtol=1e-4)
        elif a.worker == 'inductor':
            from torch._dynamo.backends.registry import lookup_backend
            backend = lookup_backend('inductor')
            compile_phases = []

            def timed_inductor(gm, inputs):
                (directory / 'fx_graph.py').write_text(gm.code)
                start = time.perf_counter()
                compiled = backend(gm, inputs)
                compile_phases.append(time.perf_counter() - start)
                return compiled

            setup_start = time.perf_counter()
            compiled_model = torch.compile(model, backend=timed_inductor, fullgraph=True, dynamic=False)
            result['compile_wrapper_s'] = time.perf_counter() - setup_start
            run = lambda: compiled_model(x)
            sync = torch.cuda.synchronize
            result['compile_scope'] = 'Inductor backend compilation; excludes Dynamo capture and first execution'
        else:
            import jax
            from myconv_jax import conv2d_manual_jax
            if jax.default_backend() != 'gpu':
                raise RuntimeError(f'JAX backend is {jax.default_backend()}, expected gpu.')
            jax.config.update('jax_default_matmul_precision', 'highest')
            jax.config.update('jax_enable_compilation_cache', False)
            result['jax'] = jax.__version__
            result['jax_devices'] = str(jax.devices())
            args = tuple(jax.device_put(v, jax.devices('gpu')[0]) for v in (x_np, w_np, b_np))
            for value in args:
                value.block_until_ready()
            # Shapes/stride/padding are static; arrays remain runtime inputs.
            fn = jax.jit(lambda xx, ww, bb: conv2d_manual_jax(
                xx, ww, bb, stride=a.stride, padding=a.padding))
            start = time.perf_counter()
            lowered = fn.lower(*args)
            result['jax_lower_s'] = time.perf_counter() - start
            (directory / 'stablehlo.txt').write_text(lowered.as_text())
            start = time.perf_counter()
            compiled = lowered.compile()
            result['jax_xla_compile_s'] = time.perf_counter() - start
            result['host_compile_s'] = result['jax_lower_s'] + result['jax_xla_compile_s']
            (directory / 'optimized_hlo.txt').write_text(compiled.as_text() or '')
            run = lambda: compiled(*args)
            # JAX uses its own streams: synchronize the returned JAX array below.
            sync = lambda: None
            result['compile_scope'] = 'JAX tracing/lowering + XLA compile; excludes execution'

        def complete():
            value = run()
            if a.worker == 'jax':
                value.block_until_ready()
            else:
                sync()
            return value

        sync()
        start = time.perf_counter()
        output = complete()
        result['first_call_s'] = time.perf_counter() - start
        if a.worker == 'inductor':
            result['host_compile_s'] = sum(compile_phases)
            from torch._dynamo.utils import compile_times
            (directory / 'dynamo_compile_times.txt').write_text(str(compile_times()))
        if a.worker == 'jax':
            actual = torch.from_numpy(np.asarray(output).copy())
        else:
            actual = output.detach().cpu()
        expected = reference.cpu()
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        result['correct'] = True
        result['max_abs_error'] = (actual - expected).abs().max().item()
        result['output_shape'] = list(actual.shape)
        for _ in range(a.warmup):
            complete()
        wall_us = []
        for _ in range(a.repeats):
            start = time.perf_counter()
            complete()
            wall_us.append((time.perf_counter() - start) * 1e6)
        result['steady_wall_median_us'] = statistics.median(wall_us)
        result['steady_wall_min_us'] = min(wall_us)
        result['steady_wall_p90_us'] = float(np.percentile(wall_us, 90))
        result['wall_samples_us'] = wall_us

        # Profile only warmed-up execution. CPU+CUDA tracks expose host launches.
        trace_path = directory / 'torch_trace.json'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            for i in range(a.profile_calls):
                with torch.profiler.record_function(f'{a.worker}_conv_{i}'):
                    complete()
        prof.export_chrome_trace(str(trace_path))
        (directory / 'torch_operators.txt').write_text(
            prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=100))
        if a.worker == 'jax':
            # Kineto may not capture kernels launched by another framework.
            # Preserve native JAX GPU trace too, and never substitute CPU events.
            import shutil
            native_dir = directory / 'jax_profile'
            with jax.profiler.trace(str(native_dir), create_perfetto_trace=True):
                for i in range(a.profile_calls):
                    with jax.profiler.TraceAnnotation(f'jax_conv_{i}'):
                        complete()
            native_paths = list(native_dir.rglob('perfetto_trace.json.gz'))
            if not native_paths:
                native_paths = list(native_dir.rglob('*.trace.json.gz'))
            if not native_paths:
                raise RuntimeError('JAX did not export a Chrome/Perfetto trace.')
            native_path = directory / 'jax_native.json.gz'
            shutil.copy2(native_paths[0], native_path)
            try:
                metrics = summarize_trace(trace_path, a.profile_calls, directory)
                result['trace_source'] = 'PyTorch profiler'
            except RuntimeError:
                metrics = summarize_trace(native_path, a.profile_calls, directory)
                result['trace_source'] = 'native JAX profiler (Kineto had no GPU kernels)'
        else:
            metrics = summarize_trace(trace_path, a.profile_calls, directory)
            result['trace_source'] = 'PyTorch profiler'
        result.update(metrics)
        # Identify each inference's GPU window using synchronized, non-overlapping
        # CPU annotations, then measure first-kernel start to last-kernel end.
        from trace_utils import kernel_events
        import gzip
        opener = gzip.open if result['trace'].endswith('.gz') else open
        with opener(result['trace'], 'rt') as f:
            events = json.load(f)['traceEvents']
        windows = [e for e in events if e.get('ph') == 'X'
                   and e.get('name', '').startswith(f'{a.worker}_conv_')]
        kernels = kernel_events(result['trace'])
        spans = []
        for window in windows:
            subset = [e for e in kernels if window['ts'] <= e['ts']
                      and e['ts'] + e['dur'] <= window['ts'] + window['dur']]
            if subset:
                spans.append(max(e['ts'] + e['dur'] for e in subset) - min(e['ts'] for e in subset))
        if len(spans) != a.profile_calls:
            raise RuntimeError('Cannot resolve per-call GPU windows; inspect trace before reporting timing.')
        result['gpu_span_mean_us'] = statistics.mean(spans)
        result['gpu_span_samples_us'] = spans
        if a.worker == 'inductor':
            result['inductor_compile_count'] = len(compile_phases)
            if len(compile_phases) != 1:
                raise RuntimeError('Unexpected recompilation during measurement; inspect compile logs.')
        save_json(directory / 'result.json', result)
        print(json.dumps({k: result[k] for k in ('variant', 'H', 'KH', 'correct',
              'host_compile_s', 'gpu_kernel_sum_us', 'gpu_span_mean_us', 'steady_wall_median_us')}, indent=2))


def main(a):
    if a.worker:
        worker(a)
        return
    output = Path(a.output).resolve()
    # Keep previous raw evidence; use a new output path for every experiment.
    output.mkdir(parents=True, exist_ok=False)
    records, errors = [], []
    save_json(output / 'experiment.json', vars(a))
    for size in a.sizes:
        for filt in a.filters:
            if size + 2 * a.padding < filt:
                raise ValueError('Filter is larger than padded input.')
            for variant in a.variants:
                directory = output / f'{variant}_h{size}_k{filt}'
                cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', variant,
                       '--size', str(size), '--filter', str(filt), '--output', str(directory)]
                for option in ('batch', 'in_channels', 'out_channels', 'stride', 'padding',
                               'warmup', 'repeats', 'profile_calls'):
                    cmd.extend(['--' + option.replace('_', '-'), str(getattr(a, option))])
                print(f'Running {variant}: H=W={size}, KH=KW={filt}', flush=True)
                with open(output / f'{directory.name}.log', 'w') as log:
                    completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    errors.append(directory.name)
                    print(f'FAILED: see {directory.name}.log', flush=True)
                else:
                    records.append(json.loads((directory / 'result.json').read_text()))
    fields = ['variant', 'H', 'KH', 'N', 'C_in', 'C_out', 'stride', 'padding', 'correct',
              'max_abs_error', 'host_compile_s', 'first_call_s', 'gpu_kernel_sum_us',
              'gpu_span_mean_us', 'kernels_per_call', 'steady_wall_median_us',
              'steady_wall_p90_us', 'trace_source', 'compile_scope']
    with open(output / 'summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    save_json(output / 'summary.json', records)
    save_json(output / 'failures.json', errors)
    if records:
        subprocess.run([sys.executable, str(HERE / 'plot_colab_results.py'), str(output)], check=True)
    if errors:
        raise SystemExit(f'{len(errors)} experiments failed; see failures.json and logs. Results are incomplete.')
    print(f'All {len(records)} experiments passed. Results: {output}')


if __name__ == '__main__':
    args = parser().parse_args()
    if min(args.warmup, args.repeats, args.profile_calls, args.batch,
           args.in_channels, args.out_channels, args.stride, *args.sizes, *args.filters) <= 0:
        raise SystemExit('Counts and dimensions must be positive.')
    if args.padding < 0:
        raise SystemExit('Padding must be nonnegative.')
    main(args)
