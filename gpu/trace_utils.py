"""Read actual device kernels from PyTorch/JAX Chrome traces (times are us)."""
import gzip
import json
from collections import defaultdict
from pathlib import Path


def kernel_events(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        data = json.load(f)
    events = data['traceEvents'] if isinstance(data, dict) else data
    complete = [e for e in events if e.get('ph') == 'X' and e.get('dur', 0) > 0]
    kernels = [e for e in complete if 'kernel' in str(e.get('cat', '')).lower().split(',')]
    if kernels:
        return kernels
    # Native JAX/XProf traces use GPU process + Stream thread metadata.
    gpu_pids, streams = set(), set()
    for e in events:
        name = str(e.get('args', {}).get('name', ''))
        if e.get('ph') == 'M' and e.get('name') == 'process_name' and 'GPU' in name.upper():
            gpu_pids.add(e.get('pid'))
        if e.get('ph') == 'M' and e.get('name') == 'thread_name' and 'stream' in name.lower():
            streams.add((e.get('pid'), e.get('tid')))
    return [e for e in complete
            if e.get('pid') in gpu_pids and (e.get('pid'), e.get('tid')) in streams
            and not any(s in e.get('name', '').lower() for s in ('memcpy', 'memset'))]


def summarize_trace(path, calls, directory):
    kernels = kernel_events(path)
    if not kernels:
        raise RuntimeError(f'No GPU kernel events in {path}; do not report CPU timing as GPU timing.')
    totals = defaultdict(lambda: [0, 0.0])
    for e in kernels:
        totals[e['name']][0] += 1
        totals[e['name']][1] += e['dur']
    import csv
    with open(Path(directory) / 'kernels.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['kernel', 'count', 'total_us', 'mean_us'])
        for name, (count, duration) in sorted(totals.items(), key=lambda item: -item[1][1]):
            writer.writerow([name, count, duration, duration / count])
    # GPU spans are computed separately using per-call annotations in the benchmark.
    return {
        'gpu_kernel_sum_us': sum(e['dur'] for e in kernels) / calls,
        'kernels_per_call': len(kernels) / calls,
        'trace': str(path),
    }
