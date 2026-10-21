"""CPU semantic study and separately gated, CUDA-only SDPA backend experiment."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

import numpy as np
from lab.attention_reference import attention, prefix_mask
from lab.attention_failure import validate_case, verify_failure

SPEC = Path('configs/attention-backend-v1.json')
SOURCES = [str(SPEC), 'lab/attention_reference.py', 'lab/attention_failure.py', __file__]


def save(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def begin(output, mode):
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Commit a clean protocol before running the study')
    output.mkdir(parents=True, exist_ok=False)
    root = Path.cwd().resolve()
    hashes = {}
    for filename in SOURCES:
        source = Path(filename).resolve()
        relative = source.relative_to(root)
        dest = output / 'source' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
        hashes[str(relative)] = digest(source)
    info = dict(mode=mode, status='running', platform=platform.platform(),
                git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                source_sha256=hashes, cuda_performance_measured=False)
    save(output / 'run.json', info)
    return info


def torch_attention(torch, q, k, v, arm):
    """B,H,L,D -> B,H,L,D. Cached queries are the suffix of all available keys."""
    length, keys = q.shape[-2], k.shape[-2]
    if not 0 < length <= keys:
        raise ValueError('Invalid causal prefix lengths')
    if arm == 'eager_fp32':
        scores = q.float() @ k.float().transpose(-1, -2) / q.shape[-1] ** 0.5
        mask = torch.as_tensor(prefix_mask(length, keys), device=q.device)
        weights = scores.masked_fill(~mask, float('-inf')).softmax(dim=-1)
        return (weights @ v.float()).to(q.dtype)
    # Single-token decode can see ALL cached/current keys. Non-square
    # is_causal=True would incorrectly use an upper-left aligned triangle.
    mask = None
    if 1 < length < keys:
        mask = torch.as_tensor(prefix_mask(length, keys), device=q.device)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=length == keys)


def semantics(torch):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    rng = np.random.default_rng(20261008)
    rows = []
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for length, keys in [(7, 7), (1, 17), (3, 17)]:
            for width in (8, 32):
                for strided in (False, True):
                    for pattern in ('normal', 'zeros', 'large_logits'):
                        arrays = [rng.normal(size=(2, 2, n, width)).astype('float32')
                                  for n in (length, keys, keys)]
                        if pattern == 'zeros':
                            arrays[0].fill(0); arrays[1].fill(0)
                        if pattern == 'large_logits':
                            arrays[0] *= 20; arrays[1] *= 20
                        inputs = []
                        for array in arrays:
                            if strided:
                                backing = np.zeros((*array.shape[:-1], width * 2), dtype='float32')
                                backing[..., ::2] = array
                                array = backing[..., ::2]
                            inputs.append(torch.from_numpy(array))
                        reference = attention(*arrays)
                        errors = {}; normalized_errors = {}
                        for arm in ('eager_fp32', 'math'):
                            actual = torch_attention(torch, *inputs, arm).numpy()
                            np.testing.assert_allclose(actual, reference, atol=2e-4, rtol=2e-4)
                            errors[arm] = float(np.max(np.abs(actual - reference)))
                            normalized_errors[arm] = float(np.max(np.abs(actual-reference)/(2e-4+2e-4*np.abs(reference))))
                        rows.append(dict(query_length=length, key_length=keys, width=width,
                                         strided=strided, pattern=pattern, max_abs_error=errors,
                                         normalized_error=normalized_errors))
        # Hand-computable counterexample: uniform logits, values 1,2,3,4.
        q = torch.zeros(1, 1, 1, 1); k = torch.zeros(1, 1, 4, 1)
        v = torch.arange(1, 5, dtype=torch.float32).reshape(1, 1, 4, 1)
        correct = torch_attention(torch, q, k, v, 'math').item()
        wrong = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).item()
        assert correct == 2.5 and wrong == 1.0
        # Cached chunk equals final rows of full causal prefill on the same Q/K/V.
        q, k, v = [torch.from_numpy(rng.normal(size=(1, 2, 17, 8)).astype('float32')) for _ in range(3)]
        full = torch_attention(torch, q, k, v, 'math')
        for length in (1, 3):
            torch.testing.assert_close(torch_attention(torch, q[..., -length:, :], k, v, 'math'),
                                       full[..., -length:, :], atol=2e-5, rtol=2e-5)
    return dict(cases=rows, tolerance=dict(atol=2e-4, rtol=2e-4),
                decode_counterexample=dict(expected=2.5, correct=correct, upper_left_causal=wrong),
                cached_suffix_checks=2, correctness_only=True, cuda_performance_measured=False)


def summarize(records, spec):
    expected = {(b, l, s, r, a) for b in spec['batches'] for l, s in spec['shapes']
                for r in range(spec['rounds']) for a in spec['arms']}
    cells = {}
    for row in records:
        key = tuple(row[k] for k in ('batch', 'query_length', 'key_length', 'round', 'arm'))
        if key not in expected or key in cells:
            raise ValueError('Duplicate or unexpected timing cell')
        for field in ('event_ms_per_call', 'wall_ms_per_call'):
            if not np.isfinite(row[field]) or row[field] <= 0:
                raise ValueError('Invalid duration')
        cells[key] = row
    if set(cells) != expected:
        raise ValueError('Incomplete matrix')
    rows = []
    for b in spec['batches']:
        for l, s in spec['shapes']:
            med = {a: statistics.median(cells[b, l, s, r, a]['event_ms_per_call']
                                       for r in range(spec['rounds'])) for a in spec['arms']}
            faster = sum(cells[b, l, s, r, 'flash']['event_ms_per_call'] <
                         cells[b, l, s, r, 'math']['event_ms_per_call'] for r in range(spec['rounds']))
            ratio = med['math'] / med['flash']
            rows.append(dict(batch=b, query_length=l, key_length=s, median_event_ms=med,
                             math_over_flash=ratio, faster_rounds=faster,
                             speed_gate=ratio >= spec['minimum_speedup'] and faster >= spec['minimum_faster_rounds']))
    return dict(shapes=rows, model_speedup_claim=False, custom_kernel_claim=False)


def cuda_benchmark(torch, output, spec, info):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError('NVIDIA CUDA required; CPU/MPS/ROCm substitution is forbidden')
    if torch.__version__.split('+')[0] != spec['torch_version']:
        raise RuntimeError('Protocol requires PyTorch ' + spec['torch_version'])
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    info.update(device=prop.name, capability=list(torch.cuda.get_device_capability()),
                total_memory=prop.total_memory, cuda_version=torch.version.cuda,
                tf32=False)
    save(output / 'run.json', info)
    rng = np.random.default_rng(spec['seed']); order = random.Random(spec['seed'])
    records, checks, memory = [], [], []
    def context(arm):
        return (nullcontext() if arm == 'eager_fp32' else
                sdpa_kernel(SDPBackend.MATH if arm == 'math' else SDPBackend.FLASH_ATTENTION))
    with torch.inference_mode():
        for b in spec['batches']:
            for length, keys in spec['shapes']:
                host_inputs = [rng.normal(size=(b, spec['heads'], n, spec['head_dim'])).astype('float16')
                               for n in (length, keys, keys)]
                inputs = []
                def evaluate(arm):
                    # Retain the original host inputs even if transfer/reference fails.
                    if not inputs:
                        inputs.extend(torch.from_numpy(array).cuda() for array in host_inputs)
                    with context(arm):
                        return torch_attention(torch, *inputs, arm).float().cpu().numpy()
                def compare(actual, reference):
                    torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(reference),
                                               atol=spec['atol'], rtol=spec['rtol'])
                validate_case(output, checks, dict(batch=b, query_length=length, key_length=keys),
                              host_inputs, spec['arms'], evaluate, compare, spec['atol'], spec['rtol'])
                # A failed shape raises before warmup/memory/timing. Earlier shape
                # timings remain partial records; a failed run never gets a summary.
                for arm in spec['arms']:
                    with context(arm):
                        for _ in range(spec['warmups']):
                            y = torch_attention(torch, *inputs, arm); del y
                        torch.cuda.synchronize()
                        before = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                        y = torch_attention(torch, *inputs, arm)
                        torch.cuda.synchronize()
                        memory.append(dict(batch=b, query_length=length, key_length=keys, arm=arm,
                                           incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated()-before))
                        del y
                save(output / 'memory.json', memory)
                for r in range(spec['rounds']):
                    arms = list(spec['arms']); order.shuffle(arms)
                    for arm in arms:
                        with context(arm):
                            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                            torch.cuda.synchronize(); wall = time.perf_counter()
                            start.record()
                            for _ in range(spec['repeats']):
                                y = torch_attention(torch, *inputs, arm); del y
                            end.record(); end.synchronize()
                            wall_ms = (time.perf_counter()-wall)*1000/spec['repeats']
                            records.append(dict(batch=b, query_length=length, key_length=keys,
                                                round=r, arm=arm, event_ms_per_call=start.elapsed_time(end)/spec['repeats'],
                                                wall_ms_per_call=wall_ms))
                            save(output / 'timings.json', records)
                del inputs
    save(output / 'summary.json', summarize(records, spec))
    info['cuda_performance_measured'] = True


def run(output, mode):
    info = begin(output, mode)
    try:
        import torch
        info.update(torch_version=torch.__version__, numpy_version=np.__version__)
        torch.set_num_threads(1)
        if mode == 'cpu-semantics':
            save(output / 'semantics.json', semantics(torch))
        else:
            cuda_benchmark(torch, output, json.loads(SPEC.read_text()), info)
        info['status'] = 'complete'
    except Exception as exc:
        info.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        info['artifact_sha256'] = {p.name: digest(p) for p in output.iterdir()
                                  if p.is_file() and p.suffix in ('.json', '.npz') and p.name != 'run.json'}
        save(output / 'run.json', info)


def verify(output):
    info = json.loads((output / 'run.json').read_text())
    for filename, value in info['source_sha256'].items():
        if digest(output / 'source' / filename) != value:
            raise ValueError('Archived source hash mismatch')
    if info['status'] != 'complete':
        raise ValueError('Study did not complete')
    expected_files = ({'semantics.json'} if info['mode'] == 'cpu-semantics' else
                      {'correctness.json', 'memory.json', 'timings.json', 'summary.json'})
    if set(info['artifact_sha256']) != expected_files:
        raise ValueError('Missing or unexpected artifacts')
    for filename, value in info['artifact_sha256'].items():
        if digest(output / filename) != value:
            raise ValueError('Artifact hash mismatch')
    if info['mode'] == 'cpu-semantics':
        data = json.loads((output / 'semantics.json').read_text())
        expected = {(l,s,d,t,p) for l,s in [(7,7),(1,17),(3,17)] for d in (8,32)
                    for t in (False,True) for p in ('normal','zeros','large_logits')}
        observed = {tuple(r[k] for k in ('query_length','key_length','width','strided','pattern')) for r in data['cases']}
        if len(data['cases']) != 36 or observed != expected or info['cuda_performance_measured']:
            raise ValueError('Semantic study scope mismatch')
        if data['decode_counterexample'] != dict(expected=2.5, correct=2.5, upper_left_causal=1.0) or data['cached_suffix_checks'] != 2:
            raise ValueError('Causal checks failed')
        for row in data['cases']:
            if set(row['normalized_error']) != {'math', 'eager_fp32'}:
                raise ValueError('Missing semantic arm')
            if any(not np.isfinite(v) or not 0 <= v <= 1 for v in row['normalized_error'].values()):
                raise ValueError('Semantic tolerance failed')
        print(json.dumps(dict(evidence_valid=True, cpu_cases=36, cuda_performance_measured=False)))
    else:
        spec = json.loads((output / 'source' / SPEC).read_text())
        summary = summarize(json.loads((output / 'timings.json').read_text()), spec)
        if summary != json.loads((output / 'summary.json').read_text()):
            raise ValueError('Summary mismatch')
        expected = {(b,l,s,a) for b in spec['batches'] for l,s in spec['shapes'] for a in spec['arms']}
        for filename, field in [('correctness.json','normalized_error'), ('memory.json','incremental_peak_allocated_bytes')]:
            rows = json.loads((output / filename).read_text())
            observed = {tuple(r[k] for k in ('batch','query_length','key_length','arm')) for r in rows}
            if len(rows) != len(expected) or observed != expected:
                raise ValueError('Incomplete correctness or memory matrix')
            if any(not np.isfinite(r[field]) or r[field] < 0 for r in rows):
                raise ValueError('Invalid validation measurement')
            if filename == 'correctness.json' and any(r[field] > 1 for r in rows):
                raise ValueError('CUDA correctness failed')
        if not info['cuda_performance_measured']:
            raise ValueError('Missing CUDA execution')
        print(json.dumps(dict(evidence_valid=True, summary=summary)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['cpu-semantics', 'cuda', 'verify', 'verify-failure'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'verify-failure':
        print(json.dumps(verify_failure(args.output), indent=2, allow_nan=False))
    else:
        verify(args.output) if args.mode == 'verify' else run(args.output, args.mode)
