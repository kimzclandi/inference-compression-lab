"""Verify RC4 evidence and write a compact bilingual review, without inference.

Requires NumPy 2.2.6. The output directory must be new. This command replays
archived evidence; it does not train a new policy, benchmark hardware, inspect
GitHub access or publish a release.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from lab.artifact_integrity import git_identity
from lab.evidence import reserve_directory
from lab.quantization_diagnostics import read, sha

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'experiments/review_qa_evidence.py', 'experiments/verify_release_rc4.py',
    'configs/qa-risk/study.json', 'configs/qa-risk/policy.json',
    'results/qa-risk-v2/training/selection.json',
    'results/qa-risk-v2/evaluation-int8/summary.json',
    'results/qa-specialist-performance-v1/summary.json',
    'results/qa-risk-performance-v1/summary.json',
)


def verify_evidence():
    from experiments.verify_release_rc4 import verify
    return verify(ROOT)


def snapshot():
    return {name: sha(ROOT / name) for name in SOURCES}


def compact(acceptance):
    # These summaries are only read AFTER complete raw-record verification.
    qa = read(ROOT / 'results/qa-risk-v2/evaluation-int8/summary.json')['summary']
    base = read(ROOT / 'results/qa-specialist-performance-v1/summary.json')
    risk = read(ROOT / 'results/qa-risk-performance-v1/summary.json')
    return dict(
        evidence_acceptance=acceptance['technical_acceptance'],
        task_scope='bounded_local_english_supplied_passage_prototype',
        quality=dict(counts=qa['selective'], system=qa['system']['overall'],
                     precision_wilson95=acceptance['risk']['accepted_precision_wilson95']),
        performance=dict(
            base_pipeline_ms={k: v * 1000 for k, v in base['median_of_process_medians_seconds'].items()},
            base_fp32_over_int8=base['fp32_over_int8_speedup'],
            base_scope=base['scope'],
            ranked_int8_ms=risk['median_of_process_medians_seconds'] * 1000,
            ranked_scope=risk['scope'],
            hardware='Historical single M4 Max CPU; fixed single-window inputs; no new timing run'),
        original_fallback_confirmed=acceptance['original_quantization_fallback_confirmed'],
        compression_noninferiority_confirmed=acceptance['compression_quality_noninferiority_confirmed'],
        general_deployment_verified=acceptance['general_qa_deployment_quality'],
        publication='not_checked_or_performed_by_this_command',
        root_license_status=acceptance['root_code_license_status'],
        numerical_scope='Raw features/labels checked, then verified archived training matrix replayed. '
                        'Convergence from arbitrarily regenerated floating-point inputs is not established.',
    )


def render(report):
    q, p = report['quality'], report['performance']
    c = q['counts']
    return f'''# QA evidence review / QA 证据核验

**离线证据核验通过 / Offline evidence verification passed.**
这是已公布材料的重算，不是新模型推理、质量评估或性能测量。
This is a replay of published evidence, not a new inference, quality evaluation or benchmark.

| 指标 / Metric | 已核验值 / Verified value |
|---|---:|
| 接受答案精度 / Accepted exact-match precision | {c['accepted_correct']}/{c['accepted']} |
| 可回答覆盖 / Answerable coverage | {c['accepted_answerable']}/{c['answerable']} ({c['answerable_answer_coverage']:.2%}) |
| 不可回答误接受 / Unanswerable false accepts | {c['accepted_unanswerable']}/{c['unanswerable']} |
| 全部问题 system EM / System EM over all questions | {q['system']['em']:.2%} (n={q['system']['n']}) |
| Precision 描述性 Wilson 95% 下界 / Descriptive lower bound | {q['precision_wilson95']['lower']:.2%} |
| 基础 FP32 / Base FP32 | {p['base_pipeline_ms']['fp32']:.3f} ms |
| 基础 INT8 / Base INT8 | {p['base_pipeline_ms']['int8']:.3f} ms |
| 基础流水线加速比 / Base-only speedup | {p['base_fp32_over_int8']:.3f}x |
| 含排序与拒答的 INT8 / INT8 with ranking and abstention | {p['ranked_int8_ms']:.3f} ms |

1. 性能范围不同，1.33x不属于完整排序流水线；历史启动核验/加载约18.04秒不含在热请求内。
   Different timing scopes: do not transfer the base speedup to the ranked pipeline or hide startup costs.
2. 27个接受答案不足以保证总体precision≥90%；只支持有限本地原型。
   The point gates do not establish population risk or deployment readiness.
3. Qwen回退确认、specialist v1质量及量化非劣失败均保留；FP32排序头校准失败，未进入新评估。
   Earlier failures remain; there is no new paired compression noninferiority result.
4. Civil_disobedience全32题拒答；37个可回答拒答中19个raw答案正确，详见下方失败分析。
   Coverage loss and article-level failure remain material limitations.
5. 存档矩阵重建通过不等于任意平台重新生成浮点输入后训练均稳定。
   Archived-input reconstruction does not establish arbitrary regenerated-input convergence.
6. 未检查远端发布状态，也未执行发布。
   Remote publication is outside this command.

核验范围与文件SHA256见同目录`review.json`；完整验收见`acceptance.json`。
Source paths below are relative to the source checkout/archive, not this report directory:

- `docs/qa-system-overview.md`: architecture and engineering decisions / 系统与实现导航
- `docs/qa-risk-study.md`: fixed protocol and limitations / 协议与边界
- `results/qa-risk-v2/evaluation-int8/predictions.jsonl`: raw evidence / 原始记录
- `results/qa-risk-review-v1/evaluation-independent.json`: article failures / 文章失效
- `docs/release-reproduction.md`: dependencies, no-Git usage and access / 复现与访问

本工具不分发模型或排序head参数，不改变数据、阈值或历史结果。
No model/head parameters are distributed and no historical experiment is changed.
'''


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def run(output_dir):
    output = reserve_directory(output_dir)
    state = dict(status='running', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                 **git_identity(ROOT), verification_kind='offline_archived_evidence_replay')
    try:
        before = snapshot()
        acceptance = verify_evidence()
        if acceptance.get('technical_acceptance') != 'pass':
            raise ValueError('Complete evidence verification did not pass')
        report = compact(acceptance)
        if snapshot() != before:
            raise ValueError('Review inputs changed during verification')
        report.update({**state, 'status': 'complete', 'source_sha256': before})
        markdown = render(report)
        write_json(output / 'acceptance.json', acceptance)
        write_json(output / 'review.json', report)
        (output / 'review.md').write_text(markdown)
        state.update(status='complete', output_sha256={name: sha(output / name)
                     for name in ('acceptance.json', 'review.json', 'review.md')})
        return report
    except Exception as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(output / 'run.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args.output_dir)
    except Exception as error:
        print(f'Evidence review failed: {error}. Preserve the failure; check dependencies and evidence.', file=sys.stderr)
        return 1
    print(render(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
