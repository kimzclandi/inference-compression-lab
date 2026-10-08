"""Bounded single-worker admission policies; no model/backend dependencies."""
from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_s: float
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError('nonempty request ID required')
        if not math.isfinite(self.arrival_s) or self.arrival_s < 0:
            raise ValueError('arrival must be finite and nonnegative')
        if any(type(x) is not int or x <= 0 for x in (self.prompt_tokens, self.output_tokens)):
            raise ValueError('positive integer token budgets required')


class AdmissionQueue:
    """Known arrivals are admitted only when their real deadline has passed.

    The short-budget policy scores prompt_tokens + decode_weight*output_tokens.
    An older ready request may be bypassed at most max_bypasses times. This
    bounds reorder count, not wall-clock waiting time or serving SLA.
    """
    def __init__(self, requests, policy='fcfs', max_bypasses=3, decode_weight=32):
        requests = list(requests)
        if policy not in ('fcfs', 'short_budget'):
            raise ValueError('unknown policy')
        if type(max_bypasses) is not int or max_bypasses < 0:
            raise ValueError('invalid bypass bound')
        if type(decode_weight) is not int or decode_weight <= 0:
            raise ValueError('invalid decode weight')
        if len({x.request_id for x in requests}) != len(requests):
            raise ValueError('duplicate request ID')
        self.pending = sorted(enumerate(requests), key=lambda x: (x[1].arrival_s, x[0]))
        self.policy, self.max_bypasses, self.decode_weight = policy, max_bypasses, decode_weight
        self.bypasses = {x.request_id: 0 for x in requests}
        self.cancelled = set()
        self._next_index = len(requests)

    def submit(self, request):
        """Caller serializes submissions/pop/cancel with its own lock."""
        if request.request_id in self.bypasses:
            raise ValueError('duplicate request ID')
        self.pending.append((self._next_index, request))
        self._next_index += 1
        self.pending.sort(key=lambda x: (x[1].arrival_s, x[0]))
        self.bypasses[request.request_id] = 0

    def cancel(self, request_id):
        for i, (_, req) in enumerate(self.pending):
            if req.request_id == request_id:
                self.pending.pop(i)
                self.cancelled.add(request_id)
                return True
        return False

    def pop_ready(self, now_s):
        if not math.isfinite(now_s) or now_s < 0:
            raise ValueError('invalid clock')
        ready = [(i, req) for i, req in self.pending if req.arrival_s <= now_s]
        if not ready:
            return None
        chosen = ready[0]
        if self.policy == 'short_budget':
            saturated = next((item for item in ready
                              if self.bypasses[item[1].request_id] >= self.max_bypasses), None)
            chosen = saturated or min(ready, key=lambda x: (
                x[1].prompt_tokens + self.decode_weight*x[1].output_tokens,
                x[1].arrival_s, x[0]))
        # Every skipped older request consumes one of its bounded bypasses.
        for item in ready:
            if item == chosen:
                break
            self.bypasses[item[1].request_id] += 1
        self.pending.remove(chosen)
        return chosen[1]


def percentile(values, q):
    """Linear interpolation of sorted observations, matching NumPy default."""
    values = sorted(values)
    if not values or not 0 <= q <= 1:
        raise ValueError('nonempty data and q in [0,1] required')
    position = (len(values)-1)*q
    lower = int(position)
    return values[lower] + (values[min(lower+1, len(values)-1)]-values[lower])*(position-lower)


def summarize(rows, elapsed_s):
    if elapsed_s <= 0 or not rows:
        raise ValueError('positive elapsed and nonempty rows required')
    result = {'requests': len(rows), 'tokens': sum(len(x['tokens']) for x in rows),
              'elapsed_s': elapsed_s, 'throughput_tokens_s': sum(len(x['tokens']) for x in rows)/elapsed_s}
    for metric in ('queue_s', 'ttft_s', 'tpot_s', 'total_s', 'service_s'):
        values = [row[metric] for row in rows]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid latency')
        result[metric] = {'mean': statistics.mean(values), 'p50': percentile(values, .5),
                          'p95': percentile(values, .95)}
    return result


def consume_stream(stream, on_token, cancel_after=None, fail_after=None):
    """Close native iterator on success, cancellation, callback or injected failure.

    Closing stops future iteration; it cannot revoke already enqueued GPU work.
    Caller owns synchronization and release of its per-request KV cache.
    """
    try:
        for i, item in enumerate(stream, 1):
            on_token(item)
            if fail_after is not None and i == fail_after:
                raise RuntimeError('injected consumer failure')
            if cancel_after is not None and i == cancel_after:
                return 'cancelled'
        return 'complete'
    finally:
        stream.close()
