"""CPU error summaries and strictly scoped diagnostic hooks. No performance API."""
from contextlib import contextmanager
import hashlib

_ACTIVE=False


def array_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()


def error_metrics(left,right,atol,rtol):
    """Elementwise |left-right| <= atol+rtol*|right|; max/ULP are descriptive."""
    import numpy as np
    a=np.asarray(left);b=np.asarray(right)
    if a.shape!=b.shape:raise ValueError('shape mismatch')
    if not (np.isfinite(a).all() and np.isfinite(b).all()):raise ValueError('nonfinite')
    af=a.astype('float64');bf=b.astype('float64');delta=np.abs(af-bf)
    fail=~np.isclose(a,b,atol=atol,rtol=rtol);float64_fail=delta>(atol+rtol*np.abs(bf));different=(a!=b)
    maximum=tuple(int(i) for i in np.unravel_index(int(np.argmax(delta)),a.shape))
    first=lambda mask: [int(i) for i in np.argwhere(mask)[0]] if mask.any() else None
    row=dict(shape=list(a.shape),left_sha256=array_sha(a),right_sha256=array_sha(b),
        bitwise_equal=bool(a.dtype==b.dtype and a.tobytes()==b.tobytes()),equal_values=bool(not different.any()),
        unequal_elements=int(different.sum()),first_unequal=first(different),
        allclose=bool(not fail.any()),failed_elements=int(fail.sum()),first_failure=first(fail),
        float64_gate_allclose=bool(not float64_fail.any()),float64_failed_elements=int(float64_fail.sum()),float64_first_failure=first(float64_fail),
        gate_method='numpy.isclose on original dtypes; float64 gate separately descriptive',
        max_abs=float(delta[maximum]),max_coordinate=list(maximum),left_at_max=float(af[maximum]),right_at_max=float(bf[maximum]),
        abs_percentiles=dict(zip(('p50','p95','p99'),[float(x) for x in np.percentile(delta,[50,95,99])])),
        atol=atol,rtol=rtol,relative_reference='right')
    if a.dtype==b.dtype==np.float16:
        def ordered(x):
            bits=x.view(np.uint16).astype(np.int32);magnitude=bits&0x7fff
            return np.where(bits&0x8000,0x8000-magnitude,0x8000+magnitude)
        ulp=np.abs(ordered(a)-ordered(b))
        row['ulp_max']=int(ulp.max());row['ulp_p99']=float(np.percentile(ulp,99))
    return row


@contextmanager
def diagnostic_hooks(qwen_module, candidate_module, callback, candidate_mode):
    """One synchronous diagnostic process only; never a service monkeypatch.

    Caller callback(native flag,q,k,v,out,cache) records the actual boundary.
    Save/restore exact function identities even on callback/model failure.
    """
    global _ACTIVE
    if _ACTIVE:raise RuntimeError('nested or concurrent diagnostic hooks unsupported')
    if candidate_mode not in ('native','shared_compiled'):raise ValueError('mode')
    original_native=qwen_module.scaled_dot_product_attention
    original_candidate=candidate_module.attention
    def native(q,k,v,**kwargs):
        out=original_native(q,k,v,**kwargs)
        callback(True,q,k,v,out,kwargs.get('cache'))
        return out
    def candidate(q,k,v,mode='shared_compiled'):
        if mode!='shared_compiled':raise ValueError('unexpected adapter route')
        out=original_candidate(q,k,v,mode=candidate_mode)
        callback(False,q,k,v,out,None)
        return out
    _ACTIVE=True
    try:
        qwen_module.scaled_dot_product_attention=native
        candidate_module.attention=candidate
        yield
    finally:
        qwen_module.scaled_dot_product_attention=original_native
        candidate_module.attention=original_candidate
        _ACTIVE=False
