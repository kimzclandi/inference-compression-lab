"""Inference-only fixed-capacity contiguous K/V storage; no paging or eviction."""


class AppendOnlyKV:
    """Returned prefix tensors are borrowed read-only views, not owned snapshots.

    Single writer, same execution stream; no reset, growth, concurrent access or
    autograd support. A synchronous copy exception keeps the visible length and
    prior prefix intact. Asynchronous device errors are not recoverable here.
    """
    def __init__(self, keys, values, capacity):
        import torch
        if (keys.ndim != 4 or keys.shape != values.shape or min(keys.shape) < 1
                or keys.dtype != values.dtype or keys.device != values.device
                or keys.requires_grad or values.requires_grad):
            raise ValueError('Matching, nonempty inference K/V tensors required')
        if type(capacity) is not int or capacity < keys.shape[-2]:
            raise ValueError('Capacity must include the initial prefix')
        shape=(*keys.shape[:2],capacity,keys.shape[-1])
        self._keys=torch.empty(shape,dtype=keys.dtype,device=keys.device)
        self._values=torch.empty_like(self._keys)
        self.capacity=capacity
        self.length=keys.shape[-2]
        self._keys[:,:,:self.length].copy_(keys)
        self._values[:,:,:self.length].copy_(values)

    @staticmethod
    def _copy(destination, source):
        destination.copy_(source)

    def append(self, keys, values):
        expected=(*self._keys.shape[:2],keys.shape[-2] if keys.ndim==4 else 0,self._keys.shape[-1])
        if (keys.ndim!=4 or tuple(keys.shape)!=expected or keys.shape!=values.shape
                or keys.shape[-2]<1 or keys.dtype!=self._keys.dtype or values.dtype!=self._values.dtype
                or keys.device!=self._keys.device or values.device!=self._values.device
                or keys.requires_grad or values.requires_grad):
            raise ValueError('Append shape, dtype and device must match; no broadcasting or autograd')
        end=self.length+keys.shape[-2]
        if end>self.capacity:
            raise OverflowError('Fixed cache capacity exhausted')
        self._copy(self._keys[:,:,self.length:end],keys)
        self._copy(self._values[:,:,self.length:end],values)
        # Never expose a half-committed K/V pair after a synchronous exception.
        self.length=end
        return self.view()

    def view(self):
        return self._keys[:,:,:self.length],self._values[:,:,:self.length]
