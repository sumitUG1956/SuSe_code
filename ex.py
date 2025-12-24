import cupy as cp

a = cp.random.rand(10_000_000)
b = cp.random.rand(10_000_000)

c = a + b
cp.cuda.Stream.null.synchronize()

print(c[:5])
