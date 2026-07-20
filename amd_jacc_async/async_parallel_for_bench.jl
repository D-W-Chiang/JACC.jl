include("../setup.jl")

using BenchmarkTools

function matvecmul(i, a1, a2, a3, x, y, SIZE)
        if i == 1
            y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
        elseif i == SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
        elseif i > 1 && i < SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * +x[i] + a1[i] * +x[i + 1]
        end
end

devices = AMDGPU.devices()

a0 = JACC.Async.ones(2, SIZE)
a1 = JACC.Async.ones(2, SIZE)
a2 = JACC.Async.ones(2, SIZE)
r  = JACC.Async.ones(1, SIZE)
p  = JACC.Async.ones(2, SIZE)
s1 = JACC.Async.ones(2, SIZE)
s2 = JACC.Async.zeros(1, SIZE)
x  = JACC.Async.zeros(2, SIZE)
AMDGPU.device!(devices[2])
a1 = a1 * 4
p  = p * 0.5
AMDGPU.device!(devices[1])
r  = r * 0.5

JACC.Async.parallel_for(2, SIZE, matvecmul, a0, a1, a2, p, s1, SIZE)

alpha1 = JACC.Async.parallel_reduce(2, SIZE, dot, p, s1)
alpha0 = JACC.Async.parallel_reduce(1, SIZE, dot, r, r)
JACC.Async.synchronize() 

alpha = JACC.to_host(alpha0)[] / JACC.to_host(alpha1)[]
negative_alpha = alpha * -1.0

copyto!(s2, s1) 

#warmup
JACC.Async.parallel_for(2, SIZE, axpy, alpha, x, p)
JACC.Async.parallel_for(1, SIZE, axpy, negative_alpha, r, s2)
JACC.Async.synchronize()

#ten iterations 
result = @benchmark begin
	for i = 1:10
		JACC.Async.parallel_for(2, $SIZE, axpy, $alpha, $x, $p)
		JACC.Async.parallel_for(1, $SIZE, axpy, $negative_alpha, $r, $s2)
		JACC.Async.synchronize()
	end
end samples=1 evals=1

#average time: 
print(round(Int, result.times[1]/10))
