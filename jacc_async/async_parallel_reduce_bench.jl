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

r  = JACC.Async.ones(1, SIZE)
p  = JACC.Async.ones(2, SIZE)
s1 = JACC.Async.zeros(2, SIZE)
r  = r * 0.5
CUDA.device!(1)
p  = p * 0.5

#warmup run
JACC.Async.parallel_reduce(2, SIZE, dot, p, s1)
JACC.Async.parallel_reduce(1, SIZE, dot, r, r)
JACC.Async.synchronize()

#ten iterations
result = @benchmark begin 
for i=1:10
	JACC.Async.parallel_reduce(2, $SIZE, dot, $p, $s1)
	JACC.Async.parallel_reduce(1, $SIZE, dot, $r, $r)
	JACC.Async.synchronize()
end
end samples = 1 evals = 1

print(round(Int, result.times[1]/10))

