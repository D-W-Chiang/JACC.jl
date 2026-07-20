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
r_old = JACC.Async.zeros(2, SIZE)
r_aux = JACC.Async.zeros(2, SIZE)

r  = r * 0.5
CUDA.device!(1)
p  = p * 0.5

copyto!(r_old, r)

beta1 = JACC.Async.parallel_reduce(2, SIZE, dot, r_old, r_old)
beta0 = JACC.Async.parallel_reduce(1, SIZE, dot, r, r)
beta = JACC.to_host(beta0)[] / JACC.to_host(beta1)[]

copyto!(r_aux, r) 

#warmup run
JACC.Async.parallel_for(2, SIZE, axpy, beta, r_aux, p)
JACC.Async.parallel_reduce(1, SIZE, dot, r, r) 
JACC.Async.synchronize()

result = @benchmark begin
#ten iterations
	for i = 1:10
		JACC.Async.parallel_for(2, $SIZE, axpy, $beta, $r_aux, $p)
		JACC.Async.parallel_reduce(1, $SIZE, dot, $r, $r)
		JACC.Async.synchronize()
	end
end samples = 1 evals = 1 

#average time: 
print(round(Int, result.times[1]/10))
