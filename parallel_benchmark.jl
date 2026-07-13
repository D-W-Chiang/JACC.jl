include("setup.jl")

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

a0 = JACC.ones(SIZE)
a1 = JACC.ones(SIZE)
a2 = JACC.ones(SIZE)
r  = JACC.ones(SIZE)
p  = JACC.ones(SIZE)
s  = JACC.zeros(SIZE)
x  = JACC.zeros(SIZE)
r_old = JACC.zeros(SIZE)
r_aux = JACC.zeros(SIZE)

a1 = a1 * 4
r  = r * 0.5
p  = p * 0.5

copyto!(r_old, r)

beta0 = JACC.parallel_reduce(SIZE, dot, r, r)
beta1 = JACC.parallel_reduce(SIZE, dot, r_old, r_old)
beta = beta0 / beta1

copyto!(r_aux, r)

#warmup run
JACC.parallel_for(SIZE, axpy, beta, r_aux, p)
JACC.parallel_reduce(SIZE, dot, r, r) 

result = @benchmark begin
#ten iterations
	for i = 1:10
		JACC.parallel_for($SIZE, axpy, $beta, $r_aux, $p)
		JACC.parallel_reduce($SIZE, dot, $r, $r)

	end
end samples = 1 evals = 1 

#average time: 
print(round(Int, result.times[1]/10))
