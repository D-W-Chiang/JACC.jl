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
a1 = a1 * 4
r  = r * 0.5
p  = p * 0.5

function matvecmul(i, a1, a2, a3, x, y, SIZE)
        if i == 1
            y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
        elseif i == SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
        elseif i > 1 && i < SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * +x[i] + a1[i] * +x[i + 1]
        end
end

JACC.parallel_for(SIZE, matvecmul, a0, a1, a2, p, s, SIZE)

alpha0 = JACC.parallel_reduce(SIZE, dot, r, r)
alpha1 = JACC.parallel_reduce(SIZE, dot, p, s)

alpha = alpha0 / alpha1
negative_alpha = alpha * -1.0

#warmup run
JACC.parallel_for(SIZE, axpy, negative_alpha, r, s)
JACC.parallel_for(SIZE, axpy, alpha, x, p) 

result = @benchmark begin
#ten iterations 
	for i = 1:10
		JACC.parallel_for($SIZE, axpy, $negative_alpha, $r, $s)
		JACC.parallel_for($SIZE, axpy, $alpha, $x, $p)
	end
end samples = 1 evals = 1 

#average time:
print(round(Int, result.times[1]/10))
