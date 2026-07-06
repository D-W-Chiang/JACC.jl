include("setup.jl")

using BenchmarkTools

SIZE = 10

function matvecmul(i, a1, a2, a3, x, y, SIZE)
        if i == 1
            y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
        elseif i == SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
        elseif i > 1 && i < SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * +x[i] + a1[i] * +x[i + 1]
        end
end

r  = JACC.ones(SIZE)
p  = JACC.ones(SIZE)
s  = JACC.zeros(SIZE)
r  = r * 0.5
p  = p * 0.5

#warmup run
JACC.parallel_reduce(SIZE, dot, r, r)
JACC.parallel_reduce(SIZE, dot, p, s)

#ten iterations
result = @benchmark begin 
for i = 1:10
	JACC.parallel_reduce($SIZE, dot, $r, $r)
	JACC.parallel_reduce($SIZE, dot, $p, $s)
end
end samples = 1 evals = 1

print(round(Int, result.times[1]/10))

