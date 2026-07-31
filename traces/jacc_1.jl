include("../setup.jl")
using ROCTX

function matvecmul(i, a1, a2, a3, x, y, SIZE)
        if i == 1
            y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
        elseif i == SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
        elseif i > 1 && i < SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * +x[i] + a1[i] * +x[i + 1]
        end
end

SIZE = 1000000
r  = JACC.ones(SIZE)
p  = JACC.ones(SIZE)
s  = JACC.zeros(SIZE)
r  = r * 0.5
p  = p * 0.5

#warmup
JACC.parallel_reduce(SIZE, dot, r, r)
JACC.parallel_reduce(SIZE, dot, p, s)

#traced loop
ROCTX.range_push("Full_Axpy_Loop")
for i = 1:10
	ROCTX.range_push("Iteration_$i")

	JACC.parallel_reduce($SIZE, dot, $r, $r)
        JACC.parallel_reduce($SIZE, dot, $p, $s)

	ROCTX.range_pop()
end
ROCTX.range_pop()
