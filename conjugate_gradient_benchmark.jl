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
r = JACC.ones(SIZE)
p = JACC.ones(SIZE)
s = JACC.zeros(SIZE)
x = JACC.zeros(SIZE)
r_old = JACC.zeros(SIZE)
r_aux = JACC.zeros(SIZE)

a1 = a1 * 4
r = r * 0.5
p = p * 0.5

cond = 1.0

#warmup run
while cond[1, 1] >= 1e-14
        global cond

	copyto!(r_old, r)

        JACC.parallel_for(SIZE, matvecmul, a0, a1, a2, p, s, SIZE)

        alpha0 = JACC.parallel_reduce(SIZE, dot, r, r)
        alpha1 = JACC.parallel_reduce(SIZE, dot, p, s)

        alpha = alpha0 / alpha1
        negative_alpha = alpha * -1.0

        JACC.parallel_for(SIZE, axpy, negative_alpha, r, s)
        JACC.parallel_for(SIZE, axpy, alpha, x, p)

        beta0 = JACC.parallel_reduce(SIZE, dot, r, r)
        beta1 = JACC.parallel_reduce(SIZE, dot, r_old, r_old)
        beta = beta0 / beta1

        copyto!(r_aux, r)

        JACC.parallel_for(SIZE, axpy, beta, r_aux, p)
        ccond = JACC.parallel_reduce(SIZE, dot, r, r)
        cond = ccond

        copyto!(p, r_aux)
end

#reset arrays
fill!(r, 0.5)
fill!(p, 0.5)

fill!(s, 0.0)
fill!(x, 0.0)
fill!(r_old, 0.0)
fill!(r_aux, 0.0)

result = @benchmark begin
    cond = 1.0

    while cond >= 1e-14
        copyto!($r_old, $r)

        JACC.parallel_for($SIZE, $matvecmul, $a0, $a1, $a2, $p, $s, $SIZE)

        alpha0 = JACC.parallel_reduce($SIZE, dot, $r, $r)
        alpha1 = JACC.parallel_reduce($SIZE, dot, $p, $s)

        alpha = alpha0 / alpha1
        negative_alpha = alpha * -1.0

        JACC.parallel_for($SIZE, axpy, negative_alpha, $r, $s)
        JACC.parallel_for($SIZE, axpy, alpha, $x, $p)

        beta0 = JACC.parallel_reduce($SIZE, dot, $r, $r)
        beta1 = JACC.parallel_reduce($SIZE, dot, $r_old, $r_old)
        beta = beta0 / beta1

        copyto!($r_aux, $r)

        JACC.parallel_for($SIZE, axpy, beta, $r_aux, $p)
        ccond = JACC.parallel_reduce($SIZE, dot, $r, $r)
        
        cond = ccond[1, 1]

        copyto!($p, $r_aux)
    end
end samples = 1 evals = 1

println(result.times[1])
