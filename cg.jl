include("setup.jl")

function matvecmul(i, a1, a2, a3, x, y, SIZE)
        if i == 1
            y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
        elseif i == SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
        elseif i > 1 && i < SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * +x[i] + a1[i] * +x[i + 1]
        end
    end

SIZE = 1_000_000
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


while cond[1, 1] >= 1e-14
 	global p, r_old, r_aux, cond

	r_old = copy(r)

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

        r_aux = copy(r)

        JACC.parallel_for(SIZE, axpy, beta, r_aux, p)
        ccond = JACC.parallel_reduce(SIZE, dot, r, r)
        cond = ccond

        p = copy(r_aux)
    end
print(cond)
