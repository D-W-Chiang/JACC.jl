   while cond[1, 1] >= 1e-14
            copyto!(r_old, r)

            JACC.Async.parallel_for(1, SIZE, matvecmul, a0, a1, a2, p, s1, SIZE)

            alpha1 = JACC.Async.parallel_reduce(1, SIZE, dot, p, s1)
            alpha0 = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
            JACC.Async.synchronize()

            alpha = JACC.to_host(alpha0)[] / JACC.to_host(alpha1)[]
            negative_alpha = alpha * -1.0

            copyto!(s1, s2)
            JACC.Async.parallel_for(1, SIZE, axpy, alpha, x, p)
            JACC.Async.parallel_for(2, SIZE, axpy, negative_alpha, r, s2)
            JACC.Async.synchronize()

            beta1 = JACC.Async.parallel_reduce(1, SIZE, dot, r_old, r_old)
            beta0 = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
            JACC.Async.synchronize()
            beta = JACC.to_host(beta0)[] / JACC.to_host(beta1)[]

            copyto!(r_aux, r)

            JACC.Async.parallel_for(1, SIZE, axpy, beta, r_aux, p)
            ccond = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
            JACC.Async.synchronize()
            cond = JACC.to_host(ccond)[]

            copyto!(p, r_aux)
        end
