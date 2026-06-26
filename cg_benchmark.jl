using BenchmarkTools

suite = BenchmarkGroup()

function matvecmul(i, a1, a2, a3, x, y, SIZE)
        if i == 1
            y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
        elseif i == SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
        elseif i > 1 && i < SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * +x[i] + a1[i] * +x[i + 1]
        end
    end

suite["cg"] = let 

    SIZE = 100_000
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

@benchmarkable begin
    while cond[1, 1] >= 1e-14
        $r_old = copy($r)

        JACC.parallel_for($SIZE, matvecmul, $a0, $a1, $a2, $p, $s, $SIZE)

        alpha0 = JACC.parallel_reduce($SIZE, dot, $r, $r)
        alpha1 = JACC.parallel_reduce($SIZE, dot, $p, $s)

        alpha = alpha0 / alpha1
        negative_alpha = alpha * -1.0

        JACC.parallel_for($SIZE, axpy, negative_alpha, $r, $s)
        JACC.parallel_for($SIZE, axpy, alpha, $x, $p)

        beta0 = JACC.parallel_reduce($SIZE, dot, $r, $r)
        beta1 = JACC.parallel_reduce($SIZE, dot, $r_old, $r_old)
        beta = beta0 / beta1

        $r_aux = copy($r)

        JACC.parallel_for($SIZE, axpy, beta, $r_aux, $p)
        ccond = JACC.parallel_reduce($SIZE, dot, $r, $r)
        cond = ccond

        $p .= copy($r_aux) #= used to be p = copy(r_aux) but 
        wrapping the loop in @benchmark changes the scope to be local =#
    end 
end evals = 1 gcsample = true setup = ($x .= 0.0; $r .= 0.5; $p .= 0.5; cond = 1.0)
end

suite["cg_async"] = 

let 

    SIZE = 100_000
        a0 = JACC.Async.ones(1, SIZE)
        a1 = JACC.Async.ones(1, SIZE)
        a2 = JACC.Async.ones(1, SIZE)
        r = JACC.Async.ones(2, SIZE)
        p = JACC.Async.ones(1, SIZE)
        s1 = JACC.Async.zeros(1, SIZE)
        s2 = JACC.Async.zeros(2, SIZE)
        x = JACC.Async.zeros(1, SIZE)
        r_old = JACC.Async.zeros(1, SIZE)
        r_aux = JACC.Async.zeros(1, SIZE)
        a1 = a1 * 4
	CUDA.device!(1)
        r = r * 0.5
	CUDA.device!(0)
        p = p * 0.5
        cond = 1.0

@benchmarkable begin 
    while cond[1, 1] >= 1e-14
        copyto!($r, $r_old)

        JACC.Async.parallel_for(1, $SIZE, matvecmul, $a0, $a1, $a2, $p, $s1, $SIZE)

        alpha1 = JACC.Async.parallel_reduce(1, $SIZE, dot, $p, $s1)
        alpha0 = JACC.Async.parallel_reduce(2, $SIZE, dot, $r, $r)
        JACC.Async.synchronize()

        alpha = JACC.to_host(alpha0)[] / JACC.to_host(alpha1)[]
        negative_alpha = alpha * -1.0

        copyto!($s2, $s1)
        JACC.Async.parallel_for(1, $SIZE, axpy, alpha, $x, $p)
        JACC.Async.parallel_for(2, $SIZE, axpy, negative_alpha, $r, $s2)
        JACC.Async.synchronize()

        beta1 = JACC.Async.parallel_reduce(1, $SIZE, dot, $r_old, $r_old)
        beta0 = JACC.Async.parallel_reduce(2, $SIZE, dot, $r, $r)
        JACC.Async.synchronize()
        beta = JACC.to_host(beta0)[] / JACC.to_host(beta1)[]

        copyto!($r, $r_aux)

        JACC.Async.parallel_for(1, $SIZE, axpy, beta, $r_aux, $p)
        ccond = JACC.Async.parallel_reduce(2, $SIZE, dot, $r, $r)
        JACC.Async.synchronize()
        cond = JACC.to_host(ccond)[]

        copyto!($p, $r_aux)
    end
end samples = 1 evals = 1 gcsample = true setup = ($x .= 0.0; CUDA.device!(1); $r .= 0.5; $CUDA.device!(0); $p .= 0.5; cond = 1.0)
end
