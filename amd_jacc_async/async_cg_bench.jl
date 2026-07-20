include("../setup.jl")

using BenchmarkTools

function matvecmul(i, a1, a2, a3, x, y, SIZE)
    if i == 1
        y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
    elseif i == SIZE
        y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
    elseif i > 1 && i < SIZE
        y[i] = a3[i] * x[i - 1] + a2[i] * x[i] + a1[i] * x[i + 1]
    end
end

devices = AMDGPU.devices!()

a0 = JACC.Async.ones(2, SIZE)
a1 = JACC.Async.ones(2, SIZE)
a2 = JACC.Async.ones(2, SIZE)
r = JACC.Async.ones(1, SIZE)
p = JACC.Async.ones(2, SIZE)
s1 = JACC.Async.zeros(2, SIZE)
s2 = JACC.Async.zeros(1, SIZE)
x = JACC.Async.zeros(2, SIZE)
r_old = JACC.Async.zeros(2, SIZE)
r_aux = JACC.Async.zeros(2, SIZE)

AMDGPU.device!(devices[1])
r .*= 0.5
AMDGPU.device!(devices[2])
a1 .*= 4
p .*= 0.5

result = @benchmark begin
    cond = 1.0

    while cond >= 1e-14
       copyto!($r_old, $r)

        JACC.Async.parallel_for(2, $SIZE, matvecmul, $a0, $a1, $a2, $p, $s1, $SIZE)

        alpha1 = JACC.Async.parallel_reduce(2, $SIZE, dot, $p, $s1)
        alpha0 = JACC.Async.parallel_reduce(1, $SIZE, dot, $r, $r)
        JACC.Async.synchronize()

        alpha = JACC.to_host(alpha0)[] / JACC.to_host(alpha1)[]
        negative_alpha = alpha * -1.0

        copyto!($s2, $s1)
        JACC.Async.parallel_for(2, $SIZE, axpy, alpha, $x, $p)
        JACC.Async.parallel_for(1, $SIZE, axpy, negative_alpha, $r, $s2)
        JACC.Async.synchronize()

        beta1 = JACC.Async.parallel_reduce(2, $SIZE, dot, $r_old, $r_old)
        beta0 = JACC.Async.parallel_reduce(1, $SIZE, dot, $r, $r)
        JACC.Async.synchronize()

        beta = JACC.to_host(beta0)[] / JACC.to_host(beta1)[]

        copyto!($r_aux, $r)

        JACC.Async.parallel_for(2, $SIZE, axpy, beta, $r_aux, $p)
        ccond = JACC.Async.parallel_reduce(1, $SIZE, dot, $r, $r)
        JACC.Async.synchronize()

        cond = JACC.to_host(ccond)[]

        copyto!($p, $r_aux)
    end
end evals=1 samples=10 gcsample=true setup=(AMDGPU.device!(devices[1]); $r .= 0.5; $s2 .= 0.0; AMDGPU.device!(devices[2]); $p .= 0.5; $s1 .= 0.0; $x .= 0.0; $r_old .= 0.0; $r_aux .= 0.0)

println(mean(result).time)
