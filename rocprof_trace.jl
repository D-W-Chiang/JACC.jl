include("setup.jl")

using AMDGPU
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

SIZE = 10
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
p = p * 0.5
AMDGPU.device!(AMDGPU.devices()[2])
r = r * 0.5

# --- ONE SINGLE ITERATION TRACE ---

AMDGPU.ROCTX.range_push("CG_Single_Iteration")

AMDGPU.ROCTX.range_push("Copy 1")
copyto!(r_old, r)
AMDGPU.ROCTX.range_pop()

AMDGPU.ROCTX.range_push("Dispatch matvecmul")
JACC.Async.parallel_for(1, SIZE, matvecmul, a0, a1, a2, p, s1, SIZE)
AMDGPU.ROCTX.range_pop()

AMDGPU.ROCTX.range_push("Async Block 1")
alpha1 = JACC.Async.parallel_reduce(1, SIZE, dot, p, s1)
alpha0 = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
JACC.Async.synchronize()
AMDGPU.ROCTX.range_pop()

AMDGPU.ROCTX.range_push("Host calculation alpha")
alpha = JACC.to_host(alpha0)[] / JACC.to_host(alpha1)[]
negative_alpha = alpha * -1.0
AMDGPU.ROCTX.range_pop()

AMDGPU.ROCTX.range_push("Copy 2")
copyto!(s2, s1)
AMDGPU.ROCTX.range_pop()

AMDGPU.ROCTX.range_push("Async Block 2")
JACC.Async.parallel_for(1, SIZE, axpy, alpha, x, p)
JACC.Async.parallel_for(2, SIZE, axpy, negative_alpha, r, s2)
JACC.Async.synchronize()
AMDGPU.ROCTX.range_pop()

AMDGPU.ROCTX.range_push("Async Block 3")
beta1 = JACC.Async.parallel_reduce(1, SIZE, dot, r_old, r_old)
beta0 = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
JACC.Async.synchronize()
AMDGPU.ROCTX.range_pop()

AMDGPU.ROCTX.range_push("Host calculation beta")
beta = JACC.to_host(beta0)[] / JACC.to_host(beta1)[]
AMDGPU.ROCTX.range_pop()

AMDGPU.ROCTX.range_push("Copy 3")
copyto!(r_aux, r)
AMDGPU.ROCTX.range_pop()

AMDGPU.ROCTX.range_push("Async Block 4")
JACC.Async.parallel_for(1, SIZE, axpy, beta, r_aux, p)
ccond = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
JACC.Async.synchronize()
AMDGPU.ROCTX.range_pop()

AMDGPU.ROCTX.range_push("Host wrap up")
cond = JACC.to_host(ccond)[]
AMDGPU.ROCTX.range_pop()

AMDGPU.ROCTX.range_push("Copy 4")
copyto!(p, r_aux)
AMDGPU.ROCTX.range_pop()

# Pop the main iteration block
AMDGPU.ROCTX.range_pop()
