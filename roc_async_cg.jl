include("setup.jl")

using AMDGPU # Provides AMDGPU.@roctx for rocprof

function matvecmul(i, a1, a2, a3, x, y, SIZE)
    if i == 1
        y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
    elseif i == SIZE
        y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
    elseif i > 1 && i < SIZE
        y[i] = a3[i] * x[i - 1] + a2[i] * +x[i] + a1[i] * +x[i + 1]
    end
end

SIZE = 100000
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
AMDGPU.device!(2) # Equivalent to CUDA.device!(1)
r = r * 0.5
cond = 1.0


sleep(2)

AMDGPU.device!(2)

# 2. Force Julia to compile the AMDGPU runtime before your ROCTX ranges start
_ = AMDGPU.zeros(10)
AMDGPU.synchronize()

# --- WARM-UP RUN (1 Iteration to trigger JIT compilation) ---
copyto!(r_old, r)
JACC.Async.parallel_for(1, SIZE, matvecmul, a0, a1, a2, p, s1, SIZE)
alpha1_w = JACC.Async.parallel_reduce(1, SIZE, dot, p, s1)
alpha0_w = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
JACC.Async.synchronize()

alpha_w = JACC.to_host(alpha0_w)[] / JACC.to_host(alpha1_w)[]
negative_alpha_w = alpha_w * -1.0

copyto!(s2, s1)
JACC.Async.parallel_for(1, SIZE, axpy, alpha_w, x, p)
JACC.Async.parallel_for(2, SIZE, axpy, negative_alpha_w, r, s2)
JACC.Async.synchronize()

beta1_w = JACC.Async.parallel_reduce(1, SIZE, dot, r_old, r_old)
beta0_w = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
JACC.Async.synchronize()

beta_w = JACC.to_host(beta0_w)[] / JACC.to_host(beta1_w)[]
copyto!(r_aux, r)
JACC.Async.parallel_for(1, SIZE, axpy, beta_w, r_aux, p)
ccond_w = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
JACC.Async.synchronize()
copyto!(p, r_aux)
AMDGPU.synchronize()


# --- RESET STATE (In-place assignment, no new arrays) ---
# r and p were initially 0.5. x was 0.0.
AMDGPU.device!(2)
r .= 0.5
AMDGPU.device!(1) # Equivalent to CUDA.device!(0)
p .= 0.5
x .= 0.0

# These arrays get fully overwritten inside the loop anyway,
# but we fill them with 0.0 to be mathematically identical to start state.
s1 .= 0.0
AMDGPU.device!(2)
s2 .= 0.0
AMDGPU.device!(1)
r_old .= 0.0
r_aux .= 0.0

cond = 1.0

AMDGPU.@roctx "CG_Solver_Loop" begin
    while cond >= 1e-14
        AMDGPU.@roctx "CG_Single_Iteration" begin
            AMDGPU.@roctx "Copy 1" copyto!(r_old, r)

            AMDGPU.@roctx "Dispatch matvecmul" begin
                JACC.Async.parallel_for(1, SIZE, matvecmul, a0, a1, a2, p, s1, SIZE)
            end

            AMDGPU.@roctx "Async Block 1" begin
                alpha1 = JACC.Async.parallel_reduce(1, SIZE, dot, p, s1)
                alpha0 = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
                JACC.Async.synchronize()
            end

            AMDGPU.@roctx "Host calculation alpha" begin
                alpha = JACC.to_host(alpha0)[] / JACC.to_host(alpha1)[]
                negative_alpha = alpha * -1.0
            end

            AMDGPU.@roctx "Copy 2" copyto!(s2, s1)

            AMDGPU.@roctx "Async Block 2" begin
                JACC.Async.parallel_for(1, SIZE, axpy, alpha, x, p)
                JACC.Async.parallel_for(2, SIZE, axpy, negative_alpha, r, s2)
                JACC.Async.synchronize()
            end

            AMDGPU.@roctx "Async Block 3" begin
                beta1 = JACC.Async.parallel_reduce(1, SIZE, dot, r_old, r_old)
                beta0 = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
                JACC.Async.synchronize()
            end

            AMDGPU.@roctx "Host calculation beta" beta = JACC.to_host(beta0)[] / JACC.to_host(beta1)[]

            AMDGPU.@roctx "Copy 3" copyto!(r_aux, r)

            AMDGPU.@roctx "Async Block 4" begin
                JACC.Async.parallel_for(1, SIZE, axpy, beta, r_aux, p)
                ccond = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
                JACC.Async.synchronize()
            end

            AMDGPU.@roctx "Host wrap up" begin
                global cond = JACC.to_host(ccond)[]
            end

            AMDGPU.@roctx "Copy 4" copyto!(p, r_aux)
        end
    end
end
