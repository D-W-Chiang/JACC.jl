include("setup.jl")

using NVTX
using CUDA

function matvecmul(i, a1, a2, a3, x, y, SIZE)
    if i == 1
        y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
    elseif i == SIZE
        y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
    elseif i > 1 && i < SIZE
        y[i] = a3[i] * x[i - 1] + a2[i] * x[i] + a1[i] * x[i + 1]
    end
end

SIZE = 1000000000
a0 = JACC.ones(SIZE)
a1 = JACC.ones(SIZE)
a2 = JACC.ones(SIZE)
r = JACC.ones(SIZE)
p = JACC.ones(SIZE)
s = JACC.zeros(SIZE)
x = JACC.zeros(SIZE)
r_old = JACC.zeros(SIZE)
r_aux = JACC.zeros(SIZE)

a1 .*= 4
r .*= 0.5
p .*= 0.5

sleep(2)

CUDA.device!(0)

# Force Julia to compile the CUDA runtime before your NVTX ranges start
_ = CUDA.zeros(10)
CUDA.synchronize()

# --- WARMUP (Variables suffixed with _w) ---
copyto!(r_old, r)

JACC.parallel_for(SIZE, matvecmul, a0, a1, a2, p, s, SIZE)

alpha0_w = JACC.parallel_reduce(SIZE, dot, r, r)
alpha1_w = JACC.parallel_reduce(SIZE, dot, p, s)

alpha_w = alpha0_w / alpha1_w
negative_alpha_w = alpha_w * -1.0

JACC.parallel_for(SIZE, axpy, negative_alpha_w, r, s)
JACC.parallel_for(SIZE, axpy, alpha_w, x, p)

beta0_w = JACC.parallel_reduce(SIZE, dot, r, r)
beta1_w = JACC.parallel_reduce(SIZE, dot, r_old, r_old)
beta_w = beta0_w / beta1_w

copyto!(r_aux, r)

JACC.parallel_for(SIZE, axpy, beta_w, r_aux, p)
ccond_w = JACC.parallel_reduce(SIZE, dot, r, r)

cond_w = ccond_w[1]

copyto!(p, r_aux)

# --- RESET ---
r .= 0.5
p .= 0.5
x .= 0.0
s .= 0.0
r_old .= 0.0
r_aux .= 0.0

cond = 1.0

# --- MAIN PROFILE LOOP (Globals removed) ---
NVTX.@range "CG_Solver_Loop" begin
    while cond >= 1e-14
        NVTX.@range "CG_Single_Iteration" begin

            NVTX.@range "Copy 1" copyto!(r_old, r)

            NVTX.@range "Dispatch matvecmul" JACC.parallel_for(SIZE, matvecmul, a0, a1, a2, p, s, SIZE)

            NVTX.@range "Block 1" begin
                alpha0 = JACC.parallel_reduce(SIZE, dot, r, r)
                alpha1 = JACC.parallel_reduce(SIZE, dot, p, s)
            end

            NVTX.@range "Host calculation alpha" begin
                alpha = alpha0 / alpha1
                negative_alpha = alpha * -1.0
            end

            # --- FILLED IN SECTION 1 ---
            NVTX.@range "Compute axpy x and r" begin
                JACC.parallel_for(SIZE, axpy, negative_alpha, r, s)
                JACC.parallel_for(SIZE, axpy, alpha, x, p)
            end

            NVTX.@range "Reduce beta" begin
                beta0 = JACC.parallel_reduce(SIZE, dot, r, r)
                beta1 = JACC.parallel_reduce(SIZE, dot, r_old, r_old)
            end

            NVTX.@range "Host calculation beta" begin
                beta = beta0 / beta1
            end

            # --- FILLED IN SECTION 2 ---
            NVTX.@range "Copy 2" copyto!(r_aux, r)

            NVTX.@range "Compute axpy p and ccond" begin
                JACC.parallel_for(SIZE, axpy, beta, r_aux, p)
                ccond = JACC.parallel_reduce(SIZE, dot, r, r)
            end

            NVTX.@range "Host wrap up" begin
                global cond = ccond[1]
            end

            # --- FILLED IN SECTION 3 ---
            NVTX.@range "Copy 3" copyto!(p, r_aux)

        end # end CG_Single_Iteration
    end # end while loop
end # end CG_Solver_Loop
