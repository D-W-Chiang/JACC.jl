include("setup.jl")

using NVTX

function matvecmul(i, a1, a2, a3, x, y, SIZE)
            if i == 1
                y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
            elseif i == SIZE
                y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
            elseif i > 1 && i < SIZE
                y[i] = a3[i] * x[i - 1] + a2[i] * +x[i] + a1[i] * +x[i + 1]
            end
         end

SIZE = 4000000
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
CUDA.device!(1) 
r = r * 0.5


sleep(2)

CUDA.device!(1)

# 2. Force Julia to compile the CUDA runtime before your NVTX ranges start
_ = CUDA.zeros(10)
CUDA.synchronize()

# --- ONE SINGLE ITERATION TRACE ---

NVTX.@range "CG_Single_Iteration" begin
    
    NVTX.@range "Copy 1" copyto!(r_old, r)

    NVTX.@range "Dispatch matvecmul" begin
        JACC.Async.parallel_for(1, SIZE, matvecmul, a0, a1, a2, p, s1, SIZE)
    end
    
    NVTX.@range "Async Block 1" begin
        alpha1 = JACC.Async.parallel_reduce(1, SIZE, dot, p, s1)
        alpha0 = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
    	JACC.Async.synchronize()
    end    

    NVTX.@range "Host calculation alpha" begin
        alpha = JACC.to_host(alpha0)[] / JACC.to_host(alpha1)[]
        negative_alpha = alpha * -1.0
    end

    NVTX.@range "Copy 2" copyto!(s2, s1) 

    NVTX.@range "Async Block 2" begin
        JACC.Async.parallel_for(1, SIZE, axpy, alpha, x, p)
        JACC.Async.parallel_for(2, SIZE, axpy, negative_alpha, r, s2)
	JACC.Async.synchronize()
    end

    NVTX.@range "Async Block 3" begin
        beta1 = JACC.Async.parallel_reduce(1, SIZE, dot, r_old, r_old)
        beta0 = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
	JACC.Async.synchronize()
    end
    
    NVTX.@range "Host calculation beta" beta = JACC.to_host(beta0)[] / JACC.to_host(beta1)[]

    NVTX.@range "Copy 3" copyto!(r_aux, r)

    NVTX.@range "Async Block 4" begin
        JACC.Async.parallel_for(1, SIZE, axpy, beta, r_aux, p)
        ccond = JACC.Async.parallel_reduce(2, SIZE, dot, r, r)
	JACC.Async.synchronize()
    end

    NVTX.@range "Host wrap up" begin
        cond = JACC.to_host(ccond)[]
    end

    NVTX.@range "Copy 4" copyto!(p, r_aux)
    
end
