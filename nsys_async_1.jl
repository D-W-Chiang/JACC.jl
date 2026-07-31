include("setup.jl")

using NVTX

# Set device FIRST before allocating any arrays
CUDA.device!(0)

SIZE = 4000000

# Allocate arrays on the active device
r  = JACC.Async.ones(1, SIZE)
p  = JACC.Async.ones(2, SIZE)
s1 = JACC.Async.zeros(2, SIZE)

# In-place scaling (avoids extra allocations)
r .*= 0.5
CUDA.device!(1)
p .*= 0.5

sleep(2)

# Warmup / force CUDA runtime compilation
_ = CUDA.zeros(10)
CUDA.synchronize()

# --- MAIN REDUCE LOOP ---
NVTX.@range "Reduce Loop" begin
    for iter in 1:11
        NVTX.@range "Single Iteration" begin
            a_res = JACC.Async.parallel_reduce(2, SIZE, dot, p, s1)
            b_res = JACC.Async.parallel_reduce(1, SIZE, dot, r, r)
            
            # Ensure GPU finishes this iteration before closing NVTX range
            JACC.Async.synchronize()
        end
    end
end

CUDA.synchronize()
sleep(1)
