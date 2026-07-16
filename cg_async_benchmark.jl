include("setup.jl")

using BenchmarkTools
using DelimitedFiles
using Dates
using Statistics

suite = BenchmarkGroup()

SIZE = 2_000_000

function matvecmul(i, a1, a2, a3, x, y, SIZE)
        if i == 1
            y[i] = a2[i] * x[i] + a1[i] * x[i + 1]
        elseif i == SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * x[i]
        elseif i > 1 && i < SIZE
            y[i] = a3[i] * x[i - 1] + a2[i] * +x[i] + a1[i] * +x[i + 1]
        end
    end

suite["cg_async"] = 

let 
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
	CUDA.device!(1)
        a1 = a1 * 4
	CUDA.device!(0)
        r = r * 0.5
	CUDA.device!(1)
        p = p * 0.5
        cond = 1.0

@benchmarkable begin 
    while cond[1, 1] >= 1e-14
        println(cond)
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
    println(cond)
end seconds = 300 samples = 1 evals = 1 gcsample = true setup = (CUDA.device!(1); $x .= 0.0; CUDA.device!(0); $r .= 0.5; $CUDA.device!(1); $p .= 0.5; cond = 1.0)
end

results = run(suite)

#=
#store times as a single, comma-separated row 
writedlm("times.csv", permutedims(results["cg_async"].times), ',')

#store metrics in a separate file
open("metrics.txt", "w") do file
	println(file, "SIZE: $SIZE")

	println(file, "samples: $(length(results["cg_async"].times))")

	println(file, "mean_gc: $(mean(results["cg_async"].gctimes)/1e9)")

	println(file, "min: $(minimum(results["cg_async"].times)/1e9)")
	println(file, "max: $(maximum(results["cg_async"].times)/1e9)")
	println(file, "median: $(median(results["cg_async"].times)/1e9)")
	println(file, "mean: $(mean(results["cg_async"].times)/1e9)")

	println(file, "variation: $(std(results["cg_async"].times)/1e9)")
end
=#
