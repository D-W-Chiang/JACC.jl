include("setup.jl")

using BenchmarkTools
using DelimitedFiles
using Dates
using Statistics

suite = BenchmarkGroup()

SIZE = 800_000_000

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
	copyto!($r_old, $r)

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

	copyto!($r_aux, $r)

        JACC.parallel_for($SIZE, axpy, beta, $r_aux, $p)
        ccond = JACC.parallel_reduce($SIZE, dot, $r, $r)
        cond = ccond

	copyto!($p, $r_aux) #= used to be p = copy(r_aux) but 
        wrapping the loop in @benchmark changes the scope to be local =#
    end 
end seconds = 600 samples = 100 evals = 1 gcsample = true setup = ($x .= 0.0; $r .= 0.5; $p .= 0.5; cond = 1.0)
end


results = run(suite)

writedlm("times.csv", permutedims(results["cg"].times), ',')

open("metrics.txt", "w") do file
	println(file, "SIZE: $SIZE")

	println(file, "samples: $(length(results["cg"].times))")

	println(file, "mean_gc: $(mean(results["cg"].gctimes)/1e9)")

	println(file, "min: $(minimum(results["cg"].times)/1e9)")
	println(file, "max: $(maximum(results["cg"].times)/1e9)")
	println(file, "median: $(median(results["cg"].times)/1e9)")
	println(file, "mean: $(mean(results["cg"].times)/1e9)")

	println(file, "variation: $(std(results["cg"].times)/1e9)")
end
