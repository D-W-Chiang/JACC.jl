include("setup.jl")

using BenchmarkTools

alpha_device = JACC.Async.ones(1, 1)

result = @benchmark begin
        for i = 1:10
            val = JACC.to_host($alpha_device)[]
        end
end evals=1 samples=1

println(round(Int, result.times[1]/10))
