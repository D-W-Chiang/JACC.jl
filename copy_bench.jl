include("setup.jl")

using BenchmarkTools

r = JACC.Async.ones(1, SIZE)
r_old = JACC.Async.zeros(2, SIZE)
r_aux = JACC.Async.zeros(2, SIZE)

r .*= 0.5

result = @benchmark begin
	for i = 1:10
		copyto!($r_old, $r)
	end
end evals=1 samples=1

println(round(Int, result.times[1]/10))
