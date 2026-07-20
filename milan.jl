jacc_data = Tuple{Float64, Float64}[]
jacc_async_data = Tuple{Float64, Float64}[]

for size in 2_000_000:2_000_000:20_000_000
    	println("Collecting data for SIZE = $size...")

	cmd_non_async = `julia -e "SIZE=$size; include(\"jacc/cg_bench.jl\")"`
	non_async_str = readchomp(cmd_non_async)

	println("time = ", non_async_str)

	cmd_async = `julia -e "SIZE=$size; include(\"jacc_async/async_cg_bench.jl\")"`
        async_str = readchomp(cmd_async)

        println("async time = ", async_str)

	async_time = parse(Float64, async_str)
	non_async_time = parse(Float64, non_async_str)

	push!(jacc_async_data, (size, async_time))
	push!(jacc_data, (size, non_async_time))

end

println("jacc_data = ", jacc_data)
println("jacc_async_data = ", jacc_async_data)
