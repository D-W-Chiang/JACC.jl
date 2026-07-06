jacc_data = Tuple{Int, Int}[]
jacc_async_data = Tuple{Int, Int}[]

for size in 200_000:200_000:1_000_000
    	println("Collecting data for SIZE = $size...")

	cmd_async = `julia -e "SIZE=$size; include(\"async_parallel_for_benchmark.jl\")"`
	async_str = readchomp(cmd_async)

	cmd_non_async = `julia -e "SIZE=$size; include(\"parallel_for_benchmark.jl\")"`
	non_async_str = readchomp(cmd_non_async)

	async_time = parse(Int, async_str)
	non_async_time = parse(Int, non_async_str)

	push!(jacc_async_data, (size, async_time))
	push!(jacc_data, (size, non_async_time))
end

println("jacc_data = ", jacc_data)
println("jacc_async_data = ", jacc_async_data)
