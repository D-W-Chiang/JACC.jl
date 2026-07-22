times = Tuple{Float64, Float64}[]

for size in 2_000_000:2_000_000:20_000_000
        println("Collecting data for SIZE = $size...")
	
	time = `julia -e "SIZE=$size; include(\"to_host_bench.jl\")"`
        time_str = readchomp(time)

        println("time = ", time_str)

        time = parse(Float64, time_str)

        push!(times, (size, time))

end

println("data = ", times)
