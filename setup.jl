using JACC
JACC.@init_backend

include("test/common.jl")
using Test
import LinearAlgebra
using ..JACCTestCommon: axpy, dot
