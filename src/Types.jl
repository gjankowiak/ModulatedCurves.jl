struct Params
    # Discretization parameters
    N::Int64
    L::Float64

    # Model parameters
    M::Float64
    epsilon::Float64
    ρ_max::Float64
    c0::Float64

    # mode number
    mode_j::Int64

    # rho bound parameters
    potential_range::Float64
end

struct Stiffness
    beta::Function
    beta_prime::Function
    beta_second::Function
end

struct IntermediateParams
    Δs::Float64
    rho_eq::Float64
    beta_eq::Float64
    beta_eq_prime::Float64
    beta_eq_second::Float64
end

mutable struct SolverParams
    atol::Float64
    rtol::Float64
    max_iter::Int64
    step_size::Float64
    adapt::Bool
    min_step_size::Float64
    max_step_size::Float64
    step_down_threshold::Float64
    step_up_threshold::Float64
    step_factor::Float64
end

default_SP = SolverParams(1e-8,    # atol
                          1e-12,   # rtol
                          1000,    # max_iter
                          1e-3,    # step_size
                          false,   # adapt
                          1e-5,    # min_step_size
                          1e-1,    # max_step_size
                          1e-1,    # step_down_threshold
                          1e-4,    # step_up_threshold
                          1.2)     # step_factor

mutable struct Candidate
    ρ::Union{Vector{Float64},SubA}
    θ::Union{Vector{Float64},SubA}
    λx::Float64
    λy::Float64
    λM::Float64
    λcm::Float64
end

mutable struct History
    energy_prev::Float64
    residual_prev::Vector{Float64}
end

struct Result
    sol::Vector{Float64}
    iter::Int64
    energy_i::Vector{Float64}
    residual_norm_i::Vector{Float64}
    t_i::Vector{Float64}
    int_θ_i::Vector{Float64}
    converged::Bool
    finished::Bool
end

mutable struct Relaxation
    r::Float64
    threshold::Float64
    min_r::Float64
    max_r::Float64
end

struct FDMatrices
    M_phalf::SA.SparseMatrixCSC{Float64}
    M_mhalf::SA.SparseMatrixCSC{Float64}
    D1::SA.SparseMatrixCSC{Float64}
    D1_rhs::Vector{Float64}
    D1c::SA.SparseMatrixCSC{Float64}
    D1c_rhs::Vector{Float64}
    D2::SA.SparseMatrixCSC{Float64}
end
