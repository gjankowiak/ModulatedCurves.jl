using Base: @kwdef

@kwdef mutable struct Params
  # Discretization parameters
  N::Int64
  L::Float64

  # Model parameters
  M::Float64
  µ::Float64
  ρ_max::Float64
  c0::Float64

  # rho bound parameters
  potential_range::Float64
end

@kwdef struct Stiffness
  beta::Function
  beta_prime::Function
  beta_second::Function
end

@kwdef struct IntermediateParams
  Δs::Float64
  rho_eq::Float64
  beta_eq::Float64
  beta_eq_prime::Float64
  beta_eq_second::Float64
end

@kwdef mutable struct SolverParams
  atol::Float64 = 1e-9
  rtol::Float64 = 1e-13
  max_iter::Int64 = 5000
  step_size::Float64 = 1e-3
  adapt::Bool = false
  min_step_size::Float64 = 1e-4
  max_step_size::Float64 = 1e-1
  step_down_threshold::Float64 = 1e-1
  step_up_threshold::Float64 = 1e-3
  step_factor::Float64 = 1.4

  newton_step_size::Float64 = 1.0
  newton_max_iter::Int64 = 100

  abs_time_tol::Float64 = 1e-13

  max_time::Float64
end

mutable struct Candidate
  ρ::Union{Vector{Float64},SubA}
  θ::Union{Vector{Float64},SubA}
  λx::Float64
  λy::Float64
  λM::Float64
  λcm::Float64
end

@kwdef mutable struct History
  energy_prev::Float64
  residual_prev::Vector{Float64}
end

@kwdef struct Result
  sol::Vector{Float64}
  iter::Int64
  energy_i::Vector{Float64}
  energy_ρ_i::Vector{Float64}
  energy_θ_i::Vector{Float64}
  residual_norm_i::Vector{Float64}
  t_i::Vector{Float64}
  int_θ_i::Vector{Float64}
  converged::Bool
  finished::Bool
end

@kwdef struct FDMatrices
  M_phalf::SA.SparseMatrixCSC{Float64}
  M_mhalf::SA.SparseMatrixCSC{Float64}
  D1::SA.SparseMatrixCSC{Float64}
  D1_rhs::Vector{Float64}
  D1c::SA.SparseMatrixCSC{Float64}
  D1c_rhs::Vector{Float64}
  D2::SA.SparseMatrixCSC{Float64}
end

struct MaxInterationReached <: Exception end
struct TimeStepTooSmall <: Exception end
