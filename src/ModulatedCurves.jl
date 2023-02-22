module ModulatedCurves

import Printf: @sprintf, @printf

export Params, Stiffness, IntermediateParams, SolverParams
export compute_intermediate, build_flower, assemble_fd_matrices, init_plot, do_flow, check_differential
export M
export params_from_toml, _compute_intermediate

import LinearAlgebra
const LA = LinearAlgebra

import SparseArrays
const SA = SparseArrays

import UnicodePlots

import HDF5

import Serialization

import FFTW

import TOML
import Symbolics

import JankoUtils: spdiagm_const, compute_ls_intersect

import EvenParam

const SubA = SubArray{Float64,1,Array{Float64,1}}

include("./Types.jl")
include("./Check.jl")
include("./Plotting.jl")

function copy_struct(s)
    T = typeof(s)
    fields = fieldnames(T)
    return T(map(x -> getfield(s, x), fields)...)
end

function _compute_intermediate(P::Params)
    Δs = P.L / P.N
    rho_eq = P.M / P.L

    return IntermediateParams(
                              Δs, rho_eq,
                              NaN, NaN, NaN
                             )
end

function compute_intermediate(P::Params, S::Stiffness)
    Δs = P.L / P.N
    rho_eq = P.M / P.L

    return IntermediateParams(
                              Δs, rho_eq,
                              S.beta(rho_eq), S.beta_prime(rho_eq), S.beta_second(rho_eq)
                             )
end

function prompt_yes_no(s::String, default_yes::Bool=false)
    while true
        print(s)
        if default_yes
            print(" [Y/n]: ")
        else
            print(" [y/N]: ")
        end
        i = readline()
        if (default_yes && i == "") || (i == "y")
            return true
        elseif (!default_yes && i == "") || (i == "n")
            return false
        end
    end
end

function X2candidate(P::Params, X; copy::Bool=false)
    if copy
        return Candidate(X[1:P.N],
                         X[P.N+1:2*P.N],
                         X[2*P.N+1],
                         X[2*P.N+2],
                         X[2*P.N+3],
                         0)
    else
        return Candidate(view(X, 1:P.N),
                         view(X, P.N+1:2*P.N),
                         X[2*P.N+1],
                         X[2*P.N+2],
                         X[2*P.N+3],
                         0)
    end
end

function candidate2X(P::Params, c::Candidate)
    return [c.ρ; c.θ; c.λx; c.λy; c.λM]
end

function prepare_fft_plans(n::Int)
    v_r = zeros(Float64, n)
    p = FFTW.plan_rfft(v_r)

    v_i = zeros(ComplexF64, length(p*zeros(n)))
    ip = FFTW.plan_irfft(v_i, n)

    freqs = abs.(FFTW.fftfreq(n, n)[1:n÷2+1])

    return (p = p, ip = ip, freqs = freqs)
end

function get_first_frequency(u, plans, symb)
    c = abs.(plans.p*u)
    if symb == :θ
        # we skip the first frequency for θ which is always non zero
        # since the integral of θ is preserved along the flow
        # we can mask the first Fourier coefficient of the increment
        idx = findfirst(x -> !isapprox(x, 0, atol=1e-10), view(c, 2:length(c))) + 1
    else
        idx = findfirst(x -> !isapprox(x, 0, atol=1e-10), c)
    end
    @info "Amplitudes for $(symb)"
    @info "First frequency: $(idx), $(plans.freqs[idx])"
    return idx, plans.freqs[idx]
end

function build_highpass_mask(u, plans, symb)
    idx, freq0 = get_first_frequency(u, plans, symb)
    return [i >= idx for i in 1:(length(u)÷2 + 1)]
end

function apply_fft_mask(u, mask, plans)
    return plans.ip*(mask .* (plans.p * u))
end

"""
    Structure of X:
    * ρ = X[1:P.N]
    * θ = X[P.N+1, 2*P.N]
    * λx = X[2*P.N+1]
    * λy = X[2*P.N+2]
    * λM = X[2*P.N+3]
"""

function adapt_step(P::Params, IP::IntermediateParams, S::Stiffness,
        SP::SolverParams, matrices::FDMatrices,
        X::Vector{Float64}, δX::Vector{Float64}, current_residual::Vector{Float64},
        current_time_step::Float64, history::History)

    local energy, residual

    Xnew = zeros(size(X))
    t = current_time_step

    check_step_up = false
    prev_t = -1

    residual_inc_ratio = Inf

    while true
        #
        # Update
        Xnew .= X + t*δX
        cnew = X2candidate(P, Xnew)

        # Check that rho is within admissible bounds
        # IF not, reduce step size
        if ((P.potential_range > 0) &&
            ((P.ρ_max >= 0 && (maximum(cnew.ρ) > P.ρ_max)) ||
             (P.ρ_max < 0 && (minimum(cnew.ρ) < P.ρ_max))))
            t /= 2
            print("Rho above bound, reducing relaxation parameter to: ")
            println(t)
            if t < SP.min_step_size
                error("Relaxation parameter became too small")
            end
            continue
        end

        # Compute residual and energy
        residual = compute_residual(P, IP, S, matrices, Xnew)
        energy = compute_energy(P, IP, S, matrices, Xnew)

        residual_inc_ratio = LA.norm(residual - history.residual_prev)/LA.norm(residual)

        if ((energy > history.energy_prev * (1 + 2e-1)) ||
            (residual_inc_ratio > SP.step_down_threshold))
            if (energy > history.energy_prev * (1 + 2e-1))
                reason = "energy going up"
            else
                reason = "residual ratio above threshold"
            end
            if check_step_up
                t = prev_t
            else
                t /= SP.step_factor^2
                print("Reducing relaxation parameter to: ")
                print(t)
                println(", reason: ", reason)
                if t < SP.min_step_size
                    error("Relaxation parameter became too small")
                end
            end
        else
            if check_step_up
                print("Increasing relaxation parameter to: ")
                println(t)
                @goto finish
            end
            if ((history.energy_prev > energy) &&
                # (LA.norm(residual) > 1e-3) &&
                (residual_inc_ratio < SP.step_up_threshold) &&
                (t < SP.max_step_size))
                prev_t = t
                t = min(SP.step_factor*t, SP.max_step_size)
                check_step_up = true
            else
                @goto finish
            end
        end
    end

    @label finish
    history.energy_prev = energy
    history.residual_prev = current_residual

    @debug @sprintf "Residual inc ratio: %.5f, δt: %.5f\n" residual_inc_ratio t

    return Xnew, residual, t
end

function compute_winding_number(P::Params, X::Vector{Float64})
    c = X2candidate(P, X)
    return round(Int, (c.θ[end] - c.θ[1])/2π)
end

function minimizor(P::Params, IP::IntermediateParams, S::Stiffness,
        Xinit::Vector{Float64}, SP::SolverParams=SolverParams())
    matrices = assemble_fd_matrices(P, IP; winding_number=compute_winding_number(P, Xinit))

    # Initialization
    history = History(0, zeros(2*P.N+3))

    X = Base.copy(Xinit)
    n = 1
    step_size = SP.step_size

    residual = compute_residual(P, IP, S, matrices, X)

    energy_i = zeros(SP.max_iter)

    energy_ρ_i = zeros(SP.max_iter)
    energy_θ_i = zeros(SP.max_iter)

    residual_norm_i = zeros(SP.max_iter)

    history.energy_prev = energy_i[1]
    history.residual_prev .= residual

    # Loop until tolerance or max. iterations is met
    function ()
        n += 1
        # Assemble
        A = assemble_inner_system(P, IP, S, matrices, X)

        # Solve
        δX = A\-residual

        if SP.adapt
            Xnew, residual, time_step = adapt_step(P, IP, S, SP, matrices,
                                                   X, δX,
                                                   residual, step_size,
                                                   history)
            X .= Xnew
        else
            # Update
            X += time_step*δX
            residual = compute_residual(P, IP, S, matrices, X)
        end

        # Compute residual norm and energy
        residual_norm = LA.norm(residual)
        energy_ρ_i[n], energy_θ_i[n] = compute_energy_split(P, IP, S, matrices, X)
        energy_i[n] = energy_ρ_i[n] + energy_θ_i[n]
        residual_norm_i[n] = residual_norm

        converged = (LA.norm(residual - history.residual_prev) < SP.rtol) || (LA.norm(residual) < SP.atol)

        history.energy_prev = energy_i[n]
        history.residual_prev = residual

        res = Result(X, n, energy_i, energy_ρ_i, energy_θ_i, residual_norm_i, converged, converged || (n>=SP.max_iter))
        return res
    end
end

function build_flower(P::Params, IP::IntermediateParams, S::Stiffness,
        Xinit::Vector{Float64}, SP::SolverParams=SolverParams(); include_multipliers::Bool=false,
        energy_circle::Float64=0.0, dump_system::Bool=false, dump_dir::String=".",
        highpass::Bool=false)

    winding_number = compute_winding_number(P, Xinit)

    matrices = assemble_fd_matrices(P, IP; winding_number=winding_number)

    s = collect(range(0, P.L, length=P.N+1))[1:P.N]

    # Initialization
    history = History(0, zeros(2*P.N+3))

    X = Base.copy(Xinit)
    Xj_prev = Base.copy(Xinit)
    Xj = Base.copy(Xinit)

    δX = Base.copy(Xinit)

    mm_term = zeros(size(X))

    X_up = view(X, 1:2P.N)
    Xj_up = view(Xj, 1:2P.N)
    mm_term_up = view(mm_term, 1:2P.N)

    c = X2candidate(P, X)
    n = 1

    time_step_size = SP.step_size
    newton_step_size = SP.newton_step_size

    residual = compute_residual(P, IP, S, matrices, X)
    residual_prev = Base.copy(residual)

    residual_norm = 1.0

    eρ, eθ = compute_energy_split(P, IP, S, matrices, X)
    energy_i = zeros(1+SP.max_iter)
    energy_ρ_i = zeros(1+SP.max_iter)
    energy_θ_i = zeros(1+SP.max_iter)

    energy_i[1], energy_ρ_i[1], energy_θ_i[1] = eρ + eθ, eρ, eθ

    residual_norm_i = [LA.norm(residual)]
    int_θ_i = [sum(c.θ)*IP.Δs]
    t_i = zeros(1+SP.max_iter)

    history.energy_prev = energy_i[1]
    history.residual_prev .= residual

    id_matrix = Matrix{Float64}(LA.I, 2P.N+3, 2P.N+3)
    if !include_multipliers
        for i in (2P.N:2P.N+3)
            id_matrix[i,i] = 0
        end
    end

    rho_extrema = extrema(c.ρ)

    if highpass
        fft_plans = prepare_fft_plans(P.N)
        θ_flat = c.θ - 2π*winding_number*s/P.L
        fft_mask_θ = build_highpass_mask(c.θ - winding_number*s, fft_plans, :θ)
        fft_mask_ρ = build_highpass_mask(c.ρ, fft_plans, :ρ)
        fft_mask = fft_mask_θ .|| fft_mask_ρ
    end


    # Loop until tolerance or max. iterations is met
    function _f()

        Xj_prev .= X
        Xj .= X

        n_newton = 0
        iter_stationary_res = 0

        newton_failed = false

        while true
            n_newton += 1
            # print(".")
            residual .= compute_residual(P, IP, S, matrices, Xj)

            # Assemble
            A = assemble_inner_system(P, IP, S, matrices, Xj)
            A_max = maximum(abs.(A))

            # Solve
            @. mm_term_up = X_up - Xj_up

            if dump_system
                open(joinpath(dump_dir, "A.dat"), "w") do f
                    Serialization.serialize(f, id_matrix .+ time_step_size*A)
                end
                open(joinpath(dump_dir, "b.dat"), "w") do f
                    Serialization.serialize(f, -time_step_size*residual + mm_term)
                end
            end

            δX .= (id_matrix .+ time_step_size*A)\(-time_step_size*residual + mm_term)

            if highpass
                δX[1:P.N] = apply_fft_mask(view(δX, 1:P.N), fft_mask, fft_plans)
                δX[P.N+1:2P.N] = apply_fft_mask(view(δX, P.N+1:2P.N), fft_mask, fft_plans)
                # TODO update Lagrange multipliers too
            end

            Xj_prev .= Xj

            @. Xj += newton_step_size*δX
            residual .= compute_residual(P, IP, S, matrices, Xj, X, time_step_size) # the NEW residual

            # Compute residual norm and energy
            residual_norm = LA.norm(residual)
            eρ, eθ = compute_energy_split(P, IP, S, matrices, Xj)

            residual_relative = LA.norm(residual - residual_prev)/residual_norm

            if residual_relative < SP.rtol
                @show LA.norm(residual - residual_prev)
                iter_stationary_res += 1
            else
                iter_stationary_res = 0
            end

            residual_prev .= residual
            @views step_norm = LA.norm(Xj[1:2P.N] - X[1:2P.N], Inf)

            converged = (iter_stationary_res > 10) || (LA.norm(residual)/P.N < SP.atol)

            if converged

                # adapt
                if SP.adapt
                    if step_norm < SP.step_up_threshold && !newton_failed
                        if time_step_size < SP.max_step_size
                            time_step_size = min(SP.max_step_size, time_step_size*SP.step_factor)
                            println()
                            println(@sprintf("|X-Xprev| = %.5e, time step ↑ (%f)", step_norm, time_step_size))
                        end
                    elseif step_norm > SP.step_down_threshold
                        if time_step_size == SP.min_step_size
                            throw(TimeStepTooSmall())
                        end
                        time_step_size = max(SP.min_step_size, time_step_size/SP.step_factor)
                        println()
                        println(@sprintf("|X-Xprev| = %.5e, time step ↓ (%f)", step_norm, time_step_size))
                        Xj_prev .= X
                        Xj .= X

                        n_newton = 0
                        iter_stationary_res = 0
                        continue
                    end
                end

                (intersect_i, intersect_j) = is_simple(P, IP, X)


                print(@sprintf("%s[%d/%d] t=%f, Eµ: %.5e/%.5e, |X-Xprev|: %.5e, %.2f ≤ ρ ≤ %.2f, (%d,%d)", "\b"^1000, n, n_newton, t_i[max(1,n-1)] + time_step_size, eρ+eθ, eρ+eθ-energy_circle, step_norm, rho_extrema[1], rho_extrema[2], intersect_i, intersect_j))
                println()
                break
            end

            if n_newton > SP.newton_max_iter
                if time_step_size == SP.min_step_size
                    throw(TimeStepTooSmall())
                end
                time_step_size = max(SP.min_step_size, time_step_size/SP.step_factor)
                println()
                println(@sprintf("[Newton failed] |X-Xprev| = %.5e, time step ↓ (%f)", step_norm, time_step_size))
                Xj_prev .= X
                Xj .= X

                newton_failed = true

                n_newton = 0
                iter_stationary_res = 0
                continue
                # println()
                # throw(MaxInterationReached())
            end
        end

        # X += time_step_size*Xj
        step_norm = LA.norm(X - Xj, Inf)
        X .= Xj

        rho_extrema = (min(rho_extrema[1], minimum(c.ρ)),
                       max(rho_extrema[2], maximum(c.ρ)))

        n += 1

        eρ, eθ = compute_energy_split(P, IP, S, matrices, X)
        energy = eρ + eθ
        energy_i[n] = energy
        energy_ρ_i[n] = eρ
        energy_θ_i[n] = eθ

        t_i[n] = t_i[n-1] + time_step_size

        residual_norm = LA.norm(residual)
        push!(residual_norm_i, residual_norm)

        push!(int_θ_i, sum(c.θ)*IP.Δs)

        history.energy_prev = energy
        history.residual_prev = residual

        res = Result(X, n, energy_i, energy_ρ_i, energy_θ_i, residual_norm_i, t_i, int_θ_i, step_norm < SP.abs_time_tol, n>=SP.max_iter || t_i[n] >= SP.max_time || step_norm < 1e-12)

        return res
    end
    return _f, matrices
end

function compute_energy_split(P::Params, IP::IntermediateParams, S::Stiffness,
        matrices::FDMatrices, X::Vector{Float64})
    c = X2candidate(P, X)
    ρ_dot = (circshift(c.ρ, -1) - c.ρ)/IP.Δs
    θ_dot = compute_centered_fd_θ(P, matrices, c.θ)
    beta = S.beta.(c.ρ)
    E_ρ, E_θ = 0.5*P.µ*IP.Δs*sum(ρ_dot.^2), 0.5IP.Δs*sum(beta.*(θ_dot .- P.c0).^2)
    return (E_ρ, E_θ)
end

function compute_energy(P::Params, IP::IntermediateParams, S::Stiffness,
        matrices::FDMatrices, X::Vector{Float64})
    return sum(compute_energy_split(P, IP, S, matrices, X))
end

function assemble_inner_system(P::Params, IP::IntermediateParams, S::Stiffness,
        matrices::FDMatrices, X::Vector{Float64})
    N = P.N
    Δs = IP.Δs

    # Easier access to ρ, θ, λi
    c = X2candidate(P, X)

    # Compute beta and derivatves
    beta_prime_ρ = S.beta_prime.(c.ρ)
    beta_second_ρ = S.beta_second.(c.ρ)
    beta_phalf = S.beta.(matrices.M_phalf*c.ρ)
    beta_a1half = S.beta.(matrices.M_mhalf*c.ρ)
    beta_prime_phalf = S.beta_prime.(matrices.M_phalf*c.ρ)
    beta_prime_mhalf = S.beta_prime.(matrices.M_mhalf*c.ρ)

    # Compute upstream and downstream finite differences of θ
    _, θ_prime_up, θ_prime_down = compute_fd_θ(P, matrices, c.θ)
    θ_prime_centered = compute_centered_fd_θ(P, matrices, c.θ)

    A_E1_ρ = P.µ * matrices.D2 - 0.5*beta_second_ρ.*(θ_prime_centered .- P.c0).^2 .*Matrix{Float64}(LA.I, N, N)
    if P.potential_range > 0
        (_, _, w_second) = compute_potential(P, X)
        A_E1_ρ .-= w_second .* Matrix{Float64}(LA.I, N, N)
    end
    # FIXME should be minus sign
    A_E1_θ = -beta_prime_ρ.*(θ_prime_centered .- P.c0).*matrices.D1c

    A_E1_λM = ones(N)

    A_E2_ρ  = (beta_prime_phalf.*(θ_prime_down .- P.c0).*matrices.M_phalf - beta_prime_mhalf.*(θ_prime_up .- P.c0).*matrices.M_mhalf)/Δs
    A_E2_θ  = (beta_phalf.*matrices.D1[2:N+1,:] - beta_a1half.*matrices.D1[1:N,:])/Δs - (c.λx*cos.(c.θ) + c.λy*sin.(c.θ)).*Matrix{Float64}(LA.I, N, N)

    A_E2_λx = -sin.(c.θ)
    A_E2_λy =  cos.(c.θ)

    A_c1θ = -Δs*sin.(c.θ)'
    A_c2θ = Δs*cos.(c.θ)'
    A_c3ρ = Δs*ones(N)'

    A = [[ A_E1_ρ    A_E1_θ        zeros(N) zeros(N)   A_E1_λM ];
         [ A_E2_ρ    A_E2_θ        A_E2_λx  A_E2_λy    zeros(N) ];
         [ zeros(N)' A_c1θ         0        0          0 ];
         [ zeros(N)' A_c2θ         0        0          0 ];
         [ A_c3ρ     zeros(N)'     0        0          0 ]]

    return -A
end

function compute_residual(P::Params, IP::IntermediateParams, S::Stiffness,
        matrices::FDMatrices, Xj_prev::Vector{Float64})
    N = P.N
    Δs = IP.Δs

    # Easier access to ρ, θ, λi
    c = X2candidate(P, Xj_prev)

    # Compute beta and derivatves
    beta_prime_ρ = S.beta_prime.(c.ρ)
    beta_phalf = S.beta.(matrices.M_phalf*c.ρ)
    beta_a1half = S.beta.(matrices.M_mhalf*c.ρ)

    # Compute upstream and downstream finite differences of θ
    _, θ_prime_up, θ_prime_down = compute_fd_θ(P, matrices, c.θ)

    # Compute residuals for ρ
    b_E1 = P.µ * (matrices.D2 * c.ρ) - 0.5*beta_prime_ρ.*(compute_centered_fd_θ(P, matrices, c.θ) .- P.c0).^2 + c.λM*ones(N)
    if P.potential_range > 0
        (_, w_prime, _) = compute_potential(P, Xj_prev)
        b_E1 -= w_prime
    end

    # Compute residuals for θ
    b_E2 = (beta_phalf.*(θ_prime_down .- P.c0) - beta_a1half.*(θ_prime_up .- P.c0))/Δs - c.λx*sin.(c.θ) + c.λy*cos.(c.θ)

    # Assemble with constraint residuals
    res = [b_E1; b_E2; Δs*sum(cos.(c.θ)); Δs*sum(sin.(c.θ)); Δs*sum(c.ρ) - P.M]
    return -res
end

function compute_residual(P::Params, IP::IntermediateParams, S::Stiffness,
        matrices::FDMatrices, Xj_prev::Vector{Float64}, X::Vector{Float64}, time_step::Float64)
    residual = compute_residual(P, IP, S, matrices, Xj_prev)
    @views residual[1:2P.N] += (Xj_prev[1:2P.N] - X[1:2P.N])/time_step
    return residual
end

function compute_fd_θ(P::Params, matrices::FDMatrices, θ::Union{Vector{Float64}, SubA})
    fd = matrices.D1*θ + matrices.D1_rhs
    return (fd, view(fd, 1:P.N), view(fd, 2:P.N+1))
end

function compute_centered_fd_θ(P::Params, matrices::FDMatrices, θ::Union{Vector{Float64}, SubA})
    return matrices.D1c*θ + matrices.D1c_rhs
end

function assemble_fd_matrices(P::Params, IP::IntermediateParams; winding_number::Int64=1)
    N = P.N
    Δs = IP.Δs

    w = winding_number

    M = FDMatrices(
                   # Matrices for i+1/2 and i-1/2 values
                   spdiagm_const([0.5, 0.5], [0, 1], N),
                   spdiagm_const([0.5, 0.5], [-1, 0], N),

                   # Matrix for upstream and downstream 1st order finite diff
                   # Taking into account 0 and 2π boundary conditions
                   # Size (N+1 x N-1)
                   # The finite differences can be computed using the
                   # compute_fd_θ function
                   (SA.spdiagm( -N  =>  ones(1),
                               -1   => -ones(N),
                               0   =>  ones(N+1),
                               N-1 => -ones(2))[1:N+1,1:N])/Δs,

                   # Affine term for up/downstream finite differences
                   [w*2π/Δs; zeros(N-1); w*2π/Δs],

                   # Matrix for centered finite differences
                   (SA.spdiagm(1-N =>  ones(1),
                               -1  => -ones(N-1),
                               1  =>  ones(N-1),
                               N-1 => -ones(1)))*0.5/Δs,

                   # Affine term for centered finite differences
                   [w*π/Δs; zeros(N-2); w*π/Δs],

                   # Matrix for 2nd order finite difference
                   spdiagm_const([1.0, -2.0, 1.0], [-1, 0, 1], N)/Δs^2
                  )

    return M
end


function g(x::Vector{Float64}, α::Float64)
    return @. -min(α*x-1, 0.0)^2 * log(α*x)
end

function g_p(x::Vector{Float64}, α::Float64)
    return @. -(2α*min(α*x-1, 0.0)*log(α*x) + min(α*x-1, 0.0)^2/x)
end

function g_pp(x::Vector{Float64}, α::Float64)
    return @. -(2α^2*(x<(1/α))*log(α*x) + 4α*min(α*x-1, 0.0)/x - min(α*x-1, 0.0)^2/x^2)
end

function compute_potential(P::Params, X::Vector{Float64})
    α = 1/P.potential_range
    ρ_max = P.ρ_max
    c = X2candidate(P, X)

    factor = 1e1

    # Hack to cope with lack of ρ_min
    # If ρ_max is negative, with interpret -ρ_max as ρ_min
    s = sign(ρ_max)

    w = factor*g(s*(s*ρ_max .- c.ρ), α)
    w_prime = factor*g_p(s*(s*ρ_max .- c.ρ), α)
    w_second = factor*g_pp(s*(s*ρ_max .- c.ρ), α)

    return (w, w_prime, w_second)
end


function linreg(xi, fi)
    n = length(xi)
    a = ((sum(fi.*xi) - sum(xi)*sum(fi)/n)/(sum(xi.^2)))/(1 - sum(xi).^2/sum(xi.^2)/n)
    b = sum(fi .- a*xi)/n

    return (a, b)
end

function save_snapshot(h5_fn::String, P::Params, next_snapshot_idx::Int64, X::Vector{Float64}, t::Float64, n::Int64, energy_ρ::Float64, energy_θ::Float64)
    @info "Saving iteration $n"
    h5 = HDF5.h5open(h5_fn, "r+")

    HDF5.create_dataset(h5, "s$(next_snapshot_idx)X", Float64, (2P.N+3,))
    write(h5["s$(next_snapshot_idx)X"], X)

    HDF5.create_dataset(h5, "s$(next_snapshot_idx)n", Int64)
    write(h5["s$(next_snapshot_idx)n"], n)

    HDF5.create_dataset(h5, "s$(next_snapshot_idx)t", Float64)
    write(h5["s$(next_snapshot_idx)t"], t)

    HDF5.create_dataset(h5, "s$(next_snapshot_idx)energy_rho", Float64)
    write(h5["s$(next_snapshot_idx)energy_rho"], energy_ρ)

    HDF5.create_dataset(h5, "s$(next_snapshot_idx)energy_theta", Float64)
    write(h5["s$(next_snapshot_idx)energy_theta"], energy_θ)

    write(h5["s_last_idx"], next_snapshot_idx)

    close(h5)
end

function compute_xy(P::Params, IP::IntermediateParams, X::Vector{Float64})
    c = X2candidate(P, X)
    xy = zeros(P.N+1, 2)

    for i in 2:P.N+1
        xy[i,:] = xy[i-1,:] + IP.Δs*[cos(c.θ[i-1]); sin(c.θ[i-1])]
    end

    bc = sum(xy, dims=1)/P.N
    xy = xy .- bc

    return xy
end

function is_simple(P::Params, IP::IntermediateParams, X::Vector{Float64})
    xy = compute_xy(P, IP, X)
    for i in 1:P.N
        for j in i+2:P.N
            if i == 1 && j == P.N
                # the first segment and the last segment
                # intersect at the origin, skip this case
                continue
            end
            xx = compute_ls_intersect(xy[i,:], xy[i+1,:], xy[j,:], xy[j+1,:])
            if !ismissing(xx[1])
                return (i, j)
            end
        end
    end
    return (-1,-1)
end

function save_snapshot(h5_fn::String, P::Params, next_snapshot_idx::Int64, res::Result; n_last::Int64=0)
    n = res.iter
    @info "Saving iteration $n"

    h5 = HDF5.h5open(h5_fn, "r+")
    HDF5.create_dataset(h5, "s$(next_snapshot_idx)X", Float64, (2P.N+3,))
    write(h5["s$(next_snapshot_idx)X"], res.sol)

    HDF5.create_dataset(h5, "s$(next_snapshot_idx)n", Int64)
    write(h5["s$(next_snapshot_idx)n"], n)

    HDF5.create_dataset(h5, "s$(next_snapshot_idx)t", Float64)
    write(h5["s$(next_snapshot_idx)t"], res.t_i[n])

    HDF5.create_dataset(h5, "s$(next_snapshot_idx)energy_rho", Float64)
    write(h5["s$(next_snapshot_idx)energy_rho"], res.energy_ρ_i[n])

    HDF5.create_dataset(h5, "s$(next_snapshot_idx)energy_theta", Float64)
    write(h5["s$(next_snapshot_idx)energy_theta"], res.energy_θ_i[n])

    write(h5["s_last_idx"], next_snapshot_idx)

    write(h5["energy_rho"], res.energy_ρ_i)
    write(h5["energy_theta"], res.energy_θ_i)
    write(h5["t"], res.t_i)
    # h5["energy_rho"] = res.energy_ρ_i
    # h5["energy_theta"] = res.energy_θ_i

    if n_last > 0
        HDF5.create_dataset(h5, "n_last", Int64)
        write(h5["n_last"], res.iter)
    end

    close(h5)
end

function do_flow(P::Params, S::Stiffness, Xinit::Vector{Float64}, solver_params::SolverParams;
        do_plot::Bool=false, include_multipliers::Bool=false, record_movie::Bool=false, pause_after_init::Bool=false,
        plot_each_iter::Int64=1, pause_after_plot::Bool=false, snapshots_iters::Vector{Int}=Int[],
        snapshots_times::Vector{Float64}=Float64[], output_dir::String="output", dump_system::Bool=false, rho_factor=2.0, plot_range=1.2, highpass::Bool=false, kwargs...)

    IP = compute_intermediate(P, S)

    mkpath(output_dir)

    plain_plot = !isempty(snapshots_iters) || !isempty(snapshots_times) && !record_movie

    wn = compute_winding_number(P, Xinit)

    ρ_equi = P.M / P.L

    if wn != 0
        k_equi = P.L / (wn*2π)
        energy_circle = P.L * S.beta(ρ_equi) * (k_equi - P.c0)^2 / 2
    else
        k_equi = 0.0
        energy_circle = 8.94665
    end

    Xx = copy(Xinit)
    Xx_prev = copy(Xinit)

    flower, matrices = build_flower(P, IP, S, Xx, solver_params;
                                    include_multipliers=include_multipliers, energy_circle=energy_circle,
                                    dump_system=dump_system, dump_dir=output_dir, highpass=highpass)

    eρ, eθ = compute_energy_split(P, IP, S, matrices, Xinit)

    @info "E0 = $(eρ + eθ)"
    @info "Eρ = $eρ"
    @info "Eθ = $eθ"

    h5_fn = joinpath(output_dir, "data.hdf5")
    HDF5.h5open(h5_fn, "w") do h5
        h5["rho_equi"] = ρ_equi
        h5["k_equi"] = k_equi
        h5["energy_circle"] = energy_circle

        HDF5.create_dataset(h5, "s_last_idx", Int64)

        HDF5.create_dataset(h5, "energy_rho",   Float64, (1+solver_params.max_iter,))
        HDF5.create_dataset(h5, "energy_theta", Float64, (1+solver_params.max_iter,))
        HDF5.create_dataset(h5, "t", Float64, (1+solver_params.max_iter,))
    end

    save_snapshot(h5_fn, P, 0, Xinit, 0.0, 0, eρ, eθ)
    next_snapshot_idx = 1

    if do_plot
        fig, update_plot = init_plot(P, IP, S, Xx; plain=plain_plot, rho_factor=rho_factor, plot_range=plot_range)
        if length(snapshots_iters) + length(snapshots_times) > 0
            save_path = joinpath(output_dir, "snapshots", "n=00000_t=0.pdf")
            if string(M) == "CairoMakie"
                @info "Saving initial snapshot under $save_path"
                M.save(save_path, fig)
            end
        end
        if pause_after_init
            println("Hit enter to start.")
            readline()
        end
    end


    local res

    done = [false]

    iter = 1:(solver_params.max_iter-1)

    next_snapshot_time = Inf
    next_snapshot_time_idx = -1
    if length(snapshots_times) > 0
        next_snapshot_time_idx = 1
        next_snapshot_time = snapshots_times[next_snapshot_time_idx]
    end

    if record_movie
        vs = M.VideoStream(fig, framerate=10)
        M.recordframe!(vs)
    end

    function iter_flower(_i)
        res = flower()

        Xx .= res.sol
        n = res.iter

        if n % plot_each_iter == 0
            if do_plot
                update_plot(res.sol, (@sprintf "t=%f, %d, energy/rel: %.4f/%.4f" res.t_i[n] n res.energy_i[n]  (res.energy_i[n] - energy_circle)), res)
                if record_movie
                    M.recordframe!(vs)
                end
                if pause_after_plot
                    print("Press Enter to continue ")
                    readline()
                end
            end
            sleep(0.01)
        else
            # print(".")
        end

        step_norm = LA.norm(Xx - Xx_prev, Inf)
        Xx_prev .= Xx

        if res.finished || res.residual_norm_i[n] > 1e10 || (step_norm < solver_params.abs_time_tol)
            println()
            print("Finished, converged: ")
            println(res.converged)
            if do_plot
                update_plot(res.sol, (@sprintf "t=%f, %d, energy/rel: %.4f/%.4f" res.t_i[n] n res.energy_i[n]  (res.energy_i[n] - energy_circle)), res)
                if record_movie
                    M.recordframe!(vs)
                end
                rounded_t = round(res.t_i[n]; digits=5)
                save_path = joinpath(output_dir, "snapshots", "n=$(lpad(res.iter, 5, "0"))_t=$(rounded_t).last.pdf")
                if string(M) == "CairoMakie"
                    @info "Saving final snapshot under $save_path"
                    M.save(save_path, fig)
                end
            end
            done[1] = true

            try
                save_snapshot(h5_fn, P, next_snapshot_idx, res; n_last=res.iter)
            catch
                return
            end
        else
            if n in snapshots_iters
                rounded_t = round(res.t_i[n]; digits=5)
                save_path = joinpath(output_dir, "snapshots", "n=$(lpad(n, 5, "0"))_t=$(rounded_t).pdf")
                println()
                if string(M) == "CairoMakie"
                    @info "Saving snapshot under $save_path"
                    M.save(save_path, fig)
                end
                save_snapshot(h5_fn, P, next_snapshot_idx, res)
                next_snapshot_idx += 1
            end

            if res.t_i[n] > next_snapshot_time
                rounded_t = round(res.t_i[n]; digits=5)
                save_path = joinpath(output_dir, "snapshots", "n=$(lpad(n, 5, "0"))_t=$(rounded_t).pdf")
                println()
                if string(M) == "CairoMakie"
                    @info "Saving snapshot under $save_path"
                    M.save(save_path, fig)
                end
                save_snapshot(h5_fn, P, next_snapshot_idx, res)
                next_snapshot_idx += 1
                if length(snapshots_times) > next_snapshot_time_idx
                    next_snapshot_time_idx += 1
                    next_snapshot_time = snapshots_times[next_snapshot_time_idx]
                else
                    next_snapshot_time_idx = -1
                    next_snapshot_time = Inf
                end
            end
        end
    end

    while !done[1]
        iter_flower(1)
    end
    if record_movie
        # GLMakie.record(iter_flower, fig, joinpath(output_dir, "output.mp4"), iter, framerate=10)
        M.save(joinpath(output_dir, "output.mp4"), vs)
    end
end

function initial_data_smooth(P::Params; sides::Int=1, smoothing::Float64, reverse_phase::Bool=false, only_rho::Bool=false,
        rho_amplitude::Float64=1.0, rho_phase::Float64=0.0, rho_wave_number::Int64=0)
    if P.N % sides != 0
        error("N must be dividible by the number of sides")
    end

    rwn = rho_wave_number == 0 ? sides : rho_wave_number

    if !(0 <= smoothing <= 1)
        error("smoothing parameter must be between 0 and 1")
    end

    Δs = P.L/P.N

    k = Int64(P.N / sides)
    N_straight = floor(Int64, P.N/sides*(1-smoothing))

    if only_rho
        thetas = collect(range(0, 2π, length=P.N+1))[1:end-1]
    else
        thetas_1p = [zeros(N_straight); [2π/sides - i*Δs/smoothing for i in (k-N_straight-1):-1:0]]
        thetas = repeat(thetas_1p, sides) + repeat(2π/sides*(0:sides-1), inner=k)
    end

    rhos = [cos.(2π*φ*rwn/P.L) for φ in range(0, 2π, P.N+1)[1:P.N]]
    extrema_rho = extrema(rhos)
    p2p_rho = extrema_rho[2] - extrema_rho[1]
    if p2p_rho > 1e-7
        rhos .= (rhos .- extrema_rho[1])/p2p_rho*rho_amplitude
    else
        @warn "The provided function r has very low amplitude, not rescaling ρ"
    end

    rhos .-= (sum(rhos)/P.N - P.M/P.N)

    return [rhos; thetas; 0; 0; 0]
end

function initial_data_parametric(P::Params, p::Function, rho_function::Union{Function,Nothing}, p_params::NamedTuple)
    N = P.N

    rho_amplitude = p_params.rho_amplitude
    rho_phase = p_params.rho_phase

    # oversample the curve
    oversample_N = 100N
    ts = collect(range(p_params.t_min, p_params.t_max, length=oversample_N + 1))[1:oversample_N]

    p_parameterized = (φ) -> p(φ, p_params)

    xy = permutedims(hcat(p_parameterized.(ts)...))

    # resample to get evenly spaced nodes
    # xy_even = EvenParam.reparam(xy; closed=true, new_N = N+1)[1:N,:]
    xy_even = EvenParam.reparam(xy; closed=true, new_N = N)

    # Recenter
    xy_even = xy_even .- [xy_even[1,1] xy_even[1,2]]

    # Rotate so that the first segment is parallel to the x axis
    xy_even_c = xy_even[:,1] + xy_even[:,2]*im
    alpha = angle(xy_even_c[2])
    xy_even_c = xy_even_c.*exp.(-im*alpha)

    # Compute tangential angles
    xy_even_c = [real.(xy_even_c) imag.(xy_even_c)]
    xy_diff = circshift(xy_even_c, -1) - xy_even_c
    xy_angles = angle.(xy_diff[:,1]+xy_diff[:,2]*im)

    thetas = zeros(N)
    thetas[1] = xy_angles[1]
    for i in 2:N
        thetas[i] = thetas[i-1] + rem(xy_angles[i] - thetas[i-1], Float64(π), RoundNearest)
    end

    if isnothing(rho_function)
        rhos = vec(sqrt.(sum(abs2, xy_even; dims=2)))
        extrema_rho = extrema(rhos)
        p2p_rho = extrema_rho[2] - extrema_rho[1]
        if p2p_rho > 1e-7
            rhos .= (rhos .- extrema_rho[1])/p2p_rho*rho_amplitude
        else
            @warn "The provided function p has very low amplitude, not rescaling ρ"
        end

        rhos .-= (sum(rhos)/P.N - P.M/P.N)
    else
        rho_parameterized = (t) -> rho_function(t, p_params)
        _ts = collect(range(p_params.t_min, p_params.t_max, length=P.N + 1))[1:P.N]
        rhos = rho_parameterized.(_ts)
        rhos ./= (P.L/P.N*sum(rhos))/P.M

        @show (P.L/P.N*sum(rhos))

        rhos_extrema = extrema(rhos)

        println("$(rhos_extrema[1]) ≤ ρ0 ≤ $(rhos_extrema[2])")
    end

    return [rhos; thetas; 0; 0; 0]
end

function initial_data_polar(P::Params, r::Function, r_params::NamedTuple)
    N = P.N

    rho_amplitude = r_params.rho_amplitude
    rho_phase = r_params.rho_phase

    # oversample the curve
    oversample_N = 100N
    φs = collect(range(r_params.s_min, r_params.s_max, length=oversample_N + 1))[1:oversample_N]

    r_parameterized = (φ) -> r(φ, r_params)

    x = r_parameterized.(φs).*cos.(φs)
    y = r_parameterized.(φs).*sin.(φs)
    xy = [x y]

    # resample to get evenly spaced nodes
    # xy_even = EvenParam.reparam(xy; closed=true, new_N = N+1)[1:N,:]
    xy_even = EvenParam.reparam(xy; closed=true, new_N = N)

    # Recenter
    xy_even = xy_even .- [xy_even[1,1] xy_even[1,2]]

    # Rotate so that the first segment is parallel to the x axis
    xy_even_c = xy_even[:,1] + xy_even[:,2]*im
    alpha = angle(xy_even_c[2])
    xy_even_c = xy_even_c.*exp.(-im*alpha)

    # Compute tangential angles
    xy_even = [real.(xy_even_c) imag.(xy_even_c)]
    xy_diff = circshift(xy_even, -1) - xy_even
    xy_angles = angle.(xy_diff[:,1]+xy_diff[:,2]*im)

    thetas = zeros(N)
    thetas[1] = xy_angles[1]
    for i in 2:N
        thetas[i] = thetas[i-1] + rem(xy_angles[i] - thetas[i-1], Float64(π), RoundNearest)
    end

    φs = range(0, 2π, N+1)[1:N]
    # rhos = r_parameterized.(φs .+ rho_phase)
    rhos = sin.(r_params.rho_wave_number*(φs .+ rho_phase))
    extrema_rho = extrema(rhos)
    p2p_rho = extrema_rho[2] - extrema_rho[1]
    if p2p_rho > 1e-7
        rhos .= (rhos .- extrema_rho[1])/p2p_rho*2rho_amplitude
    else
        @warn "The provided function r has very low amplitude, not rescaling ρ"
    end

    rhos .-= (sum(rhos)/P.N - P.M/P.N)

    return [rhos; thetas; 0; 0; 0]
end

function eval_if_string(v)
    if typeof(v) == String
        return eval(Meta.parse(v))
    else
        return v
    end
end

function initial_data_from_dict(P, d, x_functions)
    d_params = d["parameters"][d["type"]]

    if d["type"] == "polar"
        # Convert Dict{String,Any} to NamedTuple
        r_params = NamedTuple{Tuple(Symbol.(keys(d_params)))}((values(d_params)))
        return initial_data_polar(P, x_functions.r_func, r_params)
    elseif d["type"] == "polygon"
        kw = filter(x -> !(x[1] in [:type, :r_key]), collect(zip(map(Symbol, collect(keys(d_params))), values(d_params))))
        return initial_data_smooth(P; kw...)
    elseif d["type"] == "parametric"
        p_params = NamedTuple{Tuple(Symbol.(keys(d_params)))}((values(d_params)))
        return initial_data_parametric(P, x_functions.p_func, x_functions.rho_func, p_params)
    else
        throw("Unknown initial data type")
    end
end

function params_from_toml(T::Dict)
    # Model parameters
    N  = eval_if_string(T["model"]["N"])
    L  = eval_if_string(T["model"]["L"])
    M  = eval_if_string(T["model"]["M"])
    c0 = eval_if_string(T["model"]["c0"])
    mu = eval_if_string(T["model"]["mu"])

    # unused
    ρ_max = 1000
    potential_range = -1

    return Params(N=N, L=L, M=M, c0=c0, μ=mu, ρ_max=ρ_max, potential_range=potential_range)
end

function parse_configuration(filename::String, function_defs::Dict)
    T = TOML.parsefile(filename)
    x = Symbolics.variable(:x)

    P = params_from_toml(T)

    # Solver parameters
    solver_params = SolverParams(;zip(map(Symbol, collect(keys(T["solver"]))), values(T["solver"]))...)

    # Stiffness parameters
    stiffness_key = T["model"]["stiffness"]["key"]
    stiffness_params_dict = T["model"]["stiffness"]["parameters"][stiffness_key]
    # Convert Dict{String,Any} to NamedTuple
    stiffness_params = NamedTuple{Tuple(Symbol.(keys(stiffness_params_dict)))}((values(stiffness_params_dict)))

    # Stiffness and initial data r(φ) functions

    local beta = function_defs[:beta][stiffness_key]
    local r_func = function_defs[:r][get(T["initial_data"], "r_key", "default")]

    p_key = get(T["initial_data"], "p_key", "default")
    @info "Possible keys for parameteric initial data:"
    @info join(keys(function_defs[:p]), ", ")

    local p_func = function_defs[:p][p_key]

    rho_key = get(T["initial_data"], "rho_key", "default")
    @info "Possible keys for parameteric initial data:"
    @info join(keys(function_defs[:rho]), ", ")
    if rho_key != "default"
        rho_func = function_defs[:rho][rho_key]
    else
        rho_func = nothing
    end

    beta_expr = beta(x, stiffness_params)
    beta_prime_expr = Symbolics.derivative(beta_expr, x)
    beta_second_expr = Symbolics.derivative(beta_prime_expr, x)

    beta_prime = Symbolics.build_function(beta_prime_expr, x; expression=Val{false})
    beta_second = Symbolics.build_function(beta_second_expr, x; expression=Val{false})

    S = Stiffness(beta=(x) -> beta(x,stiffness_params),
                  beta_prime=beta_prime,
                  beta_second=beta_second)

    # Possible bifurcations
    rho_eq = P.M/P.L
    b_eq, bp_eq, bpp_eq = S.beta(rho_eq), S.beta_prime(rho_eq), S.beta_second(rho_eq)

    spont_curv_coef = abs(P.c0 - 2π/P.L)

    can_bifurcate = (bp_eq^2 - 0.5b_eq*bpp_eq > 0) && (spont_curv_coef > 1e-8)

    if spont_curv_coef <= 1e-8
        @warn "The spontaneous curvature term (2π/L - c0) is very small!"
    end

    if can_bifurcate
        f = (2π/P.L - P.c0)
        µ1 = -0.5f^2*bpp_eq
        µk = [µ1; [(f/k)^2*(bp_eq^2/b_eq - 0.5bpp_eq) for k in 2:20]]
        js = 1:length(µk)
        µk_js = collect(zip(μk, js))
        sort!(μk_js, lt=(x,y) -> x[1] < y[1], rev=true)
    else
        µk_js = []
    end

    none = true

    i = 1
    while i <= length(μk_js) && µk_js[i][1] > P.µ
        µc, j = μk_js[i]
        if none
            @info "Birfucations expected for the following values of µ:"
            none = false
        end
        @info "j=$j, µc=$µc"
        i += 1
    end

    if none
        @warn "No birfucations from the circle, expect convergence to trivial state"
    end

    if i < length(µk_js)
        µc, j = μk_js[i]
        @info "Next bifurcation: j=$j, μc = $μc"
    end

    options = zip(map(Symbol, collect(keys(T["options"]))), values(T["options"]))

    # Initial condition
    Xinit = initial_data_from_dict(P, T["initial_data"], (r_func=r_func, p_func=p_func, rho_func=rho_func))

    return P, S, Xinit, solver_params, options
end

function resample(X::Vector{Float64}, new_N::Int64)
    old_N = Int((size(X, 1) - 3)/2)

    if new_N == old_N
        return X
    end

    old_ρ = X[1:old_N]
    old_θ = X[old_N+1:2old_N]

    s = range(0, 2π; length=old_N+1)[1:old_N]

    new_ρ = EvenParam.resample(old_ρ, s, new_N; periodicise=true)
    new_θ = EvenParam.resample(old_θ, s, new_N; append=[round(old_θ[end]/π)*π])

    m = vcat(new_ρ, new_θ, X[2old_N+1:end])
    return m
end

end # module
