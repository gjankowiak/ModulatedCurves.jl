module ModulatedCurves

import Printf: @sprintf, @printf

export Params, Stiffness, IntermediateParams, SolverParams
export compute_intermediate, build_flower, assemble_fd_matrices, init_plot, do_flow

import LinearAlgebra
const LA = LinearAlgebra

import SparseArrays
const SA = SparseArrays

import GLMakie

import JankoUtils: spdiagm_const
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
                    current_step_size::Float64, history::History)

    local energy, residual

    Xnew = zeros(size(X))
    t = current_step_size

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

    @printf "Residual inc ratio: %.5f, δt: %.5f" residual_inc_ratio t

    return Xnew, residual, t
end

function minimizor(P::Params, IP::IntermediateParams, S::Stiffness,
        Xinit::Vector{Float64}, SP::SolverParams=default_SP)
    matrices = assemble_fd_matrices(P, IP)

    # Initialization
    history = History(0, zeros(2*P.N+3))

    X = Base.copy(Xinit)
    n = 1
    step_size = SP.step_size

    residual = compute_residual(P, IP, S, matrices, X)

    energy_i = zeros(SP.max_iter)
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
            Xnew, residual, step_size = adapt_step(P, IP, S, SP, matrices,
                                                   X, δX,
                                                   residual, step_size,
                                                   history)
            X .= Xnew
        else
            # Update
            X += step_size*δX
            residual = compute_residual(P, IP, S, matrices, X)
        end

        # Compute residual norm and energy
        residual_norm = LA.norm(residual)
        energy = compute_energy(P, IP, S, matrices, X)
        energy_i[n] = energy
        residual_norm_i[n] = residual_norm

        converged = (LA.norm(residual - history.residual_prev) < SP.rtol) || (LA.norm(residual) < SP.atol)

        history.energy_prev = energy
        history.residual_prev = residual

        res = Result(X, n, energy_i, residual_norm_i, converged, converged || (n>=SP.max_iter))
        return res
    end
end

function build_flower(P::Params, IP::IntermediateParams, S::Stiffness,
        Xinit::Vector{Float64}, SP::SolverParams=default_SP; include_multipliers::Bool=false)
    matrices = assemble_fd_matrices(P, IP)

    # Initialization
    history = History(0, zeros(2*P.N+3))

    X = Base.copy(Xinit)
    c = X2candidate(P, X)
    n = 1
    step_size = SP.step_size

    residual = compute_residual(P, IP, S, matrices, X)

    energy_i = [compute_energy(P, IP, S, matrices, X)]
    residual_norm_i = [LA.norm(residual)]
    int_θ_i = [sum(c.θ)*IP.Δs]
    t_i = [0.0]

    history.energy_prev = energy_i[1]
    history.residual_prev .= residual

    id_matrix = Matrix{Float64}(LA.I, 2P.N+3, 2P.N+3)
    if !include_multipliers
        for i in (2P.N:2P.N+3)
            id_matrix[i,i] = 0
        end
    end

    # X_prev = Base.copy(X)

    iter_stationary_res = 0

    # Loop until tolerance or max. iterations is met
    function ()
        n += 1

        residual = compute_residual(P, IP, S, matrices, X)
        residual_norm = LA.norm(residual)

        # Assemble
        A = assemble_inner_system(P, IP, S, matrices, X)

        # Solve
        δX = (id_matrix .+ step_size*A)\(-step_size*residual)

        if SP.adapt
            Xnew, residual, step_size = adapt_step(P, IP, S, SP, matrices,
                                                   X, δX,
                                                   residual, step_size,
                                                   history)
            X .= Xnew
        else
            # Update
            X += step_size*δX
            residual = compute_residual(P, IP, S, matrices, X)
        end

        # Compute residual norm and energy
        residual_norm = LA.norm(residual)
        energy = compute_energy(P, IP, S, matrices, X)
        push!(energy_i, energy)
        push!(residual_norm_i, residual_norm)

        push!(t_i, t_i[end] + step_size)
        push!(int_θ_i, sum(c.θ)*IP.Δs)

        if (LA.norm(residual - history.residual_prev)/LA.norm(residual)) < SP.rtol
            @show LA.norm(residual - history.residual_prev)
            iter_stationary_res += 1
        else
            iter_stationary_res = 0
        end

        converged = (iter_stationary_res > 100) || (LA.norm(residual) < SP.atol)
        if converged
            @show iter_stationary_res
        end

        history.energy_prev = energy
        history.residual_prev = residual

        res = Result(X, n, energy_i, residual_norm_i, t_i, int_θ_i, converged, converged || (n>=SP.max_iter))

        return res
    end
end

function compute_energy_split(P::Params, IP::IntermediateParams, S::Stiffness,
                              matrices::FDMatrices, X::Vector{Float64})
    c = X2candidate(P, X)
    ρ_dot = (circshift(c.ρ, -1) - c.ρ)/IP.Δs
    θ_dot = compute_centered_fd_θ(P, matrices, c.θ)
    beta = S.beta(c.ρ)
    return (0.5*P.epsilon^2*IP.Δs*sum(ρ_dot.^2), 0.5IP.Δs*sum(beta.*(θ_dot .- P.c0).^2))
end

function compute_energy(P::Params, IP::IntermediateParams, S::Stiffness,
        matrices::FDMatrices, X::Vector{Float64})
    c = X2candidate(P, X)
    ρ_dot = (circshift(c.ρ, -1) - c.ρ)/IP.Δs
    θ_dot = compute_centered_fd_θ(P, matrices, c.θ)
    beta = S.beta(c.ρ)
    E = 0.5*P.epsilon^2*IP.Δs*sum(ρ_dot.^2) + 0.5IP.Δs*sum(beta.*(θ_dot .- P.c0).^2)
    if P.potential_range > 0
        (w, _, _) = compute_potential(P, X)
        E += IP.Δs*sum(w)
    end
    return E
end

function assemble_inner_system(P::Params, IP::IntermediateParams, S::Stiffness,
        matrices::FDMatrices, X::Vector{Float64})
    N = P.N
    Δs = IP.Δs

    # Easier access to ρ, θ, λi
    c = X2candidate(P, X)

    # Compute beta and derivatves
    beta_prime_ρ = S.beta_prime(c.ρ)
    beta_second_ρ = S.beta_second(c.ρ)
    beta_phalf = S.beta(matrices.M_phalf*c.ρ)
    beta_a1half = S.beta(matrices.M_mhalf*c.ρ)
    beta_prime_phalf = S.beta_prime(matrices.M_phalf*c.ρ)
    beta_prime_mhalf = S.beta_prime(matrices.M_mhalf*c.ρ)

    # Compute upstream and downstream finite differences of θ
    _, θ_prime_up, θ_prime_down = compute_fd_θ(P, matrices, c.θ)
    θ_prime_centered = compute_centered_fd_θ(P, matrices, c.θ)

    A_E1_ρ = P.epsilon^2 * matrices.D2 - 0.5*beta_second_ρ.*(θ_prime_centered .- P.c0).^2 .*Matrix{Float64}(LA.I, N, N)
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
        matrices::FDMatrices, X::Vector{Float64})
    N = P.N
    Δs = IP.Δs

    # Easier access to ρ, θ, λi
    c = X2candidate(P, X)

    # Compute beta and derivatves
    beta_prime_ρ = S.beta_prime(c.ρ)
    beta_phalf = S.beta(matrices.M_phalf*c.ρ)
    beta_a1half = S.beta(matrices.M_mhalf*c.ρ)

    # Compute upstream and downstream finite differences of θ
    _, θ_prime_up, θ_prime_down = compute_fd_θ(P, matrices, c.θ)

    # Compute residuals for ρ
    b_E1 = P.epsilon^2 * (matrices.D2 * c.ρ) - 0.5*beta_prime_ρ.*(compute_centered_fd_θ(P, matrices, c.θ) .- P.c0).^2 + c.λM*ones(N)
    if P.potential_range > 0
        (_, w_prime, _) = compute_potential(P, X)
        b_E1 -= w_prime
    end

    # Compute residuals for θ
    b_E2 = (beta_phalf.*(θ_prime_down .- P.c0) - beta_a1half.*(θ_prime_up .- P.c0))/Δs - c.λx*sin.(c.θ) + c.λy*cos.(c.θ)

    # Assemble with constraint residuals
    res = [b_E1; b_E2; Δs*sum(cos.(c.θ)); Δs*sum(sin.(c.θ)); Δs*sum(c.ρ) - P.M]
    return -res
end

function compute_fd_θ(P::Params, matrices::FDMatrices, θ::Union{Vector{Float64}, SubA})
    fd = matrices.D1*θ + matrices.D1_rhs
    return (fd, view(fd, 1:P.N), view(fd, 2:P.N+1))
end

function compute_centered_fd_θ(P::Params, matrices::FDMatrices, θ::Union{Vector{Float64}, SubA})
    return matrices.D1c*θ + matrices.D1c_rhs
end

function assemble_fd_matrices(P::Params, IP::IntermediateParams)
    N = P.N
    Δs = IP.Δs

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
                   [2π/Δs; zeros(N-1); 2π/Δs],

                   # Matrix for centered finite differences
                   (SA.spdiagm(1-N =>  ones(1),
                               -1  => -ones(N-1),
                                1  =>  ones(N-1),
                               N-1 => -ones(1)))*0.5/Δs,

                   # Affine term for centered finite differences
                   [π/Δs; zeros(N-2); π/Δs],

                   # Matrix for 2nd order finite difference
                   spdiagm_const([1.0, -2.0, 1.0], [-1, 0, 1], N)/Δs^2
                  )

    return M
end

function initial_data_smooth(P::Params; sides::Int=1, smoothing::Float64, reverse_phase::Bool=false, only_rho::Bool=false)
    if P.N % sides != 0
        error("N must be dividible by the number of sides")
    end

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

    if reverse_phase
        rho = sin.(range(0, π, length=N_straight))
        rho /= (P.M/sides/(N_straight)) / sum(rho)
        rhos = repeat([rho; zeros(k-N_straight)], sides)
    else
        rhos = repeat([zeros(N_straight); P.M/(2π*smoothing)*ones(k-N_straight)], sides)
    end


    # FIXME
    k = 6
    thetas = circshift(thetas, -k)
    thetas[end-(k-1):end] .+= 2π

    return [P.M/P.L*ones(P.N); thetas; 0; 0; 0]
end

function initial_data(P::Params, a::Real, b::Real; pulse::Int=1, pulse_amplitude::Real=2e-2, poly::Bool=false, reverse_phase::Bool=false, only_rho::Bool=false)
    throw("not converted to free θ(0)")
    N = P.N

    # Construct the ellipsis and reparameterize it
    if poly && (pulse > 2)
        x = [real(exp(2π*im*k/pulse)) for k in 0:(pulse-1)]
        y = [imag(exp(2π*im*k/pulse)) for k in 0:(pulse-1)]
        xy = [x y]
    else
        t = collect(range(-π/2, 3π/2, length=10N+1)[1:10N])
        xy = [a*cos.(t) b*sin.(t)]
    end
    xy_even = EvenParam.reparam(xy; closed=true, new_N = N)

    # Recenter
    xy_even = xy_even .- [xy_even[1,1] xy_even[1,2]]

    # Rotate so that the first segment is parallel to the x axis
    xy_even_c = xy_even[:,1] + xy_even[:,2]*im
    alpha = angle(xy_even_c[2])
    xy_even_c = xy_even_c.*exp.(-im*alpha)

    # Compute tangential angles:w
    xy_even = [real.(xy_even_c) imag.(xy_even_c)]
    xy_diff = circshift(xy_even, -1) - xy_even
    xy_angles = angle.(xy_diff[:,1]+xy_diff[:,2]*im)

    thetas = zeros(N-1)
    thetas[1] = xy_angles[2]
    for i in 2:N-1
        thetas[i] = thetas[i-1] + rem(xy_angles[i+1] - thetas[i-1], Float64(π), RoundNearest)
    end

    # FIXME: shift the initial condition
    t = collect(range(0, 2π, length=N+1)[1:N])

    if pulse > 0
        # To check, 1 was P.beta_a1
        thetas += 1.0/P.mode_j*pulse_amplitude*sin.(pulse*t[2:N])
    end

    rhos = P.M/2π*ones(N)

    if pulse > 0
        if reverse_phase
            f = -1
        else
            f = 1
        end
        rhos -= pulse_amplitude*f*cos.(pulse*t)
    end

    if only_rho
        thetas = collect(range(0, 2π, length=P.N+1))[2:end-1]
    end

    return [rhos; thetas; 0; 0; 0]
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

function do_flow(P::Params, S::Stiffness, Xinit::Vector{Float64}, solver_params::SolverParams;
        do_plot::Bool=false, include_multipliers::Bool=false, record_movie::Bool=false)

    IP = compute_intermediate(P, S)

    ρ_equi = P.M / P.L
    k_equi = P.L / 2π
    energy_circle = P.L * S.beta(ρ_equi) * (k_equi - P.c0)^2 / 2

    Xx = copy(Xinit)

    if do_plot
        fig, update_plot = init_plot(P, IP, S, Xx)
    end

    flower = build_flower(P, IP, S, Xx, solver_params; include_multipliers=include_multipliers)

    local res

    done = [false]

    iter = 1:(solver_params.max_iter-1)

    function iter_flower(_i)
        res = flower()

        Xx .= res.sol
        n = res.iter

        if n % 1 == 0
            println()
            print(n)

            print(", energy/rel: ")
            print(res.energy_i[n], "/", res.energy_i[n] - energy_circle)
            print(", residual norm: ")
            print(res.residual_norm_i[n])
            print(", min/max ρ: ")
            @show extrema(view(Xx, 1:P.N))
            if do_plot
                update_plot(res.sol, (@sprintf "%d, energy/rel: %.4f/%.4f" n res.energy_i[n]  (res.energy_i[n] - energy_circle)), res)
            end
        else
            print(".")
        end

        if res.finished || res.residual_norm_i[n] > 1e10
            print("Finished, converged: ")
            println(res.converged)
            @show res.residual_norm_i[end]
            @show extrema(res.int_θ_i)
            if do_plot
                update_plot(res.sol, (@sprintf "%d, energy/rel: %.4f/%.4f" n res.energy_i[n]  (res.energy_i[n] - energy_circle)), res)
            end
            done[1] = true
        end
    end

    if record_movie
        GLMakie.record(iter_flower, fig, "output.mp4", iter, framerate=10)
    else
        while !done[1]
            iter_flower(1)
        end
    end
end

end # module
