function check_differential(P::Params, S::Stiffness, X::Vector{Float64})
    IP = compute_intermediate(P, S)
    matrices = assemble_fd_matrices(P, IP; winding_number=compute_winding_number(P, X))

    dir = rand(2P.N + 3)

    res = compute_residual(P, IP, S, matrices, X)
    A = assemble_inner_system(P, IP, S, matrices, X)

    exponents = [ -4, -5, -6, -7]
    hs = 10.0 .^exponents

    n = length(exponents)
    errors = zeros(n)

    for (i,h) in enumerate(hs)
        res_true = compute_residual(P, IP, S, matrices, X + h*dir)
        res_approx = res + A*(h*dir)

        errors[i] = LA.norm(res_true - res_approx)
    end

    (a, b) = linreg(log10.(hs), log10.(errors))

    if a > 1.5
        @info "Slope is above 1.5, looks OK!"
    elseif a > 1.1
        @warn "Slope is below 1.5 but above 1.1, strange."
    else
        @error "Slope is 1 or below, something's wrong!"
    end

    fig = GLMakie.Figure()
    ax1 = GLMakie.Axis(fig[1,1], xlabel="step size", ylabel="error", xscale=GLMakie.log10, yscale=GLMakie.log10)

    GLMakie.scatterlines!(ax1, hs, errors, color="blue")
    GLMakie.scatterlines!(ax1, hs, 10.0^b*hs.^a, color="red")

    GLMakie.display(fig)
end

function check_differential_full(P::Params, S::Stiffness, X::Vector{Float64}, Xprev::Vector{Float64}, time_step::Float64)
    IP = compute_intermediate(P, S)
    matrices = assemble_fd_matrices(P, IP; winding_number=compute_winding_number(P, X))

    dir = rand(2P.N + 3)

    id_matrix = Matrix{Float64}(LA.I, 2P.N+3, 2P.N+3)
    for i in (2P.N:2P.N+3)
        id_matrix[i,i] = 0
    end

    res = compute_residual(P, IP, S, matrices, X, Xprev, time_step)
    A = assemble_inner_system(P, IP, S, matrices, X)

    exponents = [ -4, -5, -6, -7]
    hs = 10.0 .^exponents

    n = length(exponents)
    errors = zeros(n)

    for (i,h) in enumerate(hs)
        res_true = compute_residual(P, IP, S, matrices, X + h*dir, Xprev, time_step)
        res_approx = res + A*(h*dir)

        errors[i] = LA.norm(res_true - res_approx)
    end

    (a, b) = linreg(log10.(hs), log10.(errors))

    if a > 1.5
        @info "Slope is above 1.5, looks OK!"
    elseif a > 1.1
        @warn "Slope is below 1.5 but above 1.1, strange."
    else
        @error "Slope is 1 or below, something's wrong!"
    end

    fig = GLMakie.Figure()
    ax1 = GLMakie.Axis(fig[1,1], xlabel="step size", ylabel="error", xscale=GLMakie.log10, yscale=GLMakie.log10)

    GLMakie.scatterlines!(ax1, hs, errors, color="blue")
    GLMakie.scatterlines!(ax1, hs, 10.0^b*hs.^a, color="red")

    GLMakie.display(fig)
end
