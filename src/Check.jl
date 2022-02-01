function check_differential(P::Params, S::Stiffness, X::Vector{Float64})
    IP = compute_intermediate(P, S)
    matrices = assemble_fd_matrices(P, IP)

    @show matrices.D1

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
    @show a, b

    fig = GLMakie.Figure()
    ax1 = GLMakie.Axis(fig[1,1], xlabel="step size", ylabel="error", xscale=GLMakie.log10, yscale=GLMakie.log10)

    GLMakie.scatterlines!(ax1, hs, errors, color="blue")
    GLMakie.scatterlines!(ax1, hs, 10.0^b*hs.^a, color="red")

    GLMakie.display(fig)
end
