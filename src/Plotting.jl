# import GLMakie
# M = GLMakie
import CairoMakie
M = CairoMakie
M.activate!()

function isconstant(v::Vector{Float64}, tol::Float64=1e-10)
    m, M = extrema(v)
    @show extrema(v)
    return M - m < tol
end

function init_plot(P::Params, IP::IntermediateParams, S::Stiffness, X::Vector{Float64}; label::String="", plain::Bool=false)

    w = compute_winding_number(P, X)

    matrices = assemble_fd_matrices(P, IP; winding_number=w)

    fig = M.Figure(resolution = (2000, 2000))

    if plain
        axes = [M.Axis(fig[1, 1], aspect=M.AxisAspect(1))]
    else
        axes = [
                M.Axis(fig[1, 1], aspect=M.AxisAspect(1)) M.Axis(fig[1, 2], title=M.L"\rho") M.Axis(fig[1, 3], title="← Energy | Residual →", xscale=log10);
                M.Axis(fig[2, 1], title=M.L"\leftarrow \dot{\theta} \qquad\qquad \ddot{\theta} \rightarrow") M.Axis(fig[2, 2], title=M.L"\beta(\rho)") M.Axis(fig[2,3], yscale=log10, title=M.L"\leftarrow ||\theta - \phi||_2  \qquad | \qquad \min \dot{\theta} \rightarrow", xscale=log10)
                # M.Axis(fig[2, 1], title=M.L"\leftarrow \dot{\theta} - \frac{2\pi}{L} \qquad\qquad \ddot{\theta} \rightarrow") M.Axis(fig[2, 2], title=M.L"\beta(\rho)") M.Axis(fig[2,3], title=M.L"\leftarrow \mathrm{sign}\dot{E}_ρ \qquad\qquad \mathrm{sign}\dot{E}_θ \rightarrow", xscale=log10)
        ]

        dual_axes = [M.Axis(fig[1, 3], yticklabelcolor = :red, yaxisposition = :right, xscale=log10, yscale=log10),
                     M.Axis(fig[2, 1], yticklabelcolor = :red, yaxisposition = :right),
                     M.Axis(fig[2, 3], yticklabelcolor = :red, yaxisposition = :right, xscale=log10)]

        x_logscale_axes = [axes[1, 3]; axes[2, 3]; dual_axes[1]; dual_axes[3]]
        y_logscale_axes = [dual_axes[1]]

        M.hidexdecorations!.(dual_axes)
        M.hidespines!.(dual_axes)
    end
    # M.hidexdecorations!(dual_axes[1,1])
    # M.hidespines!(dual_axes[1,1])
    M.hidedecorations!(axes[1, 1])
    M.hidespines!(axes[1, 1])

    w_plus = max(1,w)
    M.limits!(axes[1,1], -2, 2, -2, 2)
    # M.limits!(axes[1,1], -1.5/w_plus, 1.5/w_plus, -1.5/w_plus, 1.5/w_plus)

    if !plain
        M.limits!(axes[1,3], 0.5, 1.5, 0.5, 1.5)
        M.limits!(axes[2,3], 0.5, 1.5, 0.5, 1.5)
        M.limits!(dual_axes[1], 0.5, 1.5, 0.5, 1.5)
        M.limits!(dual_axes[3], 0.5, 1.5, -1.5, 1.5)
    end

    N = P.N
    Δs = IP.Δs

    t = collect(range(0, 2π, length=N+1))[1:N]

    xy = zeros(N+1, 2)

    function compute_c(_X)
        return X2candidate(P, _X; copy=false)
    end

    function compute_xy(c_node)
        for i in 2:N+1
            xy[i,:] = xy[i-1,:] + Δs*[cos(c_node.θ[i-1]); sin(c_node.θ[i-1])]
        end

        bc = sum(xy, dims=1)/N
        xy = xy .- bc

        return xy
    end

    X_node = M.Observable(X)
    c_node = M.@lift X2candidate(P, $X_node; copy=false)
    xy_node = M.@lift compute_xy($c_node)
    θ_dot_node = M.@lift compute_centered_fd_θ(P, matrices, $c_node.θ)
    θ_dot_node_shifted = M.@lift $θ_dot_node .- 2π/P.L

    ts_node = M.Observable([0.0])
    energies_node = M.Observable([0.0])
    energies_ρ_node = M.Observable([0.0])
    energies_θ_node = M.Observable([0.0])
    d_energies_ρ_node = M.Observable([0.0])
    d_energies_θ_node = M.Observable([0.0])
    int_θ_node = M.Observable([0.0])
    residual_norm_node = M.Observable([0.0])

    diff_to_circle = M.Observable(Float64[])
    min_d_θ = M.Observable(Float64[])

    ρ_equi = P.M / P.L
    k_equi = w * P.L / 2π
    energy_circle = P.L * S.beta(ρ_equi) * (k_equi - P.c0)^2 / 2

    function compute_θ_dotdot(θ)
      r = matrices.D2*θ
      r[1] -= 2π/(IP.Δs^2)
      r[end] += 2π/(IP.Δs^2)
      return r
    end

    int_θ_0 = Inf

    θ_dotdot_node = M.@lift compute_θ_dotdot($c_node.θ)

    function update(X::Vector{Float64}, title::String, result::Result)
        X_node[] = X
        c = X2candidate(P, X; copy=false)

        if !plain
            axes[1,1].title = title
            if int_θ_0 == Inf
                int_θ_0 = result.int_θ_i[1]
            end

            ts_node[] = result.t_i .+ 1

            # extrema_int_θ = map(x -> round(x/π; digits=3), extrema(result.int_θ_i))
            # cur_int_θ = round(result.int_θ_i[end]/π; digits=3)
            # axes[2,3].title = "$(extrema_int_θ[1])π ≤ ∫θ = $(cur_int_θ)π ≤ $(extrema_int_θ[2])π"

            energies_node[] = result.energy_i
            energies_ρ_node[] = result.energy_ρ_i
            energies_θ_node[] = result.energy_θ_i

            diff_Eρ = diff(result.energy_ρ_i)
            diff_Eθ = diff(result.energy_θ_i)

            d_energies_ρ_node[] = sign.(diff_Eρ)
            d_energies_θ_node[] = sign.(diff_Eθ)

            diff_to_circle[] = vcat(diff_to_circle[], sqrt.(IP.Δs*sum(abs2, (c.θ .- c.θ[1] .- 2π/P.L*t))))
            a = minimum(θ_dot_node[])
            # min_d_θ[] = vcat(min_d_θ[], min(sign(a), a))
            min_d_θ[] = vcat(min_d_θ[], a)

            int_θ_node[] = result.int_θ_i
            residual_norm_node[] = result.residual_norm_i

            if length(result.int_θ_i) > 2

                for (i, a) in enumerate(axes[2:end])
                    # if i!=5
                        M.autolimits!(a)
                    # else
                    #     M.xlims!(a, (1, length(result.t_i)))
                    # end
                end
                for (i, a) in enumerate(dual_axes)
                    # if i != 3
                        M.autolimits!(a)
                    # else
                    #     M.xlims!(a, (1, length(result.t_i)))
                    # end
                end

            end
        end
    end

    zipped_ρ_θ_dot_node = M.lift(c -> [[(ρ, θ_dot_node[][i]) for (i,ρ) in enumerate(c.ρ)]; (c.ρ[1], θ_dot_node[][1])], c_node)
    colors = M.lift(z -> map(t -> t[1] > 0 ? (t[2] >= 0.0 ? "blue" : "lightblue") : (t[2] >= 0.0 ? "red" : "pink"), z), zipped_ρ_θ_dot_node)

    # M.lines!(axes[1,1], cos.(t), sin.(t), lw=0.5, color="gray")
    M.scatterlines!(axes[1,1], M.lift(x -> x[:,1], xy_node), M.lift(x -> x[:,2], xy_node),
                    markersize=M.lift(x -> 100*abs.([x.ρ; x.ρ[1]]), c_node),
                    markercolor=colors)
    M.lines!(axes[1,1], M.lift(x -> x[:,1], xy_node), M.lift(x -> x[:,2], xy_node), color="white", linewidth=0.3)
                    # markercolor=M.lift(x -> map(v -> v>0 ? "blue" : "red", [x.ρ; x.ρ[1]]), )
    # M.scatter!(axes[1,1], M.lift(x -> [x[1,1]], xy_node), M.lift(x -> [x[1,2]], xy_node))

    if !plain
        M.lines!(axes[1,2], t, M.lift(x -> x.ρ, c_node))
        M.hlines!(axes[1,2], ρ_equi, color="gray")

        M.lines!(axes[1,3], ts_node, energies_node, color="blue")
        M.lines!(axes[1,3], ts_node, energies_ρ_node, color="blue", linestyle=:dot)
        M.lines!(axes[1,3], ts_node, energies_θ_node, color="blue", linestyle=:dash)
        M.hlines!(axes[1,3], energy_circle, color="gray")

        M.lines!(dual_axes[1,1], ts_node, residual_norm_node, color="red")

        M.scatterlines!(axes[2,1], t, θ_dot_node_shifted, markersize=3, color="blue")
        M.scatterlines!(dual_axes[2], t, θ_dotdot_node, markersize=3, color="red")

        M.lines!(axes[2,2], t, M.lift(x -> S.beta.(x.ρ), c_node))
        M.hlines!(axes[2,2], [0.0, S.beta(ρ_equi)], color="gray")

        M.lines!(axes[2,3], diff_to_circle, color="blue")
        M.lines!(dual_axes[3], min_d_θ, color="red")
    end

    # Evolution of the kinetic/potential energy
    # M.scatterlines!(axes[2,3], d_energies_ρ_node, color="blue")
    # M.scatterlines!(dual_axes[3], d_energies_θ_node, color="red")

    # M.lines!(axes[2,3], ts_node, int_θ_node)
    # M.hlines!(axes[2,3], 0.0, color="gray")
    # axes[2,3].yticks = M.MultiplesTicks(4, pi, "π")

    M.display(fig)

    return fig, update

end
