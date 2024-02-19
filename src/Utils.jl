using Interpolations

"""
    reparam(X, dst, closed)

Given a curve in R^d discretized at N points, represented by the N-by-d matrix X,
outputs in the M-by-d matrix dst the result of resampling the curve with a fixed
arclength step with M points, where M = N if new_N is not specified, otherwise M = new_N

The starting point is always preserved. If closed is false, the ending point
is also preserved. If true, then the corresponding curve is assumed to be closed,
and computations are done assuming periodicity of the coordinates. The only preserved
point is the first one, since it conceptually also corresponds to the ending point.
"""
function reparam(X::Array{Float64,2}, dst::Array{Float64,2}; closed::Bool=true, new_N::Int64=0)
  N, d = size(X)

  eff_N = new_N == 0 ? N : new_N

  if closed
    X_e = X[[1:N; 1], :]
  else
    X_e = X
  end

  dX = [zeros(1, d); diff(X_e; dims=1)]

  l = vec(cumsum(sqrt.(sum(abs2, dX; dims=2)); dims=1))
  l_max = l[end]

  if closed
    new_nodes = collect(range(0; stop=l_max, length=eff_N + 1)[1:eff_N])
  else
    new_nodes = collect(range(0; stop=l_max, length=eff_N))
  end

  for i in 1:d
    dst[:, i] = interpolate((l,), X_e[:, i], Gridded(Linear()))(new_nodes)
  end
end # reparam

function reparam(X::Array{Float64,2}; closed::Bool=true, new_N::Int64=0)
  N, d = size(X)
  if new_N == 0
    dst = zeros(N, d)
  else
    dst = zeros(new_N, d)
  end
  reparam(X, dst; closed=closed, new_N=new_N)
  return dst
end

function reparam!(X::Array{Float64,2}; closed::Bool=true)
  reparam(X, X; closed=closed)
end

"""
    Given X which same sampled at nodes old_nodes, return the linear interpolation evaluated at new_nodes.
    If periodicise if true, X[1] is appended to X and one node is appended to old_nodes, assuming a constent step
"""
function resample(X::Array{T}, old_nodes::AbstractVector{T}, new_nodes::AbstractVector{T}; periodicise::Bool=false, append::Union{Nothing,Array{T}}=nothing) where {T<:Real}
  @assert !periodicise || isnothing(append)

  new_N = length(new_nodes)
  nd = ndims(X)

  (X_ext, old_nodes_ext) = if periodicise
    if nd == 1
      (vcat(X, X[1]), vcat(old_nodes, 2old_nodes[end] - old_nodes[end-1]))
    else
      (vcat(X, X[1, :]'), vcat(old_nodes, 2old_nodes[end] - old_nodes[end-1]))
    end
  elseif !isnothing(append)
    if nd == 1
      (vcat(X, append), vcat(old_nodes, 2old_nodes[end] - old_nodes[end-1]))
    else
      (vcat(X, vec(append)'), vcat(old_nodes, 2old_nodes[end] - old_nodes[end-1]))
    end
  else
    (X, old_nodes)
  end

  if nd == 1
    dst = vec(interpolate((old_nodes_ext,), X_ext, Gridded(Linear()))(new_nodes))
  elseif nd == 2
    d = size(X, 2)
    dst = zeros(new_N, d)

    for i in 1:d
      dst[:, i] = interpolate((old_nodes_ext,), X_ext[:, i], Gridded(Linear()))(new_nodes)
    end
  else
    throw(ArgumentError("X must have at most 2 dimensions"))
  end

  return dst
end

"""
    Given X which same sampled at the nodes corresponding to old_range, return the linear interpolation evaluated
    the range with the same bounds with length new_N.
    If periodicise if true, X[1] is appended to X and one node is appended to old_range, assuming a constent step
"""
function resample(X::Array{T}, old_range::AbstractVector{T}, new_N::Int64; periodicise::Bool=false, append::Union{Nothing,Array{T}}=nothing) where {T<:Real}
  @assert !periodicise || isnothing(append)
  new_range = if periodicise || !isnothing(append)
    s = if isa(old_range, StepRangeLen)
      step(old_range)
    else
      old_range[2] - old_range[1]
    end
    range(old_range[1], old_range[end] + s; length=new_N + 1)[1:new_N]
  else
    range(old_range[1], old_range[end]; length=new_N)
  end

  return resample(X, old_range, new_range; periodicise=periodicise, append=append)
end

macro cross(v1, v2)
  return esc(:($v1[1] * $v2[2] - $v1[2] * $v2[1]))
end

"""
    compute_ls_intersect(A1, A2, B1, B2; exclude_ends=false, tol=1e-11)

    Check if two two line segments A1A2 and B1B2 intersect.
    If exclude_ends is true, end points are not considered part of the segments.

    Segments are considered parallel if the absolute value of their cross-product is less than tol.
    Parallel or almost parallel segments are considering not intersecting.

    Returns a tuple ({C1; C2], endpoint) if the segments intersects, where [C1; C2] is the intersection point
    and endpoint is true if the intersection lies on an endpoint of the input segments.
    Otherwise, returns (missing, false).
"""
function compute_ls_intersect(a1::Vector{Float64}, a2::Vector{Float64},
  b1::Vector{Float64}, b2::Vector{Float64};
  exclusive::Bool=false, tol::Float64=1e-11)::Tuple{Union{Vector{Float64},Missing},Bool}
  if a1 == b1
    if exclusive
      return (missing, false)
    else
      return (a1, true)
    end
  elseif a1 == b2
    if exclusive
      return (missing, false)
    else
      return (a1, true)
    end
  elseif a2 == b1
    if exclusive
      return (missing, false)
    else
      return (a1, true)
    end
  elseif a2 == b2
    if exclusive
      return (missing, false)
    else
      return (a1, true)
    end
  end

  c = @cross(a2 - a1, b2 - b1)
  if abs(c) < tol
    # segments are // and we don't deal with overlap
    return (missing, false)
  end

  t = @cross(b1 - a1, b2 - b1) / c
  u = @cross(b1 - a1, a2 - a1) / c

  if (0 <= t <= 1) && (0 <= u <= 1)
    return ((a1 + t * (a2 - a1)), false)
  else
    return (missing, false)
  end
end

"""
    spdiagm_const(coeffs, diag_idx, N)

Constructs a sparse N-by-N matrix, where the diagonal number diag_idx[i] is filled with value coeffs[i].
The diagonal is wrapped to extend over the diagonal -sign(diag_idx[i])*(N-abs(diag_idx[i])).
This is useful for the approximation of differential operators on functions defined on the torus.
"""
function spdiagm_const(coeffs::Array{Float64,1}, diag_idx::Array{Int64,1}, N::Int64)
  if length(coeffs) != length(diag_idx)
    throw("the number of diagonals does not match the number of indices")
  end
  pairs = ()
  for n in 1:length(coeffs)
    i, c = diag_idx[n], coeffs[n]
    conjug = -sign(i) * (N - abs(i))
    if i == 0
      pairs = (pairs..., i => c * ones(N))
      continue
    end

    pairs = (pairs..., i => c * ones(N - abs(i)), conjug => c * ones(N - abs(conjug)))
  end
  return SA.spdiagm(pairs...)
end
