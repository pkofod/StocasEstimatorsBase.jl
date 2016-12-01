module StocasEstimatorsBase

using Optim, Reexport
@reexport using StatsBase
@reexport using Stocas
const AbstractUtility = Stocas.AbstractUtility
const AbstractState = Stocas.AbstractState
export EstimationMethod, EstimationResults, AbstractUtility,
        AbstractTrace, ConvergenceInfo,
        AbstractState, hessian, gradient, loglikelihood,
        tstat, fit

import StatsBase: loglikelihood, fit, nobs
import Base: gradient


abstract type EstimationMethod end
fit(E::EstimationMethod) = warn("fit is not implemented for $E")

abstract type ConvergenceInfo end

# --- AbstractTrace ---

abstract type AbstractTrace end
abstract type Trace end

type EstimationResults{T<:EstimationMethod, Tf<:Real, Tobj}
    E::T
    loglikelihood::Tf
    ∇loglikelihood::Vector{Tf}
    ∇²loglikelihood::Matrix{Tf}
    tdf::Tobj
    coef::Vector{Tf}
    conv::ConvergenceInfo
    trace
    nobs::Int
    meta
end

"""
    convinfo(res::EstimationResults)
Return convergence information from `res`.
"""
convinfo(res::EstimationResults) = res.conv
"""
    outer_iterations(res::EstimationResults)
Return the number of outer iterations from a nested estimation procedure.
"""
outer_iterations(res::EstimationResults) = outer_iterations(convinfo(res))
"""
    inner_iterations(res::EstimationResults)
    inner_iterations(res::ConvergenceInfo)
Return the number of inner iterations from a nested estimation procedure.
"""
inner_iterations(res::EstimationResults) = inner_iterations(convinfo(res))

"""
    newton_iterations(res::EstimationResults)
Return the number of newton iterations used to solve the model if the estimation
method relies on solving the model.
"""
newton_iterations(res::EstimationResults) = newton_iterations(convinfo(res))

"""
    contraction_iterations(res::EstimationResults)
Return the number of contraction iterations used to solve the model if the estimation
method relies on solving the model.
"""
contraction_iterations(res::EstimationResults) = contraction_iterations(convinfo(res))

# loglikelihood
nobs(res::EstimationResults) = res.nobs
loglikelihood(res::EstimationResults) = res.loglikelihood
loglikelihood(res::EstimationResults, x) = -res.tdf.f(x)*nobs(res)
gradient(res::EstimationResults) = res.∇loglikelihood
function gradient(res::EstimationResults, x)
    g = similar(x)
    res.tdf.g!(x, g)
    -g*nobs(res)
end
hessian(res::EstimationResults) = res.∇²loglikelihood
function hessian(res::EstimationResults, x)
    n = length(x)
    h = zeros(n, n)
    res.tdf.h!(x, h)
    -h*nobs(res)
end

# coefficients and inference
coef(res::EstimationResults) = res.coef
coef(res::EstimationResults, i) = coef(res)[i]
vcov(res::EstimationResults) = inv(-hessian(res))
stderr(res::EstimationResults) = sqrt.(diag(vcov(res)))
stderr(res::EstimationResults, i) = sqrt.(diag(vcov(res)))[i]
tstat(res, i, x) = (coef(res, i)-x)/(stderr(res, i))
tstat(res, i) = tstat(res, i, 0)
tstat(res, x::Vector) = [tstat(res, i, x[i]) for i = 1:length(x)]
tstats(res) = tstat(res, zeros(coef(res)))
end # module
