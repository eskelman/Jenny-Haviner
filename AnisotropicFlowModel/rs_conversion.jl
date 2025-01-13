#= Accept a DataFrame containing k, ω, dU/dy and Re_tau and return the reynolds stresses,this way both the CFD and DNS data can be fed in, so long as it has these in it =#

module RS
using DataFrames
using CSV
#using StatsBase
using Flux

function calculation(inputs, prediction, λ)
    # inputs = [y_plus,k,ω] (in that order)
    # predicition = [Cβ1, Cβ2] (once normalised)

    κ = 2.5
    Sij = λ .* 0.5
    Sji = λ .* 0.5
    S = (2 .* Sij .* Sij) .^ 0.5
    Ωij = 0.5 .* λ
    Ωji = -0.5 .* λ
    Ω = (2 .* Ωij .* Ωij) .^ 0.5
    S_inv = S ./ inputs[3, :]
    Ω_inv = Ω ./ inputs[3, :]
    M = max(S_inv, Ω_inv)
    Cμ = min.(1, 1 ./ (1 .+ ((M .^ 2) .^ -1)))

    iso = (2 / 3) .* inputs[2, :]
    reals = Cμ ./ (max.(inputs[3, :], κ .* S)) .^ 2

    reals = reals .* λ .^ 2
    var_uu = iso .+ reals .* ((1 / 12) .* prediction[1, :] .+ 0.5 .* prediction[2, :]) .* inputs[2, :]
    var_vv = iso .+ reals .* ((1 / 12) .* prediction[1, :] .- 0.5 .* prediction[2, :]) .* inputs[2, :]
    var_ww = iso .+ reals .* (-(1 / 6) .* prediction[1, :]) .* inputs[2, :]

    return Array(hcat(var_uu, var_vv, var_ww)')
end
end
