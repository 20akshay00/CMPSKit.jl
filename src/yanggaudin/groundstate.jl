using Printf

# groundstate with YangGaudinCMPS
function groundstate(Ĥ::LocalHamiltonian, Ψ₀::YangGaudinCMPS; kwargs...)
    return groundstate_unconstrained(Ĥ, Ψ₀; kwargs...)
end

function groundstate_unconstrained(Ĥ::LocalHamiltonian, Ψ₀::YangGaudinCMPS;
                                   gradtol=1e-7,
                                   verbosity=2,
                                   optalg=LBFGS(; gradtol=gradtol, verbosity=verbosity - 2),
                                   eigalg=defaulteigalg(Ψ₀),
                                   linalg=defaultlinalg(Ψ₀),
                                   (finalize!)=OptimKit._finalize!,
                                   kwargs...)
    δ = 1
    function retract(x, d, α)
        ΨL, = x
        QL = ΨL.Q
        RLs = ΨL.Rs
        KL = copy(QL)
        for R in RLs
            mul!(KL, R', R, +1, 1)
        end

        dRs = d
        RdR = zero(QL)
        for (R, dR) in zip(RLs, dRs)
            mul!(RdR, R', dR, true, true)
        end
        RdR *= 2

        RLs = RLs .+ α .* dRs
        KL = KL - (α / 2) * (RdR - RdR')
        QL = KL
        for R in RLs
            mul!(QL, R', R, -1, 1)
        end

        ΨL = YangGaudinCMPS(QL, RLs; gauge=:l)
        ρR, λ, info_ρR = rightenv(ΨL, ρR; eigalg=eigalg, linalg=linalg, kwargs...)
        rmul!(ρR, 1 / tr(ρR[]))
        HL, E, e, hL, info_HL = leftenv(Ĥ, (ΨL, ρL, ρR); eigalg=eigalg, linalg=linalg,
                                        kwargs...)

        if info_ρR.converged == 0 || info_HL.converged == 0
            @warn "step $α : not converged, e = $E"
            @show info_ρR
            @show info_HL
        end

        return (ΨL, ρR, HL, E, e, hL), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        return 2 * real(sum(dot.(d1, d2)))
    end

    function precondition(x, d)
        ΨL, ρR, = x
        dRs = d
        return dRs .* Ref(posreginv(ρR[0], δ))
    end

    function fg(x)
        (ΨL, ρR, HL, E, e, hL) = x

        gradQ, gradRs = gradient(Ĥ, (ΨL, ρL, ρR), HL, zero(HL); kwargs...)

        Rs = ΨL.Rs

        dRs = .-(Rs) .* 2 .* Ref(gradQ) .+ gradRs

        return E, dRs
    end

    scale!(d, α) = rmul!.(d, α)
    add!(d1, d2, α) = axpy!.(α, d2, d1)

    function _finalize!(x, E, d, numiter)
        normgrad2 = inner(x, d, d)
        δ = max(1e-12, 1e-3 * normgrad2)
        normgrad = sqrt(normgrad2)
        verbosity > 1 &&
            @info @sprintf("YangGaudinCMPS ground state: iter %4d: e = %.12f, ‖∇e‖ = %.4e",
                           numiter, E, normgrad)
        return finalize!(x, E, d, numiter)
    end

    ΨL, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, info_ρR = rightenv(ΨL; kwargs...)
    ρL = one(ρR)
    rmul!(ρR, 1 / tr(ρR[]))
    HL, E, e, hL, info_HL = leftenv(Ĥ, (ΨL, ρL, ρR); kwargs...)
    x = (ΨL, ρR, HL, E, e, hL)

    if info_ρR.converged == 0 || info_HL.converged == 0
        @warn "initial point not converged, energy = $E"
        @show info_ρR
        @show info_HL
    end

    verbosity > 0 &&
        @info @sprintf("YangGaudinCMPS ground state: initialization with e = %.12f", E)

    x, E, grad, numfg, history = optimize(fg, x, optalg; retract=retract,
                                          precondition=precondition,
                                          (finalize!)=_finalize!,
                                          inner=inner, (transport!)=transport!,
                                          (scale!)=scale!, (add!)=add!,
                                          isometrictransport=true)
    (ΨL, ρR, HL, E, e, hL) = x
    normgrad = sqrt(inner(x, grad, grad))
    if verbosity > 0
        if normgrad <= gradtol
            @info @sprintf("YangGaudinCMPS ground state: converged after %d iterations: e = %.12f, ‖∇e‖ = %.4e",
                           size(history, 1), E, normgrad)
        else
            @warn @sprintf("YangGaudinCMPS ground state: not converged to requested tol: e = %.12f, ‖∇e‖ = %.4e",
                           E, normgrad)
        end
    end
    return ΨL, ρL, ρR, E, e, normgrad, numfg, history
end