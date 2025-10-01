# Gauges:
# :n => no particular gauge, left and right fixed points completely generic
# :l => left gauge: identity left fixed point, generic right fixed point
# :L => left canonical form: identity left fixed point, diagonal right fixed point
# :r => right gauge
# :R => right canonical form:
# :s => symmetric: left and right fixed point identical
# :S => symmetric: left and right fixed point identical and diagonal

mutable struct YangGaudinCMPS{T<:PeriodicMatrixFunction,N} <: LinearCMPS{T,N}
    Q::T
    Rs::NTuple{N,T}
    gauge::Symbol
    function YangGaudinCMPS(Q::T, Rs::NTuple{N,T}; gauge::Symbol=:n) where {T,N}
        for R in Rs
            domain(R) == domain(Q) || throw(DomainMismatch())
        end
        Q0 = Q[0]
        size(Q0, 1) == size(Q0, 2) || throw(DimensionMismatch())
        for R in Rs
            size(R[0]) == size(Q0) || throw(DimensionMismatch())
        end
        return new{T,N}(Q, Rs, gauge)
    end
end
YangGaudinCMPS(Q::T, R::T; kwargs...) where {T} = YangGaudinCMPS(Q, (R,); kwargs...)

domain(::YangGaudinCMPS) = (-Inf, +Inf)
period(Ψ::YangGaudinCMPS) = period(Ψ.Q)

Base.iterate(Ψ::YangGaudinCMPS, args...) = iterate((Ψ.Q, Ψ.Rs), args...)

Base.copy(Ψ::YangGaudinCMPS) = YangGaudinCMPS(copy(Ψ.Q), map(copy, Ψ.Rs); gauge=Ψ.gauge)

virtualdim(Ψ::YangGaudinCMPS) = size(Ψ.Q[0], 1)

function LeftTransfer(Ψ₁::YangGaudinCMPS, Ψ₂::YangGaudinCMPS=Ψ₁)
    domain(Ψ₁) == domain(Ψ₂) || throw(DomainMismatch())
    return LeftTransfer(Ψ₁.Q, sqrt(2) .* Ψ₁.Rs, Ψ₂.Q, sqrt(2) .* Ψ₂.Rs)
end

function RightTransfer(Ψ₁::YangGaudinCMPS, Ψ₂::YangGaudinCMPS=Ψ₁)
    domain(Ψ₁) == domain(Ψ₂) || throw(DomainMismatch())
    return RightTransfer(Ψ₁.Q, sqrt(2) .* Ψ₁.Rs, Ψ₂.Q, sqrt(2) .* Ψ₂.Rs)
end