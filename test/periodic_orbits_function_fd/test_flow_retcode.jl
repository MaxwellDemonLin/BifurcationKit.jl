using Test
import BifurcationKit as BK
import SciMLBase
using OrdinaryDiffEqRosenbrock: Rodas5P

@testset "Flow retcode checks and PALC retry" begin
    f! = (du, u, p, t) -> (du[1] = p[1] - u[1])
    ode = SciMLBase.ODEProblem(f!, [0.2], (0.0, 1.0), [0.2])
    alg = Rodas5P()
    failed_solves = Ref(0)
    unstable = (dt, u, p, t) -> begin
        bad = p[1] > 0.35
        bad && (failed_solves[] += 1)
        bad
    end
    flow = BK.Flow(ode, alg; abstol=1e-10, reltol=1e-10,
        unstable_check=unstable)
    @test BK.evolve(flow, [0.2], [0.2], 1.0).u[1] ≈ 0.2
    @test_throws BK.FlowIntegrationFailure BK.evolve(flow, [0.4], [0.4], 1.0)
    @test_throws BK.FlowIntegrationFailure BK.evolve(flow, Val(:Full), [0.4], [0.4], 1.0)

    # A successful callback termination is used by Poincare shooting.
    stop = SciMLBase.ContinuousCallback((u, t, integrator) -> t - 0.1,
        SciMLBase.terminate!)
    event_flow = BK.Flow(ode, alg; callback=stop)
    @test BK.evolve(event_flow, [0.2], [0.2], 1.0).t ≈ 0.1

    calls = Ref(0)
    jac_calls = Ref(0)
    fail_jacobian = Ref(false)
    unrelated_bug = Ref(false)
    residual = (x, p) -> begin
        calls[] += 1
        unrelated_bug[] && error("unrelated model bug")
        BK.evolve(flow, x, p, 1.0).u .- x
    end
    jac = (x, p) -> begin
        jac_calls[] += 1
        fail_jacobian[] && BK.evolve(flow, [0.4], [0.4], 1.0)
        reshape([exp(-1.0) - 1], 1, 1)
    end
    pb = BK.BifurcationProblem(residual, [0.2], [0.2], BK.@optic(_[1]); J=jac)
    opts = BK.ContinuationPar(ds=0.2, dsmin=0.025, dsmax=0.2,
        p_min=0.0, p_max=1.0, detect_bifurcation=0, detect_fold=false,
        max_steps=12, newton_options=BK.NewtonPar(tol=1e-8, linesearch=true))
    it = BK.ContIterable(pb, BK.PALC(bls=BK.MatrixBLS()), opts)
    state, = iterate(it)
    @test state.z.p ≈ 0.2
    failed_solves[] = calls[] = jac_calls[] = 0
    iterate(it, state)
    @test !state.converged
    @test !state.stopcontinuation
    @test state.ds ≈ 0.1
    @test state.z.p ≈ 0.2
    @test state.z.u[1] ≈ 0.2
    @test calls[] == 1
    @test failed_solves[] == 1
    @test jac_calls[] == 0

    # Retry from the last valid point succeeds at the halved step.
    iterate(it, state)
    @test state.converged
    @test state.z.p ≈ 0.3
    @test state.z.u[1] ≈ 0.3

    # Failure while constructing the Jacobian also rejects immediately.
    state.z_pred.u[1] = 0.28
    state.z_pred.p = 0.3
    fail_jacobian[] = true
    failed_solves[] = jac_calls[] = 0
    BK.corrector!(state, it)
    @test !state.converged
    @test state.z.p ≈ 0.3
    @test state.z.u[1] ≈ 0.3
    @test jac_calls[] == 1
    @test failed_solves[] == 1
    fail_jacobian[] = false

    # A failed line-search trial must not be treated as a valid residual.
    state.ds = 0.1
    failed_solves[] = 0
    BK.corrector!(state, it)
    @test !state.converged
    @test state.z.p ≈ 0.3
    @test failed_solves[] == 1

    # Failure at dsmin stops instead of repeatedly trying the same step.
    state.ds = opts.dsmin
    state.z_pred.p = 0.4
    state.z_pred.u[1] = 0.4
    iterate(it, state)
    @test !state.converged
    @test state.stopcontinuation
    @test state.z.p ≈ 0.3

    # Only integration-failure exceptions are converted to rejected steps.
    unrelated_bug[] = true
    @test_throws ErrorException BK.corrector!(state, it)
end

struct FixedRetcodeTestPeriod end
(::FixedRetcodeTestPeriod)(x, T) = T - 1.0
(::FixedRetcodeTestPeriod)(x, T, dx, dT) = dT

@testset "Native Shooting finite-difference continuation" begin
    failed_solves = Ref(0)
    f! = (du, u, p, t) -> (du[1] = p[1] - u[1])
    ode = SciMLBase.ODEProblem(f!, [0.2], (0.0, 1.0), [0.2])
    unstable = (dt, u, p, t) -> begin
        bad = p[1] > 0.35
        bad && (failed_solves[] += 1)
        bad
    end
    sh = BK.Shooting(ode, Rodas5P(), 1,
        FixedRetcodeTestPeriod(); lens=BK.@optic(_[1]),
        jacobian=BK.FiniteDifferences(), update_section_every_step=0,
        abstol=1e-10, reltol=1e-10, unstable_check=unstable)
    opts = BK.ContinuationPar(ds=0.2, dsmin=0.025, dsmax=0.2,
        p_min=0.0, p_max=1.0, detect_bifurcation=0, detect_fold=false,
        max_steps=12, newton_options=BK.NewtonPar(tol=1e-8))
    branch = BK.continuation(sh, [0.2, 1.0], BK.PALC(), opts, BK.MatrixBLS();
        record_from_solution=(z, ctx; kwargs...) -> (; value=z[1]),
        verbosity=1)
    @test length(branch.branch) > 1
    @test all(pt -> pt.param <= 0.35, branch.branch)
    @test last(branch.branch).param >= 0.3 - 1e-8
    @test failed_solves[] > 0
end
