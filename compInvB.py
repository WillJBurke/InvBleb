from dune.fem.space import lagrange
from dune.fem.operator import galerkin as galerkinOperator
from dune.fem.operator import linear as linearOperator
from dune.fem.function import integrate
from dune.fem.scheme import galerkin as galerkinScheme
from ufl import *
from dune.ufl import Constant, DirichletBC
import numpy as np


def compute(surfaceGrid, Param, weakSecList, outputFlag, oFileName, deltaVis):
    #######################################
    # initial values and setup
    #######################################
    s_s = Param.s_s

    def initial_u(xval):
        if Param.spring:
            # Spring test
            return 0.0
        else:
            # all others
            return s_s * xval[1] * (bval - xval[1])

    def initial_c(xval):
        if Param.cAn:
            # c analytic test
            return 1 + cos(0.5 * pi * xval[0]) * cos(0.5 * pi * xval[1])
        else:
            # All other tests
            return 1

    # Characteristic function for weakened section
    def chi(weakSec, xval, tval):
        # Checks characteristic for each deformed zone index
        return conditional(And(And(tval >= weakSec.twl, tval <= weakSec.twu),
                                (((xval[0] - weakSec.x1wc) ** 2 + (xval[1] - weakSec.x2wc) ** 2) <= weakSec.rad**2)),
                            weakSec.ups, 0)

    def source(xval, tval):
        # Sum of all chi functions for given x,t
        return sum(chi(weakSec, xval, tval) for weakSec in weakSecList)

    # Extract time/space conditions
    deltaT = Param.deltaT
    finalT = Param.finalT
    bval = Param.bval
    # Both are order 1 spaces: piecewise linear approximations
    solutionSpace_u = lagrange(surfaceGrid, order=1, storage="istl")
    solutionSpace_c = lagrange(surfaceGrid, order=1, storage="istl")
    # Interpolate initial functions over spaces
    uh = solutionSpace_u.interpolate(initial_u, name="uh")
    omh = solutionSpace_u.interpolate([0]*solutionSpace_u.dimRange, name="omh")
    ch = solutionSpace_c.interpolate(initial_c, name="ch")
    # 0 Vector: CG initialisation
    zero_store = solutionSpace_u.interpolate([0]*solutionSpace_u.dimRange, name="storage")
    x = SpatialCoordinate(solutionSpace_u)
    # Boundary conditions: use for cg. u, om have homogenous dirichlet BC on Omega_D
    bc_u_top = DirichletBC(solutionSpace_u, 0, x[1] > bval - 1.0e-10)
    bc_u_bot = DirichletBC(solutionSpace_u, 0, x[1] < 1.0e-10)

    # Initial graphical output
    visCounter = 0
    # Set our act time to 0
    actTime = 0.0
    if outputFlag == 2:
        surfaceGrid.writeVTK(oFileName, pointdata=[uh, ch, omh], number=visCounter)
        visCounter += 1
    # Discrete function for the actual height, curvature, cortex concentration
    uh_n  = uh.copy()
    omh_n = omh.copy()
    ch_n  = ch.copy()

    #######################################
    # Model definition
    #######################################
    # Initialise test/trial functions
    u   = TrialFunction(solutionSpace_u)
    z   = TestFunction(solutionSpace_u)
    om  = TrialFunction(solutionSpace_u)
    phi = TestFunction(solutionSpace_u)
    c   = TrialFunction(solutionSpace_c)
    b   = TestFunction(solutionSpace_c)

    # Extract constants from Param
    def set_constant(attribute, default=None):
        return Constant(getattr(Param, attribute), attribute) if default is None else \
            Constant(getattr(Param, attribute, default), attribute)
    # Time step and current time parameters
    tau = Constant(deltaT, "Tau")
    t = Constant(0, "t")
    # u, om parameters
    l_a = set_constant("l_a")
    l_k = set_constant("l_k")
    l_p = set_constant("l_p")
    l_l = set_constant("l_l")
    s_m = set_constant("s_m")
    s_tot = set_constant("s_tot")
    g = Constant(-2*s_s, "g")
    # c parameters
    l_d = set_constant("l_d")
    l_t = set_constant("l_t")
    # new reaction term
    n1 = set_constant("n1", default=0)
    n2 = set_constant("n2", default=0)
    om_bar = set_constant("om_bar", default=0)
    # non-cortex dependant pressure
    p_c = set_constant("p_c", default=0)

    # Define functions used
    # _im: Implicit functions
    bend_im = inner(grad(u), grad(z))
    curv_im = l_k * (inner(grad(om), grad(z)))
    cort_im = l_d * inner(grad(c), grad(b))
    op_split_height_im = inner(grad(u), grad(phi)) + g*phi
    op_split_curve_im = inner(om, phi)
    # _ex: Explicit functions
    # sig_ex, l_ex use ch{n+1}: due to solver construction these are pre-computed so treated explicitly
    sig_ex = s_m * (1-ch) + s_tot*ch
    l_ex = (l_p * ch + l_l*(1-ch) + p_c) * z
    r_ex = ((n1 * exp(-n2*(omh_n - om_bar)**2) * (1-ch_n) - source(x, t) * ch_n + l_t * (1-ch_n)) * b)
    # Define Models from functions
    # Fourth order terms (Omega, U):
    curvModel = tau * curv_im * dx
    opSplitHeightModel = op_split_height_im * dx
    opSplitCurveModel = op_split_curve_im * dx
    # Height (no fourth order)
    uModel = ((l_a * (inner(u, z)) + tau*(sig_ex*bend_im)) - (l_a * inner(uh_n, z) + tau * l_ex)) * dx
    # Cortex
    cortModel = ((inner(c, b) + (tau * cort_im)) - (inner(ch_n, b) + tau * r_ex)) * dx

    # Initialise operators
    # galerkinOperators include explicit terms, linearOperators do not
    # galOp == A(x)-b, linOp = A(x)
    heightOp = galerkinOperator([uModel, bc_u_bot, bc_u_top])
    matHeightOp = linearOperator(heightOp)

    curvOp = galerkinOperator([curvModel, bc_u_bot, bc_u_top])
    matCurvOp = linearOperator(curvOp)

    # Schur compliment (om, u)
    stiffOp = galerkinOperator([opSplitHeightModel, bc_u_bot, bc_u_top])
    matStiff = linearOperator(stiffOp)

    massOp = galerkinScheme([-opSplitCurveModel == 0, bc_u_bot, bc_u_top])
    matMass = linearOperator(massOp)
    innerSolver = {"method": "cg", "tolerance": 1e-10, "verbose": False,
                   "preconditioning.method": "ilu"}
    matMassInv = massOp.inverseLinearOperator(matMass, parameters=innerSolver)

    # Cortex operations (c)
    cortOp = galerkinOperator(cortModel)
    matCortOp = linearOperator(cortOp)

    # c problem: (Mc + tau * l_d * Sc).c{n+1} = tau * r{n} + Mc.c{n}
    # c now reads: cortOp * ch = 0
    # u problem: (l_a*Mu + tau*Sigma_c*S_u + tau*l_k*Mw{-1}*Sw)u{n+1} = l_a*Mu*u{n} + l{n}
    # u now reads: (heightOp + matCurvOp*matMassInv*stiffOp)uh = 0
    # om problem: om{n+1} = -invMw(u{n+1} + g*phi)
    # om now reads: omh = (matMassInv)uh

    #######################################
    # CG, Uzawa solvers
    #######################################
    # Initialise conditions for CG iterations
    tolCG = 1.0e-8
    max_iter = 1000
    resid = zero_store.copy()
    direc = zero_store.copy()
    # Initialise help functions for CG solver
    help0 = zero_store.copy()
    help1 = zero_store.copy()
    help2 = zero_store.copy()
    # Define our num steps, next visualisation
    steps = int(finalT/deltaT)
    visNext = deltaVis

    for n in range(0, steps+1):
        # Set boundary constraints on uh, omh
        heightOp.setConstraints(uh)
        heightOp.setConstraints(omh)
        # Copy current values into _n values
        uh_n.assign(uh)
        ch_n.assign(ch)
        omh_n.assign(omh)
        # Re-assign time t for char
        t.assign(actTime)

        # c computation of first resid:
        cortOp(ch, resid)
        resid *= -1

        # Terminate if resid below tol
        delta = resid.scalarProductDofs(resid)
        norm_resid = sqrt(delta)
        if norm_resid >= tolCG:
            direc.assign(resid)
            for m in range(max_iter):
                help0.clear()
                # Store matCortOp . direc to reduce computation
                matCortOp(direc, help0)

                # Update ch, residual
                alpha = delta / direc.scalarProductDofs(help0)
                ch.axpy(alpha, direc)
                resid.axpy(-alpha, help0)

                # Check residual and termination
                delta_old = delta
                delta = resid.scalarProductDofs(resid)
                norm_resid = sqrt(delta)
                if norm_resid <= tolCG:
                    break

                if m < max_iter-1:
                    beta = delta / delta_old
                    help0.assign(direc)
                    direc.assign(resid)
                    direc.axpy(beta, help0)
                else:
                    # Warn of terminated cg
                    print("Warning: cg of ch terminated by reaching the maximal number of iterations."
                          " The norm of the residual is ", norm_resid, flush=True)

        # Computation of uh{n+1} first residual
        # Compute schur complement
        stiffOp(uh, help1)
        help0.clear()
        matMassInv(help1, help0)
        stiffOp.setConstraints(help0)
        matCurvOp(help0, help2)
        curvOp.setConstraints(help2)
        # u without schur
        heightOp(uh, resid)
        heightOp.setConstraints(resid)
        resid -= help2
        resid *= -1

        # CG alg for uh
        # Terminate if resid below tol
        delta = resid.scalarProductDofs(resid)
        norm_resid = sqrt(delta)
        if norm_resid >= tolCG:
            direc.assign(resid)
            for m in range(max_iter):
                # Apply the system matrix to the search direction
                matStiff(direc, help1)
                help0.clear()
                matMassInv(help1, help0)
                matCurvOp(help0, help2)
                curvOp.setConstraints(help2)
                matHeightOp(direc, help0)
                heightOp.setConstraints(help0)
                help0 -= help2

                # Update uh, residual
                alpha = delta / direc.scalarProductDofs(help0)
                uh.axpy(alpha, direc)
                resid.axpy(-alpha, help0)

                # Check residual and termination
                delta_old = delta
                delta = resid.scalarProductDofs(resid)
                norm_resid = sqrt(delta)
                if norm_resid <= tolCG:
                    break

                if m < max_iter-1:
                    beta = delta / delta_old
                    help0.assign(direc)
                    direc.assign(resid)
                    direc.axpy(beta, help0)
                else:
                    # Warn if cg fails
                    print("Warning: cg of uh terminated by reaching the maximal number of iterations."
                          " The norm of the residual is ", norm_resid, flush=True)
        # Compute omh{n+1} from found uh
        help0.clear()
        stiffOp(uh, help0)
        matMassInv(help0, omh)
        stiffOp.setConstraints(omh)
        # Update actTime, output
        actTime = n * deltaT
        if visNext <= actTime:
            if outputFlag == 2:
                surfaceGrid.writeVTK(oFileName, pointdata=[uh, ch, omh], number=visCounter)
            visNext = visNext + deltaVis
            visCounter = visCounter + 1
        print(actTime, flush=True)

    # Error Analysis (done after computation)
    if Param.errAnal:
        def diff_compute(grid, approx_func, exact_func, funcName):
            # Computes H1, L2 error for given function
            print('grid size:', grid.size(0), 'h =', Param.h, 'tau =', deltaT, 'T =', finalT)
            h1error = inner(grad(approx_func - exact_func), grad(approx_func - exact_func))
            l2error = inner(approx_func - exact_func, approx_func - exact_func)
            # Integrate errors across grid
            errors = [np.sqrt(e) for e in integrate(grid, [h1error, l2error], order=2 * solutionSpace_c.order)]
            print(f'\t {funcName} error:')
            print(f'\t | grad({funcName}_h - {funcName}) | =', '{:0.5e}'.format(errors[0]))
            print(f'\t | {funcName}_h - {funcName} | =', '{:0.5e}'.format(errors[1]))
            return errors

        if not Param.cAn:
            # Test for u convergence (stable solution)
            exact = s_s * x[1] * (bval - x[1])
            approx = uh
            func_name = "u"
        else:
            # Test for c convergence
            # Exact taken from literature
            exact = 1 + (exp(-0.5 * (pi**2) * finalT)*cos(0.5 * pi * x[0]) * cos(0.5 * pi * x[1]))
            approx = ch
            func_name = "c"

        h_error = diff_compute(surfaceGrid, approx, exact, func_name)
        # Only want a return if conducting analysis
        return h_error



