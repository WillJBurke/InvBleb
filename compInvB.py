from dune.fem.space import lagrange
from dune.fem.operator import galerkin as galerkinOperator
from dune.fem.operator import linear as linearOperator
from dune.fem import assemble
from dune.fem.scheme import galerkin as galerkinScheme
from ufl import *
from dune.ufl import Constant, DirichletBC
import numpy as np


def compute(surfaceGrid, Param, defSecList, outputFlag, oFileName, deltaVis):
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
            return 1 + cos(0.5 * xval[0] * pi) * sin(0.5 * pi * xval[1])
        else:
            # All other tests
            return 1

    # Characteristic function for deformation
    def chi(defSec, xval, tval):
        # Checks characteristic for each deformed zone index
        return conditional(And(And(tval >= defSec.twl, tval <= defSec.twu),
                                (((xval[0] - defSec.x1wc) ** 2 + (xval[1] - defSec.x2wc) ** 2) <= defSec.rad**2)),
                            defSec.B, 0)

    def source(xval, tval):
        # Sum of all chi functions for given x,t
        return sum(chi(defSec, xval, tval) for defSec in defSecList)

    # Extract time/space conditions
    deltaT = Param.deltaT
    finalT = Param.finalT
    bval = Param.bval
    solutionSpace_u = lagrange(surfaceGrid, order=1, storage="istl")
    # solutionSpace_c becomes 2dim when protein density is implemented
    solutionSpace_c = lagrange(surfaceGrid, order=1, storage="istl")
    # Interpolate initial functions over spaces
    uh = solutionSpace_u.interpolate(initial_u, name="uh")
    omh = solutionSpace_u.interpolate([0]*solutionSpace_u.dimRange, name="omh")
    ch = solutionSpace_c.interpolate(initial_c, name="ch")
    zero_store = solutionSpace_u.interpolate([0]*solutionSpace_u.dimRange, name="storage")
    x = SpatialCoordinate(solutionSpace_u)
    # Boundary conditions: use for cg. u, om have same bc
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
    uh_n = uh.copy()
    omh_n = omh.copy()
    ch_n = ch.copy()

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
    l_pabs = set_constant("l_pabs")
    s_m = set_constant("s_m")
    s_tot = set_constant("s_tot")
    g = Constant(-2*s_s, "g")
    # c parameters
    l_d = set_constant("l_d")
    l_t = set_constant("l_t")
    n1 = set_constant("n1", default=0)
    n2 = set_constant("n2", default=0)
    om_bar = set_constant("om_bar", default=0)

    # Define functions used
    # _im is implicit in solver, _ex is explicit in solver, no suffix is pre-computed in solver
    bend_im = inner(grad(u), grad(z))
    curv_im = l_k * (inner(grad(om), grad(z)))
    cort_im = l_d * inner(grad(c), grad(b))
    op_split_height_im = inner(grad(u), grad(phi)) + g*phi
    op_split_curve_im = inner(om, phi)
    sig = s_m * (1-ch) + s_tot*ch
    l = (l_p * ch + l_pabs*(1-ch)) * z
    r = ((n1 * exp(-n2*(omh_n - om_bar)**2) * (1-ch_n) - source(x, t) * ch_n + l_t * (1-ch_n)) * b)

    # Define Models from functions
    curvModel = tau * curv_im * dx
    opSplitHeightModel = op_split_height_im * dx
    opSplitCurveModel = op_split_curve_im * dx
    uModel = ((l_a * (inner(u, z)) + tau*(sig*bend_im)) - (l_a * inner(uh_n, z) + tau * l)) * dx
    cortModel = ((inner(c, b) + (tau * cort_im)) - (inner(ch_n, b) + tau * r)) * dx

    # Initialise operators
    # galerkinOperators include explicit terms, linearOperators do not
    # galOp == A(x)-b, linOp = A(x)
    heightOp = galerkinOperator([uModel, bc_u_bot, bc_u_top])
    matHeightOp = linearOperator(heightOp)

    curvOp = galerkinOperator([curvModel, bc_u_bot, bc_u_top])
    matCurvOp = linearOperator(curvOp)

    # Schur compliment for om_h
    stiffOp = galerkinOperator([opSplitHeightModel, bc_u_bot, bc_u_top])
    matStiff = linearOperator(stiffOp)

    massOp = galerkinScheme([-opSplitCurveModel == 0, bc_u_bot, bc_u_top])
    matMass = linearOperator(massOp)
    innerSolver = {"method":"cg", "tolerance":1e-8, "verbose":False,
                   "preconditioning.method":"ilu"}
    matMassInv = massOp.inverseLinearOperator(matMass, parameters=innerSolver)

    # Cortex operations (c)
    cortOp = galerkinOperator(cortModel)
    matCortOp = linearOperator(cortOp)

    # c problem: (Mc + tau * l_d * Sc).c{n+1} = tau * r{n} + Mc.c{n}
    # c now reads: cortOp * ch = 0
    # u problem: (l_a*Mu + tau*l_d*Sc + tau*l_k*invMw*Sw)u{n+1} = l_a*Mu*u{n} + l{n}
    # u now reads: (heightOp + matCurvOp*matMassInv*stiffOp)uh = 0
    # om problem: om{n+1} = -invMw(u{n+1} + g*phi)
    # om now reads: omh = (matMassInv)uh

    #######################################
    # CG, Uzawa solvers
    #######################################
    # Initialise conditions for CG iterations
    tolCG = 1.0e-6
    max_iter = 1000
    resid = zero_store.copy()
    direc = zero_store.copy()
    # Initialise help functions for CG solver
    help = zero_store.copy()
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
                help.clear()
                # Store matCortOp . direc to reduce computation
                matCortOp(direc, help)

                # Update ch, residual
                alpha = delta / direc.scalarProductDofs(help)
                ch.axpy(alpha, direc)
                resid.axpy(-alpha, help)

                # Check residual and termination
                delta_old = delta
                delta = resid.scalarProductDofs(resid)
                norm_resid = sqrt(delta)
                if norm_resid <= tolCG:
                    break

                if m < max_iter-1:
                    beta = delta / delta_old
                    help.assign(direc)
                    direc.assign(resid)
                    direc.axpy(beta, help)
                else:
                    # Warn of terminated cg
                    print("Warning: cg of ch terminated by reaching the maximal number of iterations. The norm of the residual is ", norm_resid, flush=True)

        # Computation of uh{n+1} first residual
        # Compute schur complement
        stiffOp(uh, help1)
        help.clear()
        matMassInv(help1, help)
        stiffOp.setConstraints(help)
        matCurvOp(help, help2)
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
                help.clear()
                matMassInv(help1, help)
                matCurvOp(help, help2)
                curvOp.setConstraints(help2)
                matHeightOp(direc, help)
                heightOp.setConstraints(help)
                help -= help2

                # Update uh, residual
                alpha = delta / direc.scalarProductDofs(help)
                uh.axpy(alpha, direc)
                resid.axpy(-alpha, help)

                # Check residual and termination
                delta_old = delta
                delta = resid.scalarProductDofs(resid)
                norm_resid = sqrt(delta)
                if norm_resid <= tolCG :
                    break

                if m < max_iter-1:
                    beta = delta / delta_old
                    help.assign(direc)
                    direc.assign(resid)
                    direc.axpy(beta, help)
                else:
                    # Warn if cg fails
                    print("Warning: cg of uh terminated by reaching the maximal number of iterations. The norm of the residual is ", norm_resid, flush=True)
        # Compute omh{n+1} from found uh
        help.clear()
        stiffOp(uh, help)
        matMassInv(help, omh)
        stiffOp.setConstraints(omh)
        # Update actTime, output
        actTime = n * deltaT
        if visNext <= actTime:
            if outputFlag == 2:
                surfaceGrid.writeVTK(oFileName, pointdata=[uh, ch, omh], number=visCounter)
            visNext = visNext + deltaVis
            visCounter = visCounter + 1
        print(actTime, flush=True)


