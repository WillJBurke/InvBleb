from dune.grid import structuredGrid, cartesianDomain
from dune.ufl import Constant
from compInvB import compute
from dune.alugrid import aluConformGrid
from mpi4py import MPI
from functools import partial
from dune.common import comm
import numpy as np
import time


# prevents parallel error with print statements in compute
# Will not be necessary on DUNE-fem 2.1.0
print = partial(print, flush=True) if comm.rank == 0 else lambda *args, **kwargs: None

#######################################
# Notes on model and code
#######################################
# This code is written as part of a masters dissertation: once completed it will be uploaded to the same github page
# Please see the above paper for a more full description of the model used

# Planned future changes
# Conduct error analysis for u, c in tests (0)(1), (0)(2) respectively
# Add marker to construct grid and refine/coarsen in compute
# Coarsen grid for non-deformed sections, refine for deformed sections
# Alter storage of computed functions for faster CPU time: delays in reading/writing
# Alter structure of parameter class for greater changes to tests, values

# Important note: this is run on dune-fem 2.9.0
# An updated version on dune-fem 2.1.0 will be pushed at a future date
# Known errors (DUNE 2.9.0.2)
# aluConformGrid does not have bisection compatibility (still functions as expected)
# aluConformGrid leaks 1 yasp object
# Error calculation leaks 1 yasp object for each error
# Will be fixed upon 2.1.0 update and necessary rewrites
# Not updated as should errors occur in installation, additional coding time will run over submission deadline

#######################################
# Class definitions
#######################################


# Param class: allows easy passing of ufl constant without re-compiling integrands
# See documentation for complete description of default values
class Param:
    # Parameters marked with ! are altered for various test functions
    # Height, curvature parameters
    l_a = 1.22  # 1.22 (rigidity of height (u)) !
    l_k = 0.0423  # 0.0423 (effect of curvature (omega) on height)
    l_p = 0.83  # 0.83 (initial pressure difference)
    l_pabs = 12.45  # l_p * factor (pressure difference when cortex density (c) < 1) !
    s_m = 0.4  # 0.4 (tension term inversely proportional to cortex density)
    s_tot = 4.14  # 3.74 (tension term directly proportional to cortex density)
    s_s = 0.1  # 0.1 (scalar for height of cell)
    # Cortex parameters
    l_d = 6.66  # 6.66 (diffusivity term for deformation in c) !
    l_t = 2.0  # 2.0 (reaction coefficient for reformation of c)
    # Spatial parameters
    lval = 12  # Length of cell
    bval = 8  # Width of the cell
    h = 0.1  # Spatial step size !
    deltaT = 0.0025  # Time step size
    finalT = 3  # Total runtime (minutes) !
    spring = False  # spring test boolean !
    cAn = False  # c analytic test boolean !


class WeakSecParams:
    # Parameters for characteristic of deformation zone
    def __init__(self, x1wc, x2wc, rad, twl, twu, B):
        self.x1wc = x1wc  # x1 centre for deform circle
        self.x2wc = x2wc  # x2 centre for deform circle
        self.rad = rad  # Radius of deform circle
        self.twl = twl  # Time lower bound
        self.twu = twu  # Time upper bound
        self.B = B  # Strength of weakening force


#######################################
# Tests and settings
#######################################
# Initialise our model from default parameters
Model = Param
# Create list of deformed sections, append as needed
weakSecList = []

# Tests
# System tests (0)
# (0)(0): Stable test: run with default params, no deformed sections
# (0)(1): Spring test: initial u=0, c=1, test that stable solution attracts. Adjust time scale to see full spring back
# (0)(2): C test: Ignore u, pass c as analytic solution to test accuracy of model. Adjust l_d, ignore reaction term
# Application tests (1)
# (1)(0): Simple Bleb: run with singular deformed section, default params
# (1)(1): Double Bleb Close: initialise second deform section object close to existing.
# (1)(2): Double Bleb Far B: second deformed section far away, vary B to see effect on resulting bleb
# (1)(3): Double Bleb Far B: second deformed section far away, vary twl to see effect on resulting bleb
# (1)(4): New Reaction: implement phi(omega) in reaction term for c
# l_pabs fitting tests (2)
# (2)(1-4) run with various scales of l_pabs:l_p (10*, 15*, 25*, 40*) to fit literature
# From fitting, we now have l_pabs = 12.45 (15 * l_p)

# Alter to change test run
test_set = 0  # Set of tests run: 0 = system tests, 1 = application tests, 2 = l_pabs fitting
test_spec = 2  # Specific test run: see above for description of each test in set
fileNames = [
    ["./VTLStore/SystemTests/StableTest/stable-sol", "./VTLStore/SystemTests/SpringTest/spring-test",
        "./VTLStore/SystemTests/cHeatTest/cTest"],  # system tests
    ["./VTLStore/AppTests/SimpleBleb/simple-bleb", "./VTLStore/AppTests/DoubleBleb/double-blebC",
        "./VTLStore/AppTests/DoubleBleb/double-blebFB", "./VTLStore/AppTests/DoubleBleb/double-blebFT",
        "./VTLStore/AppTests/NewReac/newReac-bleb"],  # application tests
    ["./VTLStore/PressureTests/Lpabs8/newP8-bleb", "./VTLStore/PressureTests/Lpabs12/newP12-bleb",
        "./VTLStore/PressureTests/Lpabs20/newP20-bleb", "./VTLStore/PressureTests/Lpabs40/newP-40bleb"]  # l_pabs fit
    ]
if test_set == 0:
    # System tests
    if test_spec == 1:
        # Spring test
        Model.spring = True
    elif test_spec == 2:
        # C analytic test
        Model.cAn = True
        # Overwrite for analytic solutions
        Model.l_d = 1
        Model.l_t = 0
        Model.finalT = 0.5
elif test_set == 1:
    # Application tests
    # Initial weakened section values
    x1wc1 = Constant(3.5, name="x1wc1")
    x2wc1 = Constant(3.5, name="x2wc1")
    rad1 = Constant(1, name="rad1")
    twl1 = Constant(0.0, name="twl1")
    twu1 = Constant(0.2, name="twu1")
    B1 = Constant(100, name="B1")
    # First weakened section
    weakSec1Param = WeakSecParams(x1wc1, x2wc1, rad1, twl1, twu1, B1)
    weakSecList.append(weakSec1Param)

    if 1 <= test_spec <= 3:
        # Multiple deformed sections
        # Default settings for secondary section
        x1wc2 = Constant(5.5, name="x1wc2")
        x2wc2 = Constant(5.5, name="x2wc2")
        rad2 = Constant(1, name="rad2")
        twl2 = Constant(0.0, name="twl2")
        twu2 = Constant(0.2, name="twu2")
        B2 = Constant(100, name="B2")
        if test_spec == 2:
            # Assign mirrored of defSec1, B lower
            x1wc2.assign(8.5)
            x2wc2.assign(4.5)
            B2.assign(40)
        elif test_spec == 3:
            # Assign mirrored of defSec1, shorter time
            x1wc2.assign(8.5)
            x2wc2.assign(4.5)
            twl2.assign(0.1)

        weakSec2Param = WeakSecParams(x1wc2, x2wc2, rad2, twl2, twu2, B2)
        weakSecList.append(weakSec2Param)
    elif test_spec == 4:
        # Activate phi(omega) in reaction term
        Model.n1 = 1.5  # 1.5 (phi(omega): dictates contribution of phi to r)
        Model.n2 = 1  # 1 (phi(omega): dictates shape of phi(omega))
        Model.om_bar = -4.2  # -4.2 (taken from om at bleb tip App 1: dictates peak of phi(omega))
else:
    # Fittings for l_pabs
    # Alter l_pabs to specified value for each test
    pressureVals = [8.3, 12.45, 20.75, 40]
    Model.l_pabs = pressureVals[test_spec]


#######################################
# initialise arguments for compute
#######################################
# Visualisation
outputFlag = 2  # 1 = do not visualise, 2 = write vtk to specified file for specified time steps
oFileName = fileNames[test_set][test_spec]
deltaVis = 0.01  # Visualisation time step size


# Initialise grid
# Standard surface Grid
#surfaceGrid = structuredGrid([0, 0], [Model.lval, Model.bval], [int(Model.lval/Model.h), int(Model.bval/Model.h)])
# Alternative: aluConformGrid for refinement
# TODO: implement refinement around deformation in u
# Note: for conform grid, actual h is sqrt(2)*Model.h
domain = cartesianDomain([0, 0], [Model.lval, Model.bval], [int(Model.lval/Model.h), int(Model.bval/Model.h)])
surfaceGrid = aluConformGrid(constructor=domain)


# Run desired compute function, calculate time taken
start_time = time.time()
compute(surfaceGrid, Model, weakSecList, outputFlag, oFileName, deltaVis)
# Check time taken for computation
# ~ 40 minute computation time with visualisation
print("--- %s seconds ---" % (time.time()-start_time), flush=True)

# TODO: complete error analysis on spring, cAn test
