import math

from dune.grid import cartesianDomain
from dune.ufl import Constant
from compInvB import compute
from dune.alugrid import aluConformGrid
from mpi4py import MPI
from functools import partial
from dune.common import comm
import numpy as np
import time
import matplotlib.pyplot as plt


# Prevents parallel error with print statements in compute
# Will not be necessary on DUNE-fem 2.10.0
print = partial(print, flush=True) if comm.rank == 0 else lambda *args, **kwargs: None

#######################################
# Notes on model and code
#######################################
# This code is written as part of a masters dissertation: once completed it will be uploaded to the same github page
# Please see the above paper for a more full description of the model used

# Planned future changes
# Add marker to construct grid and refine/coarsen in compute
# Coarsen grid for non-deformed sections, refine for deformed sections

# Important note: this is run on dune-fem 2.9.0
# An updated version on dune-fem 2.10.0 will be pushed at a future date
# Known errors (DUNE 2.9.0.2)
# aluConformGrid does not have bisection compatibility (still functions as expected)
# aluConformGrid leaks 1 yasp object
# Error calculation leaks 1 yasp object for each error
# Will be fixed upon 2.10.0 update and necessary rewrites
# Not updated as should errors occur in installation, additional coding time will run over submission deadline

#######################################
# Class definitions
#######################################


# Param class: allows easy passing of ufl constant without re-compiling integrands
# See documentation for complete description of default values
class Param:
    # Parameters marked with ! are altered for various test functions
    # Height, curvature parameters
    l_a = 1.22  # 1.22 (Viscosity of cytoplasm)
    l_k = 0.0423  # 0.0423 (Bending rigidity of membrane (omega))
    l_p = 0.83  # 0.83 (Pressure difference for non disrupted cortex (c))
    l_l = 12.45  # l_p * factor (Pressure difference for disrupted cortex (c)) !
    s_m = 0.4  # 0.4 (Surface tension for disrupted cortex (c))
    s_tot = 4.14  # 4.14 (Surface Tension for non disrupted cortex)
    s_s = 0.1  # 0.1 (Scalar for height of cell)
    # Cortex parameters
    l_d = 6.66  # 6.66 (Diffusivity of deformation in c) !
    l_t = 2.0  # 2.0 (Reaction coefficient for reformation of c)
    # Spatial parameters
    lval = 12  # Length of cell
    bval = 8  # Width of the cell
    h = 0.1  # Spatial step size
    deltaT = 0.0025  # Time step size
    finalT = 3  # Final time T (minutes)
    spring = False  # Spring test boolean (0)(2)
    cAn = False  # c analytic test boolean (0)(3-4)
    errAnal = False  # EOC analysis boolean (0)(1,4)


class WeakSecParams:
    # Parameters for characteristic of deformation zone
    def __init__(self, x1wc, x2wc, rad, twl, twu, ups):
        self.x1wc = Constant(x1wc, name="x1wc")  # x1 centre for weakened circle
        self.x2wc = Constant(x2wc, name="x2wc")  # x2 centre for weakened circle
        self.rad = Constant(rad, name="rad")  # Radius of weakened circle
        self.twl = Constant(twl, name="twl")  # Time lower bound
        self.twu = Constant(twu, name="twu")  # Time upper bound
        self.ups = Constant(ups, name="ups")  # Strength of weakening force


#######################################
# Tests and settings
#######################################
# Initialise model from default parameters
Model = Param
# Create list of deformed sections, append as needed
weakSecList = []

# Tests
# System tests (0)
# (0)(0): Stable test: run with default params, no deformed sections
# (0)(1): Error Analysis on (0)(0)
# (0)(2): Spring test: initial u=0, c=1, test that stable solution attracts. Adjust time scale to see full spring back
# (0)(3): C test: Ignore u, pass c as analytic solution to test accuracy of model. Adjust l_d, ignore reaction term
# (0)(4): Error Analysis on (0)(3)
# l_l fitting tests (1)
# (1)(0-3) run with various scales of l_l = l_p * [10, 15, 25, 40] to get visible deformation
# From fitting, we now have l_l = 12.45 (= 15 * l_p), used in App Tests (2), refinements (3)
# Application tests (2)
# (2)(0): Simple Bleb: run with singular deformed section, default params
# (2)(1): Double Bleb Close: initialise second deform section object close to existing
# (2)(2): Double Bleb Far B: second deformed section far away, vary B to see effect on resulting bleb
# (2)(3): Double Bleb Far B: second deformed section far away, vary twl to see effect on resulting bleb
# Refinements and parameter studies (3)
# Only (3)(0, 1, 2) are used in report
# (3)(0) New Reaction: implement eta(omega) in reaction term for c
# (3)(1) Diffusion test: reduce l_d to 0.1: observe effect of eta(omega)
# (3)(2) No fourth-order: set l_k =0 to compare with simpleBleb (1)(0)
# (3)(3) Constant pressure: set l_p, l_l to 0, pass a constant pressure term l_c equal to default l_p
# (3)(4) Constant pressure: set l_p, l_l to 0, pass l_c = l_l
# (3)(5) Rigidity test: reduce l_a by factor of 2, observe resulting dynamics in bleb shape (height increase expected)

# Test set and specification
test_set = 3  # Set of tests run: 0 = System tests, 1 = l_l fitting, 2 = Application tests, 3 = Model alterations
test_spec = 5  # Specific test run: see above for description of each test in set
outputFlag = 2  # 1 = do not visualise, 2 = write vtk to specified file for specified time steps
testNames = [["Stable sol", "Stable EOCs", "Spring u", "C analytic", "C Errors"],
             ["l_l = 8.3", "l_l = 12.45", "l_l = 20.75", "l_l = 40"],
             ["Single Bleb", "Double bleb close", "Double bleb varying ups", "Double bleb varying T"],
             ["New reaction", "New reaction reduced diffusivity", "No fourth-order term", "Constant pressure low",
              "Constant pressure high", "Rigidity reduction"]]

# Visualisation file paths
fileNames = [
     [  # System tests
        "./VTLStore/SystemTests/StableTest/stable-sol", "./VTLStore/SystemTests/StableTest/stable-sol_Err",
        "./VTLStore/SystemTests/SpringTest/spring-test", "./VTLStore/SystemTests/cHeatTest/cTest",
        "./VTLStore/SystemTests/cHeatTest/cTest_Err"
     ], [  # l_l fitting
        "./VTLStore/PressureTests/Lpabs8/newP8-bleb", "./VTLStore/PressureTests/Lpabs12/newP12-bleb",
        "./VTLStore/PressureTests/Lpabs20/newP20-bleb", "./VTLStore/PressureTests/Lpabs40/newP-40bleb"
     ], [  # Application Tests
        "./VTLStore/AppTests/SimpleBleb/simple-bleb", "./VTLStore/AppTests/DoubleBlebClose/double-blebC",
        "./VTLStore/AppTests/DoubleVarB/double-blebFB", "./VTLStore/AppTests/DoubleVarT/double-blebFT"
     ], [  # Refinements and parameter study
        "./VTLStore/ModelAltTests/NewReac/newReac-bleb", "./VTLStore/ModelAltTests/NewReacDiff/newReacD-bleb",
        "./VTLStore/ModelAltTests/NoFourth/noFourth-bleb", "./VTLStore/ModelAltTests/ConstPressure/constPressure-bleb",
        "./VTLStore/ModelAltTests/RigidityRed/reduceAlp-bleb"
     ]
    ]
oFileName = fileNames[test_set][test_spec]
print("Running test: ", testNames[test_set][test_spec], ", Visualisation =", outputFlag, flush=True)
if Model.finalT < 5:
    deltaVis = 0.01  # Visualisation step size
else:
    deltaVis = Model.finalT/500  # Prevents excessive VTK creation (storage concerns)

# Adjust parameters and settings for test chosen
if test_set == 0:
    # System tests
    if test_spec == 1 or test_spec == 4:
        # Error Analysis: Adjust final time, do not write VTKs
        Model.errAnal = True
        outputFlag = 1  # Do not write VTKs for analysis: will take too much storage
        Model.finalT = 0.3
    if test_spec <= 1:
        # Stable Test
        Model.finalT = 0.3
    if test_spec == 2:
        # Spring test
        Model.spring = True
        Model.finalT = 10
    elif test_spec >= 3:
        # C analytic tests
        Model.cAn = True
        # Adjust to equilibrium solution
        Model.l_d = 1
        Model.l_t = 0
        # Overwrite end time for analytic solutions
        if test_spec == 3:
            Model.finalT = 0.5
else:
    # Initial weakened section values (used for all other tests)
    x1wc1 = 3.5
    x2wc1 = 3.5
    rad1 = 1
    twl1 = 0.0
    twu1 = 0.2
    ups1 = 100
    # First weakened section
    weakSec1Param = WeakSecParams(x1wc1, x2wc1, rad1, twl1, twu1, ups1)
    weakSecList.append(weakSec1Param)
    if test_set == 1:
        # Fittings for l_l: Alter l_l to specified value for each test
        pressureVals = [8.3, 12.45, 20.75, 40]
        Model.l_l = pressureVals[test_spec]
    elif test_set == 2:
        # Application tests
        if test_spec != 0:
            # Multiple weakened sections
            # Constant parameters across settings
            rad2 = 1
            twu2 = 0.2
            if test_spec == 1:
                # Close to weakSec1, identical t, ups values
                x1wc2 = 5.5
                x2wc2 = 5.5
                twl2 = 0
                ups2 = 100
            elif test_spec == 2:
                # Assign mirrored of weakSec1, ups lower
                x1wc2 = 8.5
                x2wc2 = 4.5
                twl2 = 0
                ups2 = 40
            else:
                # Assign mirrored of weakSec1, shorter time
                x1wc2 = 8.5
                x2wc2 = 4.5
                twl2 = 0.1
                ups2 = 100

            weakSec2Param = WeakSecParams(x1wc2, x2wc2, rad2, twl2, twu2, ups2)
            weakSecList.append(weakSec2Param)
    elif test_set == 3:
        if test_spec <= 1:
            # Activate eta(omega) in reaction term
            Model.n1 = 1.5  # 1.5 (Dictates contribution of eta to r)
            Model.n2 = 1  # 1 (Dictates shape of eta(omega))
            Model.om_bar = -4.2  # -4.2 (Taken from om at bleb tip App 1: dictates peak of eta(omega))
            if test_spec == 1:
                # Reduce diffusion
                Model.l_d = 0.1  # (Reduce l_d to increase proportional effect of eta)
                Model.om_bar = -7.2  # -7.2 (Changed to new peak of bleb curvature (omega))
        elif test_spec == 2:
            # Remove fourth-order term
            Model.l_k = 0
        elif test_spec <= 4:
            # Remove pressure dependence on cortex
            Model.finalT = 0.7
            if test_spec == 3:
                Model.p_c = Model.l_p
            else:
                Model.p_c = Model.l_l
            Model.l_p = 0
            Model.l_l = 0
        elif test_spec == 5:
            # Reduce rigidity
            Model.l_a *= 0.5


#######################################
# initialise arguments for compute
#######################################

# Run desired compute function, calculate time taken
start_time = time.time()
if not Model.errAnal:
    # Standard run with h=0.1, deltaT=0.0025
    # Conforming grid as described in paper
    domain = cartesianDomain([0, 0], [Model.lval, Model.bval], [int(Model.lval / Model.h), int(Model.bval / Model.h)])
    surfaceGrid = aluConformGrid(constructor=domain)
    # Run with params, grid
    compute(surfaceGrid, Model, weakSecList, outputFlag, oFileName, deltaVis)
else:
    # Error analysis
    # Complete error list
    compErrors = []
    # Discretisation values
    disc_vals = [[], []]
    # Define h, deltaT values
    disc_vals[0] = [0.8, 0.4, 0.2, 0.1, 0.05, 0.025]  # 0.8, 0.4, 0.2, 0.1, 0.05, 0.025
    disc_vals[1] = [(d**2)/4 for d in disc_vals[0]]

    for d in range(0, len(disc_vals[0])):
        # Extract h, deltaT values from list
        Model.h = disc_vals[0][d]
        Model.deltaT = disc_vals[1][d]
        # Define space for h value
        domain = cartesianDomain([0, 0], [Model.lval, Model.bval],
                                [int(Model.lval / Model.h), int(Model.bval / Model.h)])
        surfaceGrid = aluConformGrid(constructor=domain)

        # Append errors for given h, deltaT parameters
        compErrors.append(compute(surfaceGrid, Model, weakSecList, outputFlag, oFileName, deltaVis))
    # Compile h1, l2 errors
    h1e_stud = [compErrors[i][0] for i in range(0, len(disc_vals[0]))]
    l2e_stud = [compErrors[i][1] for i in range(0, len(disc_vals[0]))]
    # h, deltaT EOCs
    EOCs = [[[], []], [[], []]]
    for i in range(1, len(disc_vals[0])):
        for j in range(0, 2):
            # H1 EOCs (||grad (f_{h, deltaT} - f)||)
            h1 = np.log(h1e_stud[i] / h1e_stud[i-1]) / (np.log(disc_vals[j][i] / disc_vals[j][i-1]))
            EOCs[j][0].append(h1)
            # L2 EOCs (||f_{h, deltaT} - f)||)
            l2 = np.log(l2e_stud[i] / l2e_stud[i-1]) / (np.log(disc_vals[j][i] / disc_vals[j][i-1]))
            EOCs[j][1].append(l2)
    # print errors
    print("H1 errors:", h1e_stud, flush=True)
    print("L2 errors:", l2e_stud, flush=True)
    # Print EOCs
    print("EOCs (h) H1", EOCs[0][0], flush=True)
    print("EOCs (h) L2", EOCs[0][1], flush=True)
    print("EOCs (t) H1", EOCs[1][0], flush=True)
    print("EOCs (t) L2", EOCs[1][1], flush=True)
    # Small plot of errors (h) for visualisation
    plt.plot(disc_vals[0], h1e_stud)
    plt.plot(disc_vals[0], l2e_stud)
    plt.yscale("log")
    plt.xscale("log")
    plt.show()


# Check time taken for computation
# ~ 40 minute computation time with visualisation
print("--- %s seconds ---" % (time.time()-start_time), flush=True)


