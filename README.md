# Inverse blebbing computation
This code runs a Uzawa-type solver through use of a Schur compliment for a mixed fourth-order PDE problem. Explicit details on the model will be provided at a later date.
This code is generated for a masters project. Once the report is completed and submitted it will be included for ease of understanding, along with any animations or images generated. <br>
To run various tests or simulations detailed in this code, alterations must be made to `test_set`, `test_spec` in `main.py`.
To run custom simulations with `compInvB.compute()`, a `Param` object must be created with the requisite attributes, along with a list of `WeakSecParam` objects. Examples of such construction and alteration may be found in `main.py`.

## How It's Made:

**Tech used:** Python, DUNE-FEM 2.9.0.2

The code was adapted from the [blebbing compute](https://gitlab.dune-project.org/bjorn.stinner/sfem_blebs/-/blob/master/blebbing_compute.py?ref_type=heads) created by Andreas Dedner, Bjormn Stinner, and Adam Nixon.
Using [Dune-Fem](http://link.springer.com/article/10.1007/s00607-010-0110-3/), it runs a first-order IMEX time splitting method on a mixed-space fourth order problem. This uses operator splitting to form a Uzawa solver for the given system.


## Optimizations

This code has been structured so as to allow changes in tests and deformations with minimal user input, along with minor improvements made to calculation functions so as to reduce time taken for computation (most explicitly in characteristic functions).
Further optimizations are planned for the writing of VTK files, as these take the bulk of time over extended test sessions.
 
