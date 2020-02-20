# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:18:00 2020

@author: Tom
"""

import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level('INFO')

e_height = 0.25
plt.figure()
model_options = {
        "thermal": "x-lumped",
        "external submodels": ["thermal"],
    }
models = [pybamm.lithium_ion.SPM(model_options),
          pybamm.lithium_ion.SPMe(model_options),
          pybamm.lithium_ion.DFN(),
          ]
voltages = []
external_variables = {"X-averaged cell temperature": 300.0}
for model in models:
    geometry = model.default_geometry
    param = model.default_parameter_values
    param.update(
        {
            "Current function [A]": 1.0, 
            "Electrode height [m]": "[input]",
        }
    )
    param.process_model(model)
    param.process_geometry(geometry)
    inputs= {
            "Electrode height [m]": e_height,
            }
    A_cc = param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate(u=inputs)
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 10, var.r_p: 10}
    spatial_methods = model.default_spatial_methods
    
    solver = model.default_solver
    sim = pybamm.Simulation(
        model=model,
        geometry=geometry,
        parameter_values=param,
        var_pts=var_pts,
        spatial_methods=spatial_methods,
        solver=solver,
    )
#    t_eval=np.linspace(0, 60*60, 1000)
    for i in range(360):
        sim.step(dt=10, external_variables=external_variables, inputs=inputs)
    tv = sim.solution['Terminal voltage [V]']
    time = sim.solution['Time [min]']
    voltages.append(tv(sim.solution.t))
    plt.plot(time(sim.solution.t), tv(sim.solution.t),
             label=model.__class__.__name__)
plt.legend()
