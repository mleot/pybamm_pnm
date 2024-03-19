import openpnm as op
import matplotlib.pyplot as plt
import ecm
import liionpack as lp
import pybamm
import numpy as np
from typing import Literal


def get_unit_stack_thickness(parameters):
    electrode_thickness_p = parameters["Positive electrode thickness [m]"]
    electrode_thickness_n = parameters["Negative electrode thickness [m]"]
    cc_thickness_n = parameters["Negative current collector thickness [m]"] / 2
    cc_thickness_p = parameters["Positive current collector thickness [m]"] / 2 
    separator_thickness = parameters["Separator thickness [m]"]
    return electrode_thickness_p + electrode_thickness_n + cc_thickness_n + cc_thickness_p + separator_thickness


def get_estimated_diameter(spacing, Nlayers):
    return (((Nlayers*2) + 1.4)*spacing + 10*spacing ) * 2


import math

def calculate_spiral(inner_diameter, outer_diameter, layer_thickness):
    a = layer_thickness
    phi_1 = outer_diameter * math.pi / a
    phi_2 = inner_diameter * math.pi / a
    n = (phi_1 - phi_2) / (2 * math.pi)
    L = (
        (a/(2*math.pi)) 
        * (
            (phi_1/2)* (phi_1**2 + 1)**(1/2)
            + (1/2)*math.log(phi_1 + (phi_1**2 + 1)**(1/2))
            - (phi_2/2)* (phi_2**2 + 1)**(1/2)
            - (1/2)*math.log(phi_2 + (phi_2**2 + 1)**(1/2))
        )
    )
    return n, L

def get_cell_areal_capacity(parameters: dict, electrode:Literal['positive','negative','cell'],phase:Literal['','Primary: ','Secondary: ']|None=''):
    if phase is None:
        phase = ''
    elif phase == 'Primary:':
        phase = 'Primary: '
    elif phase == 'Secondary:':
        phase = 'Secondary: '
    if electrode == 'cell':
        electrode = 'positive'
    # if 'Primary: Maximum concentration in negative electrode [mol.m-3]' in parameters:
    positive_csmax = parameters[f'{phase}Maximum concentration in {electrode.lower()} electrode [mol.m-3]']
    positive_volfrac = parameters[f'{phase}{electrode.capitalize()} electrode active material volume fraction']
    positive_thickness = parameters[f'{electrode.capitalize()} electrode thickness [m]']
    positive_correction, negative_correction = get_SOC_window_correction_factor(pybamm.ParameterValues(parameters))
    if electrode == 'positive':
        correction = positive_correction
    elif electrode == 'negative':
        correction = negative_correction
    elif electrode == 'cell':
        correction = positive_correction
    areal_capacity = (
        (positive_csmax * positive_thickness * positive_volfrac * 96485)/
        (3.6 * 100**2)
        *(correction)
    )
    return areal_capacity

def get_SOC_window_correction_factor(params):
    '''
    Returns the correction factor for the SOC window of the cell. This is used to correct the cell capacity to account for the fact that the cell's SOC window is not 0-100% SOC. The correction factor is calculated by solving the electrode SOH model and finding the difference in the electrode capacity at 0% and 100% SOC. The correction factor is then used to scale the nominal capacity of the cell to account for the SOC window.

    _extended_summary_

    Args:
        params (pybamm.ParameterValues): The parameter set for the cell model

    Returns:
        tuple[float, float]: The positive, negative correction factors
    '''
    if isinstance(params,dict):
        params = pybamm.ParameterValues(params)
    param = pybamm.LithiumIonParameters()

    Vmin = params['Open-circuit voltage at 0% SOC [V]']
    Vmax = params['Open-circuit voltage at 100% SOC [V]']
    Q_n = params.evaluate(param.n.Q_init)
    Q_p = params.evaluate(param.p.Q_init)
    Q_Li = params.evaluate(param.Q_Li_particles_init)

    inputs = {"V_min": Vmin, "V_max": Vmax, "Q_n": Q_n, "Q_p": Q_p, "Q_Li": Q_Li}


    esoh_solver = pybamm.lithium_ion.ElectrodeSOHSolver(params, param)

    esoh_solver = esoh_solver.solve(inputs)

    correction_factor_p = esoh_solver['y_0'] - esoh_solver['y_100']
    correction_factor_n = esoh_solver['x_100'] - esoh_solver['x_0']

    return correction_factor_p, correction_factor_n


def get_spiral_params(parameter_values, form_factor='18650',tesla_tabs=False):
    spacing = get_unit_stack_thickness(parameter_values)
    print(f'Unit stack thickness: {spacing*1e6} um')
    if form_factor == '18650':
        inner_diameter = 0.0035
        outer_diameter = 0.018
    else:
        raise ValueError("form_factor must be '18650' or 'pouch'")
    
    Nlayers, L = calculate_spiral(inner_diameter, outer_diameter, spacing)
    print(Nlayers, L)
    # nominal_area = parameter_values['Electrode height [m]'] * parameter_values['Electrode width [m]']
    length_3d = 0.065
    dtheta = 15
    # tesla_tabs = False
    import math
    project, arc_edges = ecm.make_spiral_net(math.floor(Nlayers/2),
                                         dtheta,
                                         spacing,
                                         inner_diameter,
                                         [-1],
                                         [0],
                                         length_3d,
                                         tesla_tabs)
    
    return project, arc_edges

def get_electrode_height(project):
    net = project.network
    return net["throat.electrode_height"][net.throats("throat.spm_resistor")].sum()

def estimate_nominal_capacity(project, parameter_values):
    electrode_height = get_electrode_height(project)
    print(electrode_height, parameter_values['Electrode width [m]'])
    actual_area = electrode_height * 0.065
    nominal_area = parameter_values['Electrode height [m]'] * parameter_values['Electrode width [m]']
    print(actual_area, nominal_area)
    nominal_capacity = get_cell_areal_capacity(parameter_values, 'negative')
    areal_capacity = parameter_values["Nominal cell capacity [A.h]"] / nominal_area
    return areal_capacity * actual_area
    # return nominal_area * electrode_height * 1e3