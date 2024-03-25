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


def get_spiral_params(parameter_values, form_factor='18650', positive_tab=None, negative_tab=None):
    '''
    Generates the spiral parameters for the cell. This function calculates the number of layers and the length of the spiral for a given form factor and electrode parameters. The function also returns the project and arc_edges for the spiral net.

    Usage:
    ```python
        parameter_values = pybamm.ParameterValues("Chen2020")
        form_factor='18650'
        positive_tabs = {'position': [25,75], 'width': [0.006,0.006]}
        negative_tabs = {'position': [0,100], 'width': [0.003,0.004]}
    
        project, arc_edges = get_spiral_params(parameter_values, form_factor, positive_tabs, negative_tabs)
    ```

    Args:
        parameter_values (pybamm.ParameterValues): The base set of parameters to use for the cell (unit stack parameters)
        form_factor (str, optional): The desired form factor. Defaults to '18650'.
        positive_tabs (dict, optional): The parameters for the positive tabs. Dictionary must include 'position' and 'width'. For multiple tabs, supply dictionary values in a list. Defaults to None.
        negative_tabs (dict, optional): The parameters for the positive tabs. Dictionary must include 'position' and 'width'. For multiple tabs, supply dictionary values in a list. Defaults to None.

    Returns:
        tuple[openpnm.Project, openpnm.arc_edges]: The openpnm project and arc_edges for the spiral net
    '''
    spacing = get_unit_stack_thickness(parameter_values)
    if form_factor == '18650':
        inner_diameter = 0.0035
        outer_diameter = 0.018
        length_3d = 0.065
    elif form_factor == '21700':
        inner_diameter = 0.0035
        outer_diameter = 0.021
        length_3d = 0.070
    else:
        raise ValueError("form_factor must be '18650', '21700' or 'pouch'")
    
    # parameter_values.update({'Electrode height [m]': length3d})
    
    Nlayers, L = calculate_spiral(inner_diameter, outer_diameter, spacing)

    # nominal_area = parameter_values['Electrode height [m]'] * parameter_values['Electrode width [m]']
    # length_3d = 0.065
    dtheta = 15
    tesla_tabs = False

    # determine tab placement locations
    ncell_per_layer = 360/dtheta
    ncell_total = ncell_per_layer * Nlayers
    # find index for center of tab location
    arc_spacing = L/ncell_total

    if positive_tab == 'tesla' or negative_tab == 'tesla':
        tesla_tabs = True
        positive_tab = None
        negative_tab = None

    import math
    project, net_arc_edges = ecm.make_spiral_net(math.floor(Nlayers/2),
                                         dtheta,
                                         spacing,
                                         inner_diameter,
                                         [-1],
                                         [0],
                                         length_3d,
                                         tesla_tabs)

    arc_edges = np.diff(net_arc_edges)
    n_arc_edges = len(arc_edges)

    if positive_tab is not None and negative_tab is not None:
        tab_positions = positive_tab['position'], negative_tab['position']
        tab_widths = positive_tab['width'], negative_tab['width']
        tab_positions_idxs = [int((i/100)*n_arc_edges) for i in tab_positions[0]], [int((i/100)*n_arc_edges) for i in tab_positions[1]]
        tab_positions = []
        for e in range(2):
            for i, pos in enumerate(tab_positions_idxs[e]):
                # for the first tab, start at 0 and find length going from that boundary
                if pos == 0:
                    summed_tab_length_from_start = np.cumsum(arc_edges)
                    tab_cutoff = np.argwhere(summed_tab_length_from_start > tab_widths[e][i])[0][0]
                    actual_tab_length = summed_tab_length_from_start[tab_cutoff+1]
                    tab_positions.append(list(range(0, tab_cutoff+1)))
                elif pos == n_arc_edges:
                    summed_tab_length_from_end = np.cumsum(arc_edges[::-1])
                    tab_cutoff = np.argwhere(summed_tab_length_from_end > tab_widths[e][i])[0][0]
                    actual_tab_length = summed_tab_length_from_end[tab_cutoff+1]
                    tab_positions.append(list(range(-tab_cutoff-2, 0)))
                else:
                    tab_length_from_idx_to_end = arc_edges[pos:]
                    tab_length_from_idx_to_start = arc_edges[:pos][::-1]
                    shorter_summed_tab_length = min(len(tab_length_from_idx_to_end), len(tab_length_from_idx_to_start))
                    bidirectional_tab_length_from_idx = []
                    for j in range(shorter_summed_tab_length):
                        bidirectional_tab_length_from_idx.append(tab_length_from_idx_to_end[j])
                        bidirectional_tab_length_from_idx.append(tab_length_from_idx_to_start[j])
                    bidirectional_tab_length_from_idx = np.array(bidirectional_tab_length_from_idx)
                    bidirectional_summed_tab_length_from_idx = np.cumsum(bidirectional_tab_length_from_idx)

                    tab_cutoff = np.argwhere(bidirectional_summed_tab_length_from_idx > tab_widths[e][i])[0][0]
                    # check if tab_cutoff is even number
                    if tab_cutoff % 2 == 0:
                        tab_cutoff = tab_cutoff//2 
                        actual_tab_length = bidirectional_summed_tab_length_from_idx[tab_cutoff]
                        tab_positions.append(list(range(pos-tab_cutoff, pos+tab_cutoff+1)))
                    else:
                        tab_cutoff = (tab_cutoff)//2
                        actual_tab_length = bidirectional_summed_tab_length_from_idx[tab_cutoff]
                        tab_positions.append(list(range(pos-tab_cutoff, pos+tab_cutoff+1)))
        
        positive_tab_positions = tab_positions[0:2]
        positive_tab_positions = positive_tab_positions[0] + positive_tab_positions[1]
        negative_tab_positions = tab_positions[2:4]
        negative_tab_positions = negative_tab_positions[0] + negative_tab_positions[1]
        # negative_tab_positions = negative_tab_positions + len(arc_edges)

    elif positive_tab is not None or negative_tab is not None:
        raise ValueError("If one tab is specified, both must be specified")
    else:
        return project, net_arc_edges
    wrk = op.Workspace()
    wrk.clear()
    project, net_arc_edges = ecm.make_spiral_net(math.floor(Nlayers/2),
                                         dtheta,
                                         spacing,
                                         inner_diameter,
                                         positive_tab_positions,
                                         negative_tab_positions,
                                         length_3d,
                                         tesla_tabs)
        
    return project, net_arc_edges

def get_electrode_height(project):
    net = project.network
    return net["throat.electrode_height"][net.throats("throat.spm_resistor")].sum()

def estimate_nominal_capacity(project, parameter_values):
    electrode_height = get_electrode_height(project)
    actual_area = electrode_height * 0.065
    nominal_area = parameter_values['Electrode height [m]'] * parameter_values['Electrode width [m]']
    nominal_capacity = get_cell_areal_capacity(parameter_values, 'negative')
    areal_capacity = parameter_values["Nominal cell capacity [A.h]"] / nominal_area
    return areal_capacity * actual_area
    # return nominal_area * electrode_height * 1e3