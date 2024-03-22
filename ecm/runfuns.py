import pybamm
import numpy as np

import ecm
from ecm.autofit_jellyroll import get_spiral_params, get_cell_areal_capacity, estimate_nominal_capacity, get_electrode_height


import bpechem
from bpechem.shared import getParameterSpace
from bpechem.util import (
    get_cell_volume, set_cell_areal_capacity, get_cell_areal_capacity, 
    get_nominal_cell_capacity, get_cell_area, set_electrode_porosity,
    set_particle_volumetric_capacity,
)
from bpechem.util import get_cell_volume, set_cell_areal_capacity, get_cell_areal_capacity, get_nominal_cell_capacity, get_cell_area, set_initial_SOC, set_electrode_porosity, set_particle_volumetric_capacity
from ecm.autofit_jellyroll import get_unit_stack_thickness, calculate_spiral
import pandas as pd
import os
import mpire
from tqdm import tqdm



print_message = False

cpu_count = mpire.cpu_count()

def set_form_factor(row, params):
    if row['form factor'] == 'cylindrical':
        params['Cell cooling surface area [m2]'] = 0.00531
        params['Electrode height [m]'] = 0.065
        params['Electrode width [m]'] = 1.58
        params['Cell volume [m3]'] = 2.42e-05
    elif row['form factor'] == 'SPFC':
        params.update(SPFC_params,check_already_exists=False)
        params['Cell volume [m3]'] = get_cell_volume(params)
    else:
        raise ValueError(f"Unknown form factor: {row['form factor']}")

    params['Nominal cell capacity [A.h]'] = get_nominal_cell_capacity(params)
    return params

def set_areal_capacity(row, params):
    areal_capacity = row['areal capacity [mA.h.cm-2]']
    params = set_cell_areal_capacity(params, areal_capacity, "positive", True)
    return params

def run_simulation_cylindrical(i, row):

    # get parameter values
    parameter_values = pybamm.ParameterValues(row['parameter_set'])
    parameter_values = set_electrode_porosity(parameter_values,row['anode porosity'],'negative')
    parameter_values = set_particle_volumetric_capacity(parameter_values,row['particle capacity [mAh.cm-3]'],'negative','')
    parameter_values = set_areal_capacity(row, parameter_values)
    parameter_values.update(marquis_heat_transfer,check_already_exists=False)
    graphite_entropic_change = pybamm.ParameterValues('Ai2020')['Negative electrode OCP entropic change [V.K-1]']
    NMC_entropic_change = pybamm.ParameterValues('ORegan2022')['Positive electrode OCP entropic change [V.K-1]']
    parameter_values.update(
        {
            "Ambient temperature [K]": row['ambient temperature [C]']+273.15,
            "Initial temperature [K]": row['ambient temperature [C]']+273.15,
            'Upper voltage cut-off [V]': 4.8,
            'Lower voltage cut-off [V]': 2.0,
            'Negative electrode OCP entropic change [V.K-1]': graphite_entropic_change,
            'Positive electrode OCP entropic change [V.K-1]': NMC_entropic_change,
        }
    )
    parameter_values = pybamm.ParameterValues(parameter_values)

    if row['form_factor'] == '18650':
        inner_diameter = 0.0035
        outer_diameter = 0.018
    else:
        raise ValueError("form_factor must be '18650' or 'pouch'")

    project, arc_edges = get_spiral_params(parameter_values)

    # ecm.plot_topology(project.network)

    estimated_capacity = estimate_nominal_capacity_2(project, parameter_values)
    parameter_values = set_initial_SOC(parameter_values,row['SoC'])
    electrode_height = get_electrode_height(project)
    stack_thickness = get_unit_stack_thickness(parameter_values)
    Nlayers, L = calculate_spiral(inner_diameter, outer_diameter, stack_thickness)
    
    # get the experiment
    experiment = get_experiment(row,estimated_capacity)

    parameter_values.update({'Electrode width [m]': 0.065})

    trans_kwargs = {'t_slice':10,'t_precision':1}

    # Run simulation
    project, output = ecm.run_simulation_lp(parameter_values=parameter_values,
                                            experiment=experiment,
                                            initial_soc=None,
                                            project=project,
                                            **trans_kwargs)



    # return simulation
    return {
        'output':output,
        'project':project,
        'arc_edges':arc_edges,
        'rate':row['rate'],
        'commands':row,
        'estimated_capacity':estimated_capacity,
        'electrode height':electrode_height,
        'unit_stack_thickness':stack_thickness,
        'N layers':Nlayers
    }

# function to get the experiment
def get_experiment(row,capacity):
    if row['experiment_type'] == 'cycling CC':
        experiment = pybamm.Experiment(
            [
                (
                    f"Charge at {row['rate']}C until 4.2V ({10/(row['rate'])} s period)",
                    "Rest for 1 minutes (5 s period)",
                    f"Discharge at {row['rate']}C until 2.5V ({10/(row['rate'])} s period)",
                    "Rest for 1 minutes (5 s period)",
                )
            ] * row['cycles']
        )
    elif row['experiment_type'] == 'fast charge CCCV':
        experiment = pybamm.Experiment(
            [
                (
                    f"Charge at {row['rate']}C until 4.2V ({10/(row['rate'])} s period)",
                    "Hold at 4.2V until C/10",
                    "Rest for 1 minutes (5 s period)",
                    f"Discharge at 0.2C until 2.5V",
                    "Rest for 1 minutes (5 s period)",
                )
            ] * row['cycles']
        )
    elif row['experiment_type'] == 'fast discharge CCCV':
        experiment = pybamm.Experiment(
            [
                (
                    f"Charge at 0.2C until 4.2V",
                    "Hold at 4.2V until C/10",
                    "Rest for 1 minutes (5 s period)",
                    f"Discharge at {row['rate']}C until 2.5V ({10/(row['rate'])} s period)",
                    "Rest for 1 minutes (5 s period)",
                )
            ] * row['cycles']
        )
    elif row['experiment_type'] == 'cycling CCCV':
        experiment = pybamm.Experiment(
            [
                (
                    f"Charge at {row['rate']}C until 4.2V ({10/(row['rate'])} s period)",
                    "Hold at 4.2V until C/10",
                    "Rest for 1 minutes (5 s period)",
                    f"Discharge at {row['rate']}C until 2.5V ({10/(row['rate'])} s period)",
                    "Rest for 1 minutes (5 s period)",
                )
            ] * row['cycles']
        )
    elif row['experiment_type'] == 'Discharge':
        experiment = pybamm.Experiment(
            [
                ('Rest for 1 second'),
                (f"Discharge at {row['rate']*capacity} A for 10 seconds")
            ], period='0.5 seconds'
        )
    elif row['experiment_type'] == 'SOC DCR':
        experiment = pybamm.Experiment(
            [
                ('Discharge at 0.2 C for 30 minutes'),
                (f"Discharge at {row['rate']*capacity} A for 30 seconds")
            ], period='0.5 seconds'
        )
    elif row['experiment_type'] == 'Full Discharge':
        steps = 1/row['rate']*3600/360
        experiment = pybamm.Experiment(
            [
                (f"Discharge at {row['rate']*capacity} A for {1/row['rate']*3600} seconds")
            ], period=f'{steps} seconds'
        )
    elif row['experiment_type'] == 'Discharge DC':
        t = np.arange(0,11,0.001)
        i = np.zeros_like(t)
        # set i to 1 where t > 1
        i[t>1] = row['rate']*capacity

        
        experiment = pybamm.Interpolant(i,t,pybamm.t)

    elif row['experiment_type'] == 'Pulse':
        experiment = pybamm.Experiment(
                [
                    ("Rest for 1 second"),
                    (f"Discharge at {row['rate']*capacity} A for 10 seconds"),
                ], period='0.5 seconds'
        )
    elif row['experiment_type'] == 'Rest':
        experiment = pybamm.Experiment(
            [
                (f"Rest for 5 seconds (1 s period)"),
            ]
        )
    elif row['experiment_type'] == 'DCR':
        experiment = pybamm.Experiment(
            [
                # (f"Discharge at 0.067C for 20 hours or until 2.5 V (5 min period)", "Rest for 10 minutes (1 s period)"),
                # (f"Charge at 1C until 4.2V (1 s period)", "Hold at 4.2V until C/20 (10 s period)", "Rest for 6 minutes (1 s period)"),
                (f"Discharge at {0.2*capacity} A for 30 minutes (10 s period)", f"Discharge at {2*capacity} A for 30 seconds (1 s period)", f"Charge at {0.2*capacity} A for 5 minutes (1 s period)"),
                (f"Discharge at {0.2*capacity} A for 30 minutes (10 s period)", f"Discharge at {2*capacity} A for 30 seconds or until 2.5V (1 s period)", f"Charge at {0.2*capacity} A for 5 minutes (1 s period)"),
                (f"Discharge at {0.2*capacity} A for 30 minutes (10 s period)", f"Discharge at {2*capacity} A for 30 seconds or until 2.5V (1 s period)", f"Charge at {0.2*capacity} A for 5 minutes (1 s period)"),
                (f"Discharge at {0.2*capacity} A for 30 minutes (10 s period)", f"Discharge at {2*capacity} A for 30 seconds or until 2.5V (1 s period)", f"Charge at {0.2*capacity} A for 5 minutes (1 s period)"),
                (f"Discharge at {0.2*capacity} A for 30 minutes (10 s period)", f"Discharge at {2*capacity} A for 30 seconds or until 2.5V (1 s period)", f"Charge at {0.2*capacity} A for 5 minutes (1 s period)"),
                (f"Rest for 60 minutes (1 minute period)", f"Discharge at {1*capacity} A for 30 seconds or until 2.5V (1 s period)",f"Charge at {0.2*capacity} A for 2.5 minutes (1 s period)"),
                (f"Rest for 60 minutes (1 minute period)", f"Discharge at {2.3*capacity} A for 30 seconds or until 2.5V (1 s period)",f"Charge at {0.2*capacity} A for 5.75 minutes (1 s period)"),
                (f"Rest for 60 minutes (1 minute period)", f"Discharge at {3*capacity} A for 30 seconds or until 2.5V (1 s period)",f"Charge at {0.2*capacity} A for 7.5 minutes (1 s period)"),
                (f"Rest for 60 minutes (1 minute period)", f"Discharge at {5*capacity} A for 30 seconds or until 2.5V (1 s period)",f"Charge at {0.2*capacity} A for 12.5 minutes (1 s period)"),
                (f"Rest for 60 minutes (1 minute period)", f"Discharge at {10*capacity} A for 30 seconds or until 2.5V (1 s period)",f"Charge at {0.2*capacity} A for 25 minutes (1 s period)"),
                (f"Discharge at {0.2*capacity} A for 30 minutes (10 s period)", f"Discharge at {2*capacity} A for 30 seconds or until 2.5V (1 s period)", f"Charge at {0.2*capacity} A for 5 minutes (1 s period)"),
                (f"Discharge at {0.2*capacity} A for 30 minutes (10 s period)", f"Discharge at {2*capacity} A for 30 seconds or until 2.5V (1 s period)", f"Charge at {0.2*capacity} A for 5 minutes (1 s period)"),
                (f"Discharge at {0.2*capacity} A for 30 minutes (10 s period)", f"Discharge at {2*capacity} A for 30 seconds or until 2.5V (1 s period)", f"Charge at {0.2*capacity} A for 5 minutes (1 s period)"),
                (f"Discharge at {0.2*capacity} A for 30 minutes (10 s period)", f"Discharge at {2*capacity} A for 30 seconds or until 2.5V (1 s period)", f"Charge at {0.2*capacity} A for 5 minutes (1 s period)"),
                (f"Discharge at {0.2*capacity} A for 30 minutes (10 s period)", f"Discharge at {2*capacity} A for 30 seconds or until 2.5V (1 s period)", f"Charge at {0.2*capacity} A for 5 minutes (1 s period)"),
            ]
        )
    else:
        raise ValueError(f"Unknown experiment type: {row['experiment_type']}")
    return experiment

def estimate_nominal_capacity_2(project,parameter_values):
    params0 = parameter_values.copy()
    params0.update({'Electrode width [m]': 0.065})
    electrode_height = get_electrode_height(project)
    params0.update({'Electrode height [m]': electrode_height})
    model = pybamm.lithium_ion.SPM()
    experiment = pybamm.Experiment(
        [
            (
                "Discharge at 0.067C until 2.5V",
                "Rest for 1 minutes",
                "Charge at 0.067C until 4.2V",
                "Rest for 1 minutes",
            )
        ]
    )

    sim = pybamm.Simulation(model, parameter_values=params0, experiment=experiment,solver=pybamm.CasadiSolver('fast with events'))

    sol = sim.solve()

    return sol.summary_variables['Measured capacity [A.h]'][0]

def convert_output(output):
    var = output["Terminal voltage [V]"]
    import numpy as np
    xs,ys = var.shape

    cellids = np.zeros_like(var)
    times = np.zeros_like(var)
    voltages = np.zeros_like(var)
    currents = np.zeros_like(var)

    # dimension 0 is time, dimension 1 is cell number
    for i in range(xs):
        times[i,:] = output['Time [s]'][i]
        voltages[i,:] = output['Pack current [A]'][i]
        currents[i,:] = output['Pack terminal voltage [V]'][i]

    # looping through cell ids, and assigning value
    for i in range(ys):
        cellids[:,i] = i

    new_output = {}
    for key, value in output.items():
        new_output[key] = value.flatten()
        # print(len(new_output[key]))

    new_output['cellids'] = cellids.flatten()

    new_output['Time [s]'] = times.flatten()
    new_output['Pack current [A]'] = voltages.flatten()
    new_output['Pack terminal voltage [V]'] = currents.flatten()
    return pd.DataFrame.from_dict(new_output)


def normalize_output(output):
    var = output['X-averaged positive electrode temperature [C]']
    xs,ys = var.shape

    cellids = np.zeros_like(var)
    times = np.zeros_like(var)
    voltages = np.zeros_like(var)
    currents = np.zeros_like(var)

    # dimension 0 is time, dimension 1 is cell number
    for i in range(xs):
        times[i,:] = output['Time [s]'][i]
        voltages[i,:] = output['Pack current [A]'][i]
        currents[i,:] = output['Pack terminal voltage [V]'][i]

    output['Time [s]'] = times
    output['Pack current [A]'] = voltages
    output['Pack terminal voltage [V]'] = currents

    return output



def run_study(parameter_space):

    with tqdm(total=len(parameter_space)) as pbar:
        # iterate over rows
        for i, row in parameter_space.iterrows():
            output_dict = run_simulation(i, row)
            output = output_dict['output']
            if output is None:
                pbar.update(1)
                continue
            # save_sim(sim, i, row)
            output = convert_output(output)
            for k,v in output_dict.items():
                if k != 'output':
                    output[k] = v
            save_output(output, i, row)
            pbar.update(1)

output_variables = [
    'Time [h]',
    'Time [s]',
    'Discharge energy [W.h]',
    'Discharge capacity [A.h]',
    'Terminal voltage [V]',
    "Current [A]", 
    "Volume-averaged cell temperature [K]",
    "Volume-averaged cell temperature [C]",
    "Volume-averaged total heating [W.m-3]",
    "Battery negative particle concentration overpotential [V]",
    "Battery positive particle concentration overpotential [V]",
    "X-averaged battery negative reaction overpotential [V]",
    "X-averaged battery positive reaction overpotential [V]",
    "X-averaged battery concentration overpotential [V]",
    "X-averaged battery electrolyte ohmic losses [V]",
    "X-averaged battery negative solid phase ohmic losses [V]",
    "X-averaged battery positive solid phase ohmic losses [V]",
    "Battery particle concentration overpotential [V]",
    "X-averaged battery reaction overpotential [V]",
    "X-averaged battery solid phase ohmic losses [V]",
    "Battery open-circuit voltage [V]",
    "Battery negative electrode bulk open-circuit potential [V]",
    "Battery positive electrode bulk open-circuit potential [V]",
]

overpotential_vars = [
    "Battery negative particle concentration overpotential [V]",
    "Battery positive particle concentration overpotential [V]",
    "X-averaged battery negative reaction overpotential [V]",
    "X-averaged battery positive reaction overpotential [V]",
    "X-averaged battery concentration overpotential [V]",
    "X-averaged battery electrolyte ohmic losses [V]",
    "X-averaged battery negative solid phase ohmic losses [V]",
    "X-averaged battery positive solid phase ohmic losses [V]",
    "Battery particle concentration overpotential [V]",
    "X-averaged battery reaction overpotential [V]",
    "X-averaged battery solid phase ohmic losses [V]",
]

marquis_heat_transfer = {
    'Negative current collector surface heat transfer coefficient [W.m-2.K-1]':0.0,
    'Positive current collector surface heat transfer coefficient [W.m-2.K-1]':0.0,
    'Negative tab heat transfer coefficient [W.m-2.K-1]':10.0,
    'Positive tab heat transfer coefficient [W.m-2.K-1]':10.0,
    'Edge heat transfer coefficient [W.m-2.K-1]':0.3,
    # 'Negative tab centre z-coordinate [m]':0.137,
    # 'Positive tab centre z-coordinate [m]':0.137,
    # 'Negative tab centre y-coordinate [m]':0.06,
    # 'Positive tab centre y-coordinate [m]':0.147,
    # 'Negative tab width [m]':0.04,
    # 'Positive tab width [m]':0.04,
}

SPFC_params = {
    'Cell cooling surface area [m2]':0.00245,
    'Electrode height [m]':0.024,
    'Electrode width [m]':0.024,
    'Negative tab centre z-coordinate [m]':0.024,
    'Positive tab centre z-coordinate [m]':0.024,
    'Negative tab centre y-coordinate [m]':0.0072,
    'Positive tab centre y-coordinate [m]':0.0168,
    'Negative tab width [m]':0.0048,
    'Positive tab width [m]':0.0048,   
}