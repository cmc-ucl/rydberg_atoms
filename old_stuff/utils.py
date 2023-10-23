import numpy as np
import matplotlib.pyplot as plt
from braket.ahs.atom_arrangement import SiteType, AtomArrangement
import json
from braket.timings.time_series import TimeSeries
from braket.ir.ahs.program_v1 import Program
from braket.task_result.analog_hamiltonian_simulation_task_result_v1 import AnalogHamiltonianSimulationTaskResult
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.driving_field import DrivingField
from braket.ahs.shifting_field import ShiftingField
from braket.ahs.field import Field
from braket.ahs.pattern import Pattern
import pandas as pd
from warnings import warn
from typing import Dict, List, Tuple
import boto3
import simplejson as sjson
from braket.aws import AwsQuantumTask

from braket.ahs.atom_arrangement import SiteType

def rabi_pulse(
    rabi_phase, 
    omega_max,
    omega_slew_rate_max,
    time_Delta_min
):
    """Construct a time series of amplitude with area equal to rabi_phase and satisfy the constriants
    
    Args:
        rabi_phase: The area of the time series
        omega_max: The maximum amplitude allowed
        omega_slew_rate_max: The maximum slew rate allowed
        time_Delta_min: The minimum time step allowed 
    

    """
    
    # Determine the ramp time to reach the max allowed rabi frequency omega_max
    t_ramp = max(time_Delta_min, omega_max/omega_slew_rate_max)
    
    # The max achievable rabi phase if there are 3 time points with 
    # times = [0, t_ramp, 2 * t_ramp]
    phase_threshold_1 = t_ramp * omega_max
    
    # The max achievable rabi phase if there are 4 time points with
    # times = [0, t_ramp, t_ramp + time_Delta_min, 2 * t_ramp + time_Delta_min]
    phase_threshold_2 = (t_ramp+time_Delta_min) * omega_max    
    
    if rabi_phase <= phase_threshold_1: 
        # Determine the amplitude if rabi_phase <= phase_threshold_1
        amplitude_value = rabi_phase/t_ramp 
        times = [0, t_ramp, 2 * t_ramp]
        amplitude_values = [0, amplitude_value, 0]
    elif rabi_phase <= phase_threshold_2:
        # Determine the amplitude if phase_threshold_1 < rabi_phase <= phase_threshold_2
        amplitude_value = rabi_phase/(t_ramp+time_Delta_min)
        times = [0, t_ramp, t_ramp + time_Delta_min, 2 * t_ramp + time_Delta_min]
        amplitude_values = [0, amplitude_value, amplitude_value, 0]
        
    else:
        # Determine the t_plateau if rabi_phase > phase_threshold_2
        t_plateau = (rabi_phase - phase_threshold_2)/omega_max + time_Delta_min
        times = [0, t_ramp, t_ramp + t_plateau, 2 * t_ramp + t_plateau]
        amplitude_values = [0, omega_max, omega_max, 0]

    # Do a self validation
    for val in amplitude_values:
        assert val <= omega_max
        
    assert (amplitude_values[1]-amplitude_values[0])/(times[1]-times[0]) <= omega_slew_rate_max
    
    for i in range(len(times)-1):
        assert (times[i+1] - times[i]) >= time_Delta_min
        
    if len(times)==3:
        assert (rabi_phase - amplitude_values[1]*times[1]) < 1e-9
    else:
        assert (rabi_phase - amplitude_values[1]*times[2]) < 1e-9
        
    return list(zip(times, amplitude_values))


## Two old version of rabi_pulse

# def rabi_pulse(
#     rabi_phase, 
#     omega_max,
#     omega_slew_rate_max,
#     time_Delta_min
# ):
#     """Construct a time series of amplitude with area equal to rabi_phase and satisfy the constriants
    
#     Args:
#         rabi_phase: The area of the time series
#         omega_max: The maximum amplitude allowed
#         omega_slew_rate_max: The maximum slew rate allowed
#         time_Delta_min: The minimum time step allowed 
    
#     Note: We will fix the ramp up time to be time_Delta_min for convenience.
#     """
    
#     t_ramp = time_Delta_min + 1e-9

#     # We first determine if we can reach omega_max during ramp up,
#     # followed by determining the maximum achievable rabi_phase if t_plateau = 0
#     if t_ramp * omega_slew_rate_max <= omega_max:
#         phase_threshold = time_Delta_min**2 * omega_slew_rate_max
#     else:
#         phase_threshold = omega_max**2 / omega_slew_rate_max
    
#     if rabi_phase <= phase_threshold: 
#         # If maximum achievable rabi_phase is larger than the desired rabi_phase
#         # Then we only need three time points, and we need to determine the 
#         # amplitude of the 2nd time point
        
#         amplitude_value = rabi_phase/t_ramp
#         times = [0, t_ramp, 2 * t_ramp]
#         amplitude_values = [0, amplitude_value, 0]
#     else:
#         # there are 4 time points, and need to make sure the time step
#         # between the 2nd and 3rd time points are larger than time_Delta_min
        
#         # First determine the amplitude reached by the ramp up
#         amplitude_value = phase_threshold/time_Delta_min
        
#         # Second determine the t_plateau needed to get the desired rabi_phase
#         t_plateau = (rabi_phase - phase_threshold)/amplitude_value
        
#         if t_plateau < time_Delta_min:
#             # need to have a lower value of amplitude_value, which can be 
#             # determined by setting t_plateau = t_ramp = time_Delta_min
#             amplitude_value = rabi_phase/(2*time_Delta_min)
#             t_plateau = time_Delta_min
            
#         times = [0, t_ramp, t_ramp + t_plateau, 2 * t_ramp + t_plateau]
#         amplitude_values = [0, amplitude_value, amplitude_value, 0]
    
#     print(times)
#     print(amplitude_values)
#     # Do a self validation
#     for val in amplitude_values:
#         assert val <= omega_max
        
#     assert (amplitude_values[1]-amplitude_values[0])/(times[1]-times[0]) <= omega_slew_rate_max
    
#     for i in range(len(times)-1):
#         assert (times[i+1] - times[i]) >= time_Delta_min
        
#     if len(times)==3:
#         assert (rabi_phase - amplitude_values[1]*times[1]) < 1e-9
#     else:
#         assert (rabi_phase - amplitude_values[1]*times[2]) < 1e-9
        
#     return list(zip(times, amplitude_values))
            
# def rabi_pulse(
#     rabi_phase, 
#     omega_max,
#     omega_slew_rate_max,
#     time_Delta_min
# ):    
#     phase_threshold = omega_max**2 / omega_slew_rate_max
#     if rabi_phase <= phase_threshold:
#         t_ramp = max(time_Delta_min, np.sqrt(rabi_phase / omega_slew_rate_max))
#         t_plateau = 0
#     else:
#         t_ramp = max(time_Delta_min, omega_max / omega_slew_rate_max)
#         t_plateau = (rabi_phase / omega_max) - t_ramp
        
#     t_pules = 2 * t_ramp + t_plateau
#     times = [0, t_ramp, t_ramp + t_plateau, t_pules]
#     amplitude_values = [0, t_ramp * omega_slew_rate_max, t_ramp * omega_slew_rate_max, 0]
    
#     return list(zip(times, amplitude_values))


def create_time_series(tv_pairs):
    ts = TimeSeries()
    for t,v in tv_pairs:
        ts.put(t, v)
    return ts


def constant_time_series(other_time_series: TimeSeries, constant: float=0.0) -> TimeSeries:
    """Obtain a constant time series with the same time points as the given time series
        Args:
            other_time_series (TimeSeries): The given time series
        Returns:
            TimeSeries: A constant time series with the same time points as the given time series
    """
    ts = TimeSeries()
    for t in other_time_series.times():
        ts.put(t, constant)
    return ts

def show_register(
    register: AtomArrangement, 
    blockade_radius: float=0.0, 
    what_to_draw: str="bond", 
    show_atom_index:bool=True
):
    """Plot the given register 
        Args:
            register (AtomArrangement): A given register
            blockade_radius (float): The blockade radius for the register. Default is 0
            what_to_draw (str): Either "bond" or "circle" to indicate the blockade region. 
                Default is "bond"
            show_atom_index (bool): Whether showing the indices of the atoms. Default is True
        
    """
    filled_sites = [site.coordinate for site in register if site.site_type == SiteType.FILLED]
    empty_sites = [site.coordinate for site in register if site.site_type == SiteType.VACANT]
    
    fig = plt.figure(figsize=(7, 7))
    if filled_sites:
        plt.plot(np.array(filled_sites)[:, 0], np.array(filled_sites)[:, 1], 'r.', ms=15, label='filled')
    if empty_sites:
        plt.plot(np.array(empty_sites)[:, 0], np.array(empty_sites)[:, 1], 'k.', ms=5, label='empty')
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    
    if show_atom_index:
        for idx, site in enumerate(register):
            plt.text(*site.coordinate, f"  {idx}", fontsize=12)
    
    if blockade_radius > 0 and what_to_draw=="bond":
        for i in range(len(filled_sites)):
            for j in range(i+1, len(filled_sites)):            
                dist = np.linalg.norm(np.array(filled_sites[i]) - np.array(filled_sites[j]))
                if dist <= blockade_radius:
                    plt.plot([filled_sites[i][0], filled_sites[j][0]], [filled_sites[i][1], filled_sites[j][1]], 'b')
                    
    if blockade_radius > 0 and what_to_draw=="circle":
        for site in filled_sites:
            plt.gca().add_patch( plt.Circle((site[0],site[1]), blockade_radius/2, color="b", alpha=0.3) )
        plt.gca().set_aspect(1)
    plt.show()   

def show_global_drive(drive, axes=None, **plot_ops):
    """Plot the driving field
        Args:
            drive (DrivingField): The driving field to be plot
            axes: matplotlib axis to draw on
            **plot_ops: options passed to matplitlib.pyplot.plot
    """   

    data = {
        'amplitude [rad/s]': drive.amplitude.time_series,
        'detuning [rad/s]': drive.detuning.time_series,
        'phase [rad]': drive.phase.time_series,
    }


    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)

    for ax, data_name in zip(axes, data.keys()):
        if data_name == 'phase [rad]':
            ax.step(data[data_name].times(), data[data_name].values(), '.-', where='post',**plot_ops)
        else:
            ax.plot(data[data_name].times(), data[data_name].values(), '.-',**plot_ops)
        ax.set_ylabel(data_name)
        ax.grid(ls=':')
    axes[-1].set_xlabel('time [s]')
    plt.tight_layout()
    plt.show()    
    

def show_drive_and_shift(drive: DrivingField, shift: ShiftingField):
    """Plot the driving and shifting fields
    
        Args:
            drive (DrivingField): The driving field to be plot
            shift (ShiftingField): The shifting field to be plot
    """        
    drive_data = {
        'amplitude [rad/s]': drive.amplitude.time_series,
        'detuning [rad/s]': drive.detuning.time_series,
        'phase [rad]': drive.phase.time_series,
    }
    
    fig, axes = plt.subplots(4, 1, figsize=(7, 7), sharex=True)
    for ax, data_name in zip(axes, drive_data.keys()):
        if data_name == 'phase [rad]':
            ax.step(drive_data[data_name].times(), drive_data[data_name].values(), '.-', where='post')
        else:
            ax.plot(drive_data[data_name].times(), drive_data[data_name].values(), '.-')
        ax.set_ylabel(data_name)
        ax.grid(ls=':')
        
    shift_data = shift.magnitude.time_series
    pattern = shift.magnitude.pattern.series
    pattern = [float(i) for i in pattern]
    axes[-1].plot(shift_data.times(), shift_data.values(), '.-', label="pattern: " + str(pattern))
    axes[-1].set_ylabel('shift [rad/s]')
    axes[-1].set_xlabel('time [s]')
    axes[-1].legend()
    axes[-1].grid()
    plt.tight_layout()
    plt.show()
        
    
def get_shift(times: List[float], values: List[float], pattern: List[float]) -> ShiftingField:
    """Get the shifting field from a set of time points, values and pattern
        Args:
            times (List[float]): The time points of the shifting field
            values (List[float]): The values of the shifting field
            pattern (List[float]): The pattern of the shifting field
        Returns:
            ShiftingField: The shifting field obtained
    """    
    assert len(times) == len(values)    
    
    magnitude = TimeSeries()
    for t, v in zip(times, values):
        magnitude.put(t, v)
    shift = ShiftingField(Field(magnitude, Pattern(pattern)))

    return shift    

# def concatenate_time_series(time_series_1: TimeSeries, time_series_2: TimeSeries, check: bool = True) -> TimeSeries:
#     """Concatenate two time series to a single time series
#         Args:
#             time_series_1 (TimeSeries): The first time series to be concatenated
#             time_series_2 (TimeSeries): The second time series to be concatenated
#         Returns:
#             TimeSeries: The concatenated time series
#     """
#     if check:
#         assert time_series_1.values()[-1] == time_series_2.values()[0]
    
#     duration_1 = time_series_1.times()[-1] - time_series_1.times()[0]
    
#     new_time_series = TimeSeries()
#     new_times = time_series_1.times() + [t + duration_1 - time_series_2.times()[0] for t in time_series_2.times()[1:]]
#     new_values = time_series_1.values() + time_series_2.values()[1:]
#     for t, v in zip(new_times, new_values):
#         new_time_series.put(t, v)
    
#     return new_time_series


# def concatenate_drives(drive_1: DrivingField, drive_2: DrivingField) -> DrivingField:
#     """Concatenate two driving fields to a single driving field
#         Args:
#             drive_1 (DrivingField): The first driving field to be concatenated
#             drive_2 (DrivingField): The second driving field to be concatenated
#         Returns:
#             DrivingField: The concatenated driving field
#     """    
#     return DrivingField(
#         amplitude=concatenate_time_series(drive_1.amplitude.time_series, drive_2.amplitude.time_series),
#         detuning=concatenate_time_series(drive_1.detuning.time_series, drive_2.detuning.time_series),
#         phase=concatenate_time_series(drive_1.phase.time_series, drive_2.phase.time_series, check=False)
#     ) 

def concatenate_drives(drive_1, drive_2):
    assert drive_1.amplitude.time_series.values()[-1] == drive_2.amplitude.time_series.values()[0]
    assert drive_1.detuning.time_series.values()[-1] == drive_2.detuning.time_series.values()[0]
    # assert drive_1.phase.time_series.values()[-1] == drive_2.phase.time_series.values()[0]

    duration_1 = drive_1.amplitude.time_series.times()[-1]
    
    new_amplitude = TimeSeries()
    for t, v in zip(drive_1.amplitude.time_series.times(), drive_1.amplitude.time_series.values()):
        new_amplitude.put(t, v)
    for t, v in zip(drive_2.amplitude.time_series.times(), drive_2.amplitude.time_series.values()):
        new_amplitude.put(t + duration_1, v)
        
    new_detuning = TimeSeries()
    for t, v in zip(drive_1.detuning.time_series.times(), drive_1.detuning.time_series.values()):
        new_detuning.put(t, v)
    for t, v in zip(drive_2.detuning.time_series.times(), drive_2.detuning.time_series.values()):
        new_detuning.put(t + duration_1, v)
        
    new_phase = TimeSeries()
    for t, v in zip(drive_1.phase.time_series.times(), drive_1.phase.time_series.values()):
        new_phase.put(t, v)
    for t, v in zip(drive_2.phase.time_series.times(), drive_2.phase.time_series.values()):
        new_phase.put(t + duration_1, v)
        
    return DrivingField(
        amplitude=new_amplitude,
        detuning=new_detuning,
        phase=new_phase
    )
    

def concatenate_drive_list(drive_list: List[DrivingField]) -> DrivingField:
    """Concatenate a list of driving fields to a single driving field
        Args:
            drive_list (List[DrivingField]): The list of driving fields to be concatenated
        Returns:
            DrivingField: The concatenated driving field
    """        
    drive = drive_list[0]
    for dr in drive_list[1:]:
        drive = concatenate_drives(drive, dr)
    return drive    
    
def get_parallel_registers(
    register: AtomArrangement, 
    inter_register_distance: float, 
    width: float, 
    height: float, 
    n_site_max: int = 256
) -> Tuple[AtomArrangement, Dict[Tuple[int, int], List[int]]]:
    """Get parallel registers that fit the QPU area
        Args:
            register (AtomArrangement): A single instance of register
            inter_register_distance (float): The distance between registers
            width (float): The width of the area
            height (float): The height of the area
            n_site_max (int): The maximum number of sites
        Returns:
            AtomArrangement: The parallel registers that fit the QPU area
            Dict[Tuple[int, int], List[int]]: Dictionary to keep track the indices 
                of the atoms in the batch
    """      
    
    x_min = min(*[site.coordinate[0] for site in register])
    x_max = max(*[site.coordinate[0] for site in register])
    y_min = min(*[site.coordinate[1] for site in register])
    y_max = max(*[site.coordinate[1] for site in register])

    single_problem_width = x_max - x_min
    single_problem_height = y_max - y_min

    # setting up a grid of problems filling the total area
    n_width = int(width   / (single_problem_width  + inter_register_distance))
    n_height = int(height / (single_problem_height + inter_register_distance))
    
    batch_mapping = dict()
    parallel_register = AtomArrangement()

    atom_number = 0 #counting number of atoms added

    for ix in range(n_width+1):
        x_shift = ix * (single_problem_width   + inter_register_distance)

        for iy in range(n_height+1):    
            y_shift = iy * (single_problem_height  + inter_register_distance)

            # reached the maximum number of batches possible given n_site_max
            if atom_number + len(register) > n_site_max: break 

            atoms = []
            for site in register:
                new_coordinate = (x_shift + site.coordinate[0], y_shift + site.coordinate[1])
                parallel_register.add(new_coordinate, site.site_type)
                atoms.append(atom_number)
                atom_number += 1

            batch_mapping[(ix,iy)] = atoms
            
    return parallel_register, batch_mapping



def save_result(results, result_type):
    for key, val in results.items():
        with open(f'./results/{result_type}/{key}_result.json', 'w') as fw:
            fw.write(sjson.dumps(val.dict(), use_decimal=True))

            
def get_atom_arrangement_1D(lim, spacing=2.499e-5, orientation='y'):
    register = AtomArrangement()
    num_atoms = int(lim/spacing)
    for i in range(num_atoms+1):
        if orientation == 'x':
            register.add((spacing * i, 0))
        else:
            register.add((0, spacing * i))            
    return register

def get_atom_arrangement_2D(x_lim, y_lim, x_spacing=2.499e-5, y_spacing=2.499e-5):
    register = AtomArrangement()
    num_atoms_x = int(x_lim/x_spacing)
    num_atoms_y = int(y_lim/y_spacing)    
    for i in range(num_atoms_x+1):
        for j in range(num_atoms_y+1):        
            register.add((x_spacing * i, y_spacing * j))
            
    # Add empty sites
    for i in range(num_atoms_x):
        for j in range(num_atoms_y):
            register.add((x_spacing * (i+1/2), y_spacing * (j+1/2)), SiteType.VACANT)
    
    return register

def get_atom_arrangement_7_atom_triangular_lattice(separation=5.05e-6):
    register = AtomArrangement()
    
    register.add((0.0, 0.0))
    
    register.add((separation, 0.0), SiteType.VACANT)
    register.add((separation * 1/2, separation * np.sqrt(3)/2), SiteType.VACANT)
    register.add((-separation * 1/2, separation * np.sqrt(3)/2), SiteType.VACANT)
    register.add((-separation, 0.0), SiteType.VACANT)
    register.add((separation * 1/2, -separation * np.sqrt(3)/2), SiteType.VACANT)
    register.add((-separation * 1/2, -separation * np.sqrt(3)/2), SiteType.VACANT)    
    
    return register


def get_local_simulator_result():
    
    with open(f'./results/Local_simulator/result.json', 'r') as fr:
        results_json = json.load(fr)    
    
    results = {}
    for key, val in results_json.items():
        results[key] = AnalogHamiltonianSimulationTaskResult.parse_raw(val)
    
    return results

def get_avg_density(result):
    measurements = result.measurements
    postSeqs = [measurement.shotResult.postSequence for measurement in measurements]
    postSeqs = 1 - np.array(postSeqs) # change the notation such 1 for rydberg state, and 0 for ground state
    
    avg_density = np.sum(postSeqs, axis=0)/len(postSeqs)
    
    return avg_density


def results_to_df(result):
    """Aggregate state counts from AHS shot results
        A count of strings (of length = # of atoms) are returned, where
        each character denotes the state of an atom (site):
            e: empty atom
            r: Rydberg state atom
            g: ground state spin
    
        Args:
            result (dict)
        Returns:
            dict: number of times each state configuration is measured
    """

    task_status_code = result['taskMetadata']['status']
    if task_status_code is not None and task_status_code != "COMPLETED":
        warn(f'status code is not `COMPLETED` ({task_status_code}).')
    data = {} 
    states = ['e', 'r', 'g']
    for shot_idx, shot in enumerate(result['measurements']):
        shot_status_code = shot['shotMetadata']['shotStatus']
        if shot_status_code != 'Success':
            warn(f'shot {shot_idx} status code is not `Success` ({shot_status_code}). Skipping.')
            continue
        
        if not data:
            data = dict([(str(idx), []) for idx in range(len(shot['shotResult']['preSequence']))])
        
        pre = shot['shotResult']['preSequence']
        post = shot['shotResult']['postSequence']
        state_idx = np.array(pre) * (1 + np.array(post))
        for idx, s in enumerate(map(lambda s_idx: states[s_idx], state_idx)):
            data[str(idx)].append(s)
    return pd.DataFrame(data)  



    # if result_type == "Aquila":
    #     task_status_code = result['taskMetadata']['status']
    #     if task_status_code != "COMPLETED":
    #         warn(f'status code is not `COMPLETED` ({task_status_code}).')
    #     data = {} 
    #     states = ['e', 'r', 'g']
    #     for shot_idx, shot in enumerate(result['measurements']):
    #         shot_status_code = shot['shotMetadata']['shotStatus']
    #         if shot_status_code != 'Success':
    #             warn(f'shot {shot_idx} status code is not `Success` ({shot_status_code}). Skipping.')
    #             continue
            
    #         if not data:
    #             data = dict([(str(idx), []) for idx in range(len(shot['shotResult']['preSequence']))])
            
    #         pre = shot['shotResult']['preSequence']
    #         post = shot['shotResult']['postSequence']
    #         state_idx = np.array(pre) * (1 + np.array(post))
    #         for idx, s in enumerate(map(lambda s_idx: states[s_idx], state_idx)):
    #             data[str(idx)].append(s)
    #     return pd.DataFrame(data)        
    
    
    # else: # "Local_simulator"
    #     measurement_result = result.measurements
    #     data = dict([(str(idx), []) for idx in range(len(measurement_result[0].shotResult.preSequence))])
    #     states = ['e', 'r', 'g']
    #     for shot in measurement_result:
    #         pre = shot.shotResult.preSequence
    #         post = shot.shotResult.postSequence
    #         state_idx = np.array(pre) * (1 + np.array(post))
    #         for idx, s in enumerate(map(lambda s_idx: states[s_idx], state_idx)):
    #             data[str(idx)].append(s)
    #     return pd.DataFrame(data)
    

def create_task(
    qpu, 
    shots,
    register,
    amplitude,
    detuning=None,
    phase=None
):
    amplitude = create_time_series(amplitude)
    if detuning is None:
        detuning = constant_time_series(amplitude, 0.0)
    if phase is None:
        phase = constant_time_series(amplitude, 0.0)
    program = AnalogHamiltonianSimulation(
        register=register,
        hamiltonian=DrivingField(
            amplitude=amplitude,
            detuning=detuning,
            phase=phase
        )
    )
    program = program.discretize(qpu)
    return {
        'shots': shots,
        'program': json.loads(program.to_ir().json()),
    }



def save_task(name, task):
    with open(f'./tasks/{name}.json', 'w') as fw:
        fw.write(json.dumps(task))


def submit_task_to_prod(
    task, 
    braket_client = boto3.client("braket"),
    DEVICE_ARN = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
    S3_BUCKET_NAME = "amazon-braket-quera-benchmarking-maolinml",
    S3_BUCKET_PREFIX = "quera-test",
    dryRun=False
):
    shots = task['shots']
    program = Program(**task['program'])    
    action = {
        'braketSchemaHeader': {
            'name': 'braket.ir.ahs.program',
            'version': '1'
        },
    }
    action.update(program.dict())
    response = braket_client.create_quantum_task(
        dryRun=dryRun,
        deviceArn=DEVICE_ARN,
        shots=shots,
        outputS3Bucket=S3_BUCKET_NAME,
        outputS3KeyPrefix=S3_BUCKET_PREFIX,
        action=sjson.dumps(action, use_decimal=True),
    )
    return response

def submit_task(
    task, 
    braket_client = boto3.client("braket", endpoint_url="https://braket-gamma.us-east-1.amazonaws.com", region_name="us-east-1"),
    DEVICE_ARN = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
    S3_BUCKET_NAME = "amazon-braket-quera-benchmarking-maolinml",
    S3_BUCKET_PREFIX = "quera-test",
    dryRun=False
):
    shots = task['shots']
    program = Program(**task['program'])    
    action = {
        'braketSchemaHeader': {
            'name': 'braket.ir.ahs.program',
            'version': '1'
        },
    }
    action.update(program.dict())
    response = braket_client.create_quantum_task(
        dryRun=dryRun,
        deviceArn=DEVICE_ARN,
        shots=shots,
        outputS3Bucket=S3_BUCKET_NAME,
        outputS3KeyPrefix=S3_BUCKET_PREFIX,
        action=sjson.dumps(action, use_decimal=True)
    )
    return response
    

# Get back the results from S3 as a json file

def get_result(
    task_arn: str,
    braket_client = boto3.client("braket")
) -> AnalogHamiltonianSimulationTaskResult:
    """Get result of a task.
    
    Args:
        task_arn (str): Amazon Resource Number of the quantum task
    
    Returns:
        AnalogHamiltonianSimulationTaskResult: result
    """
    
    task_response = braket_client.get_quantum_task(quantumTaskArn=task_arn)
    assert task_response['status'] == 'COMPLETED'
    aws_session = AwsQuantumTask._aws_session_for_task_arn(task_arn=task_arn)
    result_string = aws_session.retrieve_s3_object_body(
            task_response["outputS3Bucket"],
            task_response["outputS3Directory"] + f"/{AwsQuantumTask.RESULTS_FILENAME}",
        )
    result = AnalogHamiltonianSimulationTaskResult.parse_raw(result_string)
    return result    