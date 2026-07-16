import numpy as np
import Nodular_tumour_2 as sim
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed # multiprocessing
from tqdm import tqdm # progress bar

def parameters_generator(initial_params: list, file_name="parameters.txt"):
    parameters = {
        "O2_DIFFUSION_K": initial_params[0], "AGE_HEALTHY_ADULT": initial_params[1], "AGE_HEALTHY_G2": initial_params[2], "AGE_PROLIF_TUMOUR_ADULT": initial_params[3], "AGE_PROLIF_TUMOUR_G2": initial_params[4], "VOLUME_MCF10A_ml": initial_params[5], "VOLUME_MCF7_ml": initial_params[6],
        "FIELD_WIDTH": 512, "FIELD_HEIGHT": 512, "DT": 1.0, "MAX_TIME": 200,
        "DX": -1.0,
        "H_DIFFUSION_K": 1.10E-09,
        "O2_HEALTHY_LIVE_CONSUMPTION": -1.0,
        "H_HEALTHY_NECR_COUNT": -1.0,
        "H_HEALTHY_LIVE_CONSUMPTION": -1.0,
        "H_HEALTHY_DEATH_LIMIT": -1.0,
        "H_HEALTHY_APOPTOSIS_CONSUMPTION": 0,
        "O2_PROLIF_TUMOUR_LIVE_CONSUMPTION": -1.0,
        "H_PROLIF_TUMOUR_RELEASE_COUNT": -1.0,
        "H_PROLIF_TO_QUISC_LIMIT": -1.0,
        "O2_QUISC_TUMOUR_LIVE_CONSUMPTION": -1.0,
        "O2_QUISC_TUMOUR_PROLIF_LIMIT": -1.0,
        "H_QUISC_TO_PROLIF_LIMIT": -1.0,
        "H_QUISC_TUMOUR_RELEASE_COUNT": -1.0,
        "H_QUISC_TUMOUR_DEATH_LIMIT": -1.0,
        "H_QUISC_TUMOUR_NECR_COUNT": -1.0,
        "O2_SPAWN": -1.0,
        "O2_P_HEALTHY_mm_Hg": 65,
        "O2_P_TUMOUR_mm_Hg": 30,
        "O2_RHO_HEALTHY": -1.0,
        "O2_RHO_TUMOUR": -1.0,
        "O2_HEALTHY_LIVE_CONSUPTION_NOT_HOUR": 1.57E-04,
        "O2_TUMOUR_LIVE_CONSUPTION_NOT_HOUR": 1.57E-04,
        "CELL_VOLUME": -1.0,
        "CELL_HYPOXIA": 1.25E-05,
        "O2_M": 32,
        "CELL_SPHERE_VOLUME": -1.0,
        "MFC_7_AVE_N": 1.50E-10,
        "H_PROLIF_TUMOUR_GENERATION_SPEED": 5.2,
        "H_QUISC_TUMOUR_GENERATION_SPEED": 8.1,
        "H_REABSORTION_SPEED": 3.92E-01,
        "H_CELL_NORMAL_N": 5.01E-05,
        "H_INCELL_N": 6.30E-05,
        "VOLUME_MCF10A" : -1.0,
        "VOLUME_MCF7": -1.0,
        "R_HEALTHY": -1.0,
        "R_TUMOUR": -1.0
    }

    parameters['VOLUME_MCF10A'] = parameters['VOLUME_MCF10A_ml']/1.0e+6
    parameters['VOLUME_MCF7'] = parameters['VOLUME_MCF7_ml']/1.0e+6
    parameters['R_HEALTHY'] = (parameters['VOLUME_MCF10A']*3/(4*np.pi))**(1/3)
    parameters['R_TUMOUR'] = (parameters['VOLUME_MCF7']*3/(4*np.pi))**(1/3)
    parameters['DX'] = parameters['R_TUMOUR']*2

    parameters['O2_RHO_HEALTHY'] = parameters['O2_P_HEALTHY_mm_Hg'] * 4.29e-08
    parameters['O2_RHO_TUMOUR'] = parameters['O2_P_TUMOUR_mm_Hg'] * 4.29e-08

    parameters['CELL_VOLUME'] = parameters['DX'] * parameters['DX'] * parameters['DX']
    parameters['CELL_SPHERE_VOLUME'] = (4*np.pi)/(3*8) * parameters['CELL_VOLUME']

    parameters['O2_HEALTHY_LIVE_CONSUMPTION'] = parameters['O2_HEALTHY_LIVE_CONSUPTION_NOT_HOUR']*parameters['VOLUME_MCF10A_ml']*3600*parameters['O2_RHO_HEALTHY']/parameters['O2_M']
    parameters['H_HEALTHY_NECR_COUNT'] = parameters['VOLUME_MCF10A_ml']*parameters['H_INCELL_N']
    parameters['H_HEALTHY_LIVE_CONSUMPTION'] = parameters['CELL_VOLUME']*parameters['H_REABSORTION_SPEED']*parameters['H_CELL_NORMAL_N']
    parameters['H_HEALTHY_DEATH_LIMIT'] = 5.0118e-05 * parameters['CELL_VOLUME']
    parameters['O2_PROLIF_TUMOUR_LIVE_CONSUMPTION'] = parameters['O2_TUMOUR_LIVE_CONSUPTION_NOT_HOUR']*parameters['VOLUME_MCF7_ml']*3600*parameters['O2_RHO_TUMOUR']/parameters['O2_M']
    parameters['H_PROLIF_TUMOUR_RELEASE_COUNT'] = parameters['H_PROLIF_TUMOUR_GENERATION_SPEED'] * 1e-9 * 60 * parameters['MFC_7_AVE_N']
    parameters['H_PROLIF_TO_QUISC_LIMIT'] = 0.000398107 * parameters['CELL_VOLUME']
    parameters['O2_QUISC_TUMOUR_LIVE_CONSUMPTION'] = parameters['O2_PROLIF_TUMOUR_LIVE_CONSUMPTION'] / 2
    parameters['O2_QUISC_TUMOUR_PROLIF_LIMIT'] = parameters['CELL_VOLUME'] * parameters['CELL_HYPOXIA'] * 1e+3
    parameters['H_QUISC_TO_PROLIF_LIMIT'] = 0.000398107 * parameters['CELL_VOLUME']
    parameters['H_QUISC_TUMOUR_RELEASE_COUNT'] = parameters['H_QUISC_TUMOUR_GENERATION_SPEED'] * 1e-9 * 60 * parameters['MFC_7_AVE_N']
    parameters['H_QUISC_TUMOUR_DEATH_LIMIT'] = 0.000630957 * parameters['CELL_VOLUME']
    parameters['H_QUISC_TUMOUR_NECR_COUNT'] = parameters['VOLUME_MCF7_ml'] * parameters['H_INCELL_N']
    parameters['O2_SPAWN'] = parameters['CELL_VOLUME'] * 1e+6 * 18 * 8.04e-9

    result = {
        "FIELD_WIDTH": parameters['FIELD_WIDTH'],
        "FIELD_HEIGHT": parameters['FIELD_HEIGHT'],
        "DT": parameters['DT'],
        "MAX_TIME": parameters['MAX_TIME'],
        "DX": parameters['DX'],
        "O2_DIFFUSION_K": parameters['O2_DIFFUSION_K'],
        "H_DIFFUSION_K": parameters['H_DIFFUSION_K'],
        "O2_HEALTHY_LIVE_CONSUMPTION": parameters['O2_HEALTHY_LIVE_CONSUMPTION'],
        "H_HEALTHY_NECR_COUNT": parameters['H_HEALTHY_NECR_COUNT'],
        "H_HEALTHY_LIVE_CONSUMPTION": parameters['H_HEALTHY_LIVE_CONSUMPTION'],
        "H_HEALTHY_DEATH_LIMIT": parameters['H_HEALTHY_DEATH_LIMIT'],
        "H_HEALTHY_APOPTOSIS_CONSUMPTION": parameters['H_HEALTHY_APOPTOSIS_CONSUMPTION'],
        "AGE_HEALTHY_ADULT": parameters['AGE_HEALTHY_ADULT'],
        "AGE_HEALTHY_G2": parameters['AGE_HEALTHY_G2'],
        "O2_PROLIF_TUMOUR_LIVE_CONSUMPTION": parameters['O2_PROLIF_TUMOUR_LIVE_CONSUMPTION'],
        "H_PROLIF_TUMOUR_RELEASE_COUNT": parameters['H_PROLIF_TUMOUR_RELEASE_COUNT'],
        "H_PROLIF_TO_QUISC_LIMIT": parameters['H_PROLIF_TO_QUISC_LIMIT'],
        "AGE_PROLIF_TUMOUR_ADULT": parameters["AGE_PROLIF_TUMOUR_ADULT"],
        "AGE_PROLIF_TUMOUR_G2": parameters['AGE_PROLIF_TUMOUR_G2'],
        "O2_QUISC_TUMOUR_LIVE_CONSUMPTION": parameters['O2_QUISC_TUMOUR_LIVE_CONSUMPTION'],
        "O2_QUISC_TO_PROLIF_LIMIT": parameters['O2_QUISC_TUMOUR_PROLIF_LIMIT'],
        "H_QUISC_TO_PROLIF_LIMIT": parameters['H_QUISC_TO_PROLIF_LIMIT'],
        "H_QUISC_TUMOUR_RELEASE_COUNT": parameters['H_QUISC_TUMOUR_RELEASE_COUNT'],
        "H_QUISC_TUMOUR_DEATH_LIMIT": parameters['H_QUISC_TUMOUR_DEATH_LIMIT'],
        "H_QUISC_TUMOUR_NECR_COUNT": parameters['H_QUISC_TUMOUR_NECR_COUNT'],
        "O2_SPAWN": parameters['O2_SPAWN'],
    }
    with open(file_name, 'w', encoding='utf-8') as file:
        for name in result:
            print(name, result[name], file=file)
    return result

def make_parameters_files(list_parameters, folder, fname_prefix):
    result = []
    for i in range(len(list_parameters)):
        result.append(parameters_generator(list_parameters[i], folder+'/'+fname_prefix+str(i)+'.txt'))
    return result

def make_sim(parameters_fname, result_fname, seed=42):
    np.random.seed(seed)
    sim.import_parameters(parameters_fname)
    sim.fields['O2'][:] = 6 * sim.parameters['O2_HEALTHY_LIVE_CONSUMPTION'] * sim.parameters['DT']
    d = 5
    sim.fields['cells'][sim.parameters['FIELD_HEIGHT']//2-d:sim.parameters['FIELD_HEIGHT']//2+d, 
                    sim.parameters['FIELD_WIDTH']//2-d:sim.parameters['FIELD_WIDTH']//2+d] = 2 

    max_step = int(sim.parameters['MAX_TIME']/sim.parameters['DT'])
    for _ in range(max_step):
        sim.make_step(fields=sim.fields, params=sim.parameters)
    return sim.save_data(result_fname, sim.fields)

def run_single_sim(file_name, params_folder, result_folder):
    start_time = time.perf_counter()
    try:
        result = make_sim(params_folder+'/'+file_name, result_folder+'/result_'+file_name, seed=42)
        status = "ready"
    except Exception as e:
        result = e
        status = 'error'
    timer = time.perf_counter() - start_time
    return file_name, status, result, timer

def make_sims(params_folder, result_folder, max_workers=None):
    file_names = [f for f in os.listdir(params_folder) if os.path.isfile(os.path.join(params_folder, f)) and f.endswith('.txt')]
    nums_of_sim = len(file_names)
    file_to_idx = {name: idx for idx, name in enumerate(file_names)}
    results = [None] * nums_of_sim
    status_count = [0, 0]
    total_timer = -time.perf_counter()
    print(f"Generating {nums_of_sim} simulations.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        
        futures = {
            executor.submit(run_single_sim, fname, params_folder, result_folder): fname  for fname in file_names
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Симуляции"):
            fname, status, res, timer = future.result()
            status_count[status=='ready'] += 1
            idx = file_to_idx[fname]
            results[idx] = res

    total_timer += time.perf_counter()
    print(f"Total parallel time: {total_timer:.2f} s: successufly: {status_count[1]}, errors: {status_count[0]}")
    return results

            
if __name__ == '__main__':
    print(make_sims('parameters', 'simulation_results'))