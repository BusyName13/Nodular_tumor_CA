import numpy as np

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
        "H_TUMOUR_DEATH_LIMIT": -1.0,
        "H_QUISC_TUMOUR_NECR_COUNT": -1.0,
        "O2_SPAWN": -1.0,
        "O2_P_HEALTHY_mm_Hg": 65,
        "O2_P_TUMOUR_mm_Hg": 30,
        "O2_RHO_HEALTHY": -1.0,
        "O2_RHO_TUMOUR": -1.0,
        "O2_HEALTHY_LIVE_CONSUPTION_NOT_HOUR": 1.57E-04,
        "O2_HEALTHY_LIVE_CONSUPTION_NOT_HOUR": 1.57E-04,
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

    parameters['O2_RHO_HEALTHY'] = parameters['O2_P_HEALTHY_mm_Hg'] * 0.0000000429
    parameters['O2_RHO_TUMOUR'] = parameters['O2_P_TUMOUR_mm_Hg'] * 0.0000000429

    parameters['CELL_VOLUME'] = parameters['DX'] * parameters['DX'] * parameters['DX']
    parameters['CELL_SPHERE_VOLUME'] = (4*np.pi)/(3*8) * parameters['CELL_VOLUME']
    

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
        "H_QUISC_TUMOUR_NECR_COUNT": parameters['H_QUISC_TUMOUR_NECR_COUNT'],
        "O2_SPAWN": parameters['O2_SPAWN'],
    }