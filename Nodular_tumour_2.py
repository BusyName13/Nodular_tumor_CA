import numpy as np
import scipy.fft as sfft
import numba # taichi
import pygame
import graphic as g
import colors as col
import time

### Classes
class FFT_Diffusion:
    def __init__(self, size, dx, dt, D_K):
        self.H, self.W = size
        self.dx = dx
        self.dt = dt
        self.D_K = D_K
        # self.fft_buffer = np.empty((self.H, self.W // 2 + 1), dtype=np.complex128)

        # Find kernel
        kx = 2 * np.pi * sfft.rfftfreq(self.W, d=dx)
        ky = 2 * np.pi * sfft.fftfreq(self.H, d=dx)
        kx2 = kx[None, :]**2
        ky2 = ky[:, None]**2
        k2 = kx2 + ky2
        self.kernel = np.exp(-D_K * k2 * dt)

    def diffuse(self, field: np.ndarray) -> np.ndarray:
        field_hat = sfft.rfft2(field, workers=-1, overwrite_x=True)
        field_hat *= self.kernel
        return sfft.irfft2(field_hat, s=field.shape, workers=-1, overwrite_x=True)

### Consts
cell_cols = np.array([
    0, #Empty_cell
    col.col_to_num((240, 160, 117)), # Healthy_cell
    col.col_to_num((242, 39, 39)), # Proliferating tumour cell
    col.col_to_num((102, 17, 17)), # Quiscent tumour cell
])

parameters = {'FIELD_WIDTH': 1000, 'FIELD_HEIGHT': 750, 'FIELD_SIZE': (-1, -1),
              'DT': 1.0, 'DT_S': 3600.0, 'DX': 3e-05,
              'O2_DIFFUSION_K': 2.41e-9, 'H_DIFFUSION_K': 1.1e-9, 
              'O2_HEALTHY_LIVE_CONSUMPTION': -1.0, 'O2_HEALTHY_MITOSIS_CONSUMPTION': -1.0, 'O2_HEALTHY_HYPOXIA_LIMIT': -1.0, 'H_HEALTHY_NECR_COUNT': -1.0, 'H_HEALTHY_LIVE_CONSUMPTION': -1.0, 'H_HEALTHY_DEATH_LIMIT': -1.0, 'H_HEALTHY_APOPTOSIS_CONSUMPTION': -1.0, 'AGE_HEALTHY_MITOSIS': -1.0, 
              'O2_PROLIF_TUMOUR_LIVE_CONSUMPTION': -1.0, 'O2_PROLIF_TUMOUR_MITOSIS_CONSUMPTION': -1.0, 'O2_PROLIF_TUMOUR_QUISC_LIMIT': -1.0, 'H_PROLIF_TUMOUR_RELEASE_COUNT': -1.0, 'AGE_PROLIF_TUMOUR_MITOSIS': -1.0, 
              'O2_QUISC_TUMOUR_LIVE_CONSUMPTION': -1.0, 'O2_QUISC_TUMOUR_PROLIF_LIMIT': -1.0, 'O2_QUISC_TUMOUR_NECR_LIMIT': -1.0, 'H_QUISC_TUMOUR_RELEASE_COUNT': -1.0}
parameters["FIELD_SIZE"] = (parameters['FIELD_HEIGHT'], parameters['FIELD_WIDTH'])
parameters["DT_S"] = 3600 * parameters['DT']
parameters["O2_LIVE_CONSUMPTION"] = np.array([0.0, parameters['O2_HEALTHY_LIVE_CONSUMPTION'], parameters['O2_PROLIF_TUMOUR_LIVE_CONSUMPTION'], parameters['O2_QUISC_TUMOUR_LIVE_CONSUMPTION']])
def import_parameters(file_name: str, params=parameters):
    dict = {}
    with open(file_name, 'r', encoding='utf-8') as f:
        # for _ in range(2):
        #     key, val = f.readline().split()
        #     dict[key] = int(val)
        dict.update({key: float(val) if (('.' in val) or ('e' in val)) else int(val) for line in f for key, val in [line.split()]})
        params.update(dict)
        params['FIELD_SIZE'] = (params['FIELD_HEIGHT'], params['FIELD_WIDTH'])
        params['DT_S'] = 3600 * params['DT']
        params["O2_LIVE_CONSUMPTION"] = np.array([0.0, params['O2_HEALTHY_LIVE_CONSUMPTION'], params['O2_PROLIF_TUMOUR_LIVE_CONSUMPTION'], params['O2_QUISC_TUMOUR_LIVE_CONSUMPTION']])
    return dict
import_parameters("parameters.txt", parameters)
fields =  {"cells":  np.ones(parameters['FIELD_SIZE'], dtype=np.uint8),
           "O2":     np.zeros(parameters['FIELD_SIZE'], dtype=np.float64), 
           "O2_dif": FFT_Diffusion(parameters['FIELD_SIZE'], parameters['DX'], parameters['DT_S'], parameters['O2_DIFFUSION_K']),
           "H":      np.zeros(parameters['FIELD_SIZE'], dtype=np.float64),
           "H_dif":  FFT_Diffusion(parameters['FIELD_SIZE'], parameters['DX'], parameters['DT_S'], parameters['H_DIFFUSION_K']),
           "age":    parameters['DT'] * np.random.randint(0, 100, parameters['FIELD_SIZE'], dtype=np.uint64)%parameters['AGE_HEALTHY_MITOSIS'],
           "G2":     np.zeros(parameters['FIELD_SIZE'], dtype=np.uint8)
           }

def save_data(file_name, fields, format="%.18e", filter=None):
    np.savetxt(file_name, fields['cells'], fmt=format)


### Functions

# Math
@numba.njit(parallel=True, fastmath=True)
def conv2d_calc(field, kernel, pad_field): 
    k_h, k_w = kernel.shape 
    f_h, f_w = field.shape 
    c_i, c_j = k_h//2, k_w//2 

    pad_field[ c_i:-c_i, c_j:-c_j] = field 
    pad_field[ :c_i, c_j:-c_j] = field[-c_i:, :] 
    pad_field[-c_i:, c_j:-c_j] = field[ :c_i, :] 
    pad_field[ :, :c_j] = pad_field[:, -2*c_j:-c_j] 
    pad_field[ :, -c_j:] = pad_field[:, c_j:2*c_j] 

    res = np.zeros_like(field) 
    for f_i in numba.prange(f_h): 
        for k_i in range(k_h): 
            for k_j in range(k_w): 
                for f_j in range(f_w): 
                    res[f_i, f_j] += pad_field[f_i+k_i, f_j+k_j] * kernel[k_i, k_j] 
    return res

cache = {}
def conv2d(field, kernel):
    k_h, k_w = kernel.shape
    f_h, f_w =  field.shape
    c_i, c_j = k_h//2, k_w//2
    if (f_h, f_w, k_h, k_w) in cache:
        pad_field, res = cache[(f_h, f_w, k_h, k_w)]
    else:
        pad_field = np.empty((f_h+2*c_i, f_w+2*c_j), dtype=field.dtype)  
        res = np.zeros_like(field)
        cache[(f_h, f_w, k_h, k_w)] = (pad_field, res)
    return conv2d_calc(field, kernel, pad_field)
# _ = conv2d(np.ones((3, 3), dtype=np.float64), np.ones((3, 3), dtype=np.float64))
# _ = conv2d(np.ones((3, 3), dtype=np.int64), np.ones((3, 3), dtype=np.uint8))

# Graphic
render_arr = np.zeros((parameters['FIELD_HEIGHT'], parameters['FIELD_WIDTH'], 4), dtype=np.uint8) # BGRA
render_arr[..., 3] = 255
def render_field(surface, fields, cols, filter=None):
    # if filter == None:
    #     render_arr[..., 0]
    # if filter == 'cells':
    #     pygame.surfarray.blit_array(surface, cols[fields[0]['cells']])
    #     return True

    O2_max = fields['O2'].max()
    H_max = fields['H'].max()
    render_arr[..., 0] = (fields['O2']/O2_max*155).astype(np.uint8)
    render_arr[..., 1] = (fields['H']/H_max*155).astype(np.uint8)
    render_arr[..., 2] = fields['cells']*80
    pygame.surfarray.blit_array(surface, render_arr.view(np.uint32).reshape(parameters['FIELD_SIZE']).T)

# Logic

buffer_mask = np.empty(parameters['FIELD_SIZE'], dtype=bool)
buffer_float = np.empty(parameters['FIELD_SIZE'], dtype=np.float64)

def O2_live_consumption(fields, params=parameters):
    global buffer_mask, buffer_float
    params['O2_LIVE_CONSUMPTION'].take(fields['cells'], out=buffer_float)
    fields['O2'] -= buffer_float*params['DT']
    np.less(fields['O2'], 0.0, out=buffer_mask)
    fields['O2'][buffer_mask] = 0.0
    fields['H'][buffer_mask] += params['H_HEALTHY_NECR_COUNT']
    fields['cells'][buffer_mask] = 0

def H_healthy_consumption(fields, params=parameters):
    pass
def H_cells_death(fields, params=parameters):
    pass
def H_prolif_to_quiscent(fields, params=parameters):
    pass
def healthy_mitosis(fields, params=parameters):
    pass
def prolif_mitosis(fields, params=parameters):
    pass
def H_O2_quiscent_to_prolif(fields, params=parameters):
    pass
def O2_G2_cells_consumption(fields, params=parameters):
    pass
def age_cells_inc(fields):
    fields['age'] += 1

def O2_diffusion(fields):
    fields['O2'] = fields['O2_dif'].diffuse(fields['O2'])

def H_diffusion(fields):
    fields['H'] = fields['H_dif'].diffuse(fields['H'])
    
    



pygame.init()
scr = pygame.display.set_mode((parameters['FIELD_WIDTH']+100, parameters['FIELD_HEIGHT']+100))
font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

field_pixels = pygame.Surface((parameters['FIELD_WIDTH'], parameters['FIELD_HEIGHT']))
step = 0
max_step = 5000
timer = 0.0

fields['O2'] = 1e-13*np.random.random(parameters['FIELD_SIZE'])
fields['H'] = 1e-13*np.random.random(parameters['FIELD_SIZE'])
fields['cells'][:, :parameters['FIELD_WIDTH']//2] = 0 
fields['cells'][parameters['FIELD_HEIGHT']//2:, parameters['FIELD_WIDTH']//2:] = 2 
fields['cells'][parameters['FIELD_HEIGHT']//2:, parameters['FIELD_WIDTH']//4*3:] = 3

scr.fill((25, 25, 25))
is_simulating = False

running = True
while running:
    # step += 1
    clock.tick()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                is_simulating = not is_simulating
            if event.key == pygame.K_i:
                save_data('output_sameple.txt', fields, format="%d")
            # print(fields[0]['cells'])

    ### Make_step
    if is_simulating: 
        start_time = time.perf_counter_ns()

        O2_live_consumption(fields, parameters)
        H_healthy_consumption(fields, parameters)
        H_cells_death(fields, parameters)
        H_prolif_to_quiscent(fields, parameters)
        healthy_mitosis(fields, parameters)
        prolif_mitosis(fields, parameters)
        H_O2_quiscent_to_prolif(fields, parameters)
        O2_G2_cells_consumption(fields, parameters)
        age_cells_inc(fields)

        
        # for _ in range(100):
        O2_diffusion(fields)
        H_diffusion(fields)

        end_time = time.perf_counter_ns()
        timer += end_time - start_time
        step += 1

        # is_simulating = False

    # pygame.surfarray.blit_array(field_pixels, cell_cols[fields[0]['cells']])
    render_field(field_pixels, fields, cell_cols)
    scr.blit(field_pixels, (50, 50))    
    pygame.draw.rect(scr, col.BLACK, (0, 0, parameters['FIELD_WIDTH']+200, 20))
    scr.blit(font.render(f"{step} {(timer)/max(1, step)/10**(9-3):.3f} {fields['O2'].sum():.3e} {clock.get_fps():.1f}", True, (255, 255, 255)), (0, 0))
    pygame.display.flip()
    if step >= max_step:
        running = False

print(f"t = {(timer)/max(1, step)/10**(9-3)} ms, shape={fields['O2'].shape}")

pygame.quit()