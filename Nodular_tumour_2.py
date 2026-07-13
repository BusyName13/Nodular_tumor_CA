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
              'MAX_TIME': -1.0, 'DT': 1.0, 'DT_S': 3600.0, 'DX': 3e-05,
              'O2_DIFFUSION_K': 2.41e-9, 'H_DIFFUSION_K': 1.1e-9, 
              'O2_HEALTHY_LIVE_CONSUMPTION': -1.0, 'O2_HEALTHY_MITOSIS_CONSUMPTION': -1.0, 'O2_HEALTHY_HYPOXIA_LIMIT': -1.0, 'H_HEALTHY_NECR_COUNT': -1.0, 'H_HEALTHY_LIVE_CONSUMPTION': -1.0, 'H_HEALTHY_DEATH_LIMIT': -1.0, 'H_HEALTHY_APOPTOSIS_CONSUMPTION': -1.0, 'AGE_HEALTHY_ADULT': -1.0, 'AGE_HEALTHY_G2': -1.0,
              'O2_PROLIF_TUMOUR_LIVE_CONSUMPTION': -1.0, 'O2_PROLIF_TUMOUR_MITOSIS_CONSUMPTION': -1.0, 'O2_PROLIF_TUMOUR_QUISC_LIMIT': -1.0, 'H_PROLIF_TUMOUR_RELEASE_COUNT': -1.0, 'AGE_PROLIF_TUMOUR_ADULT': -1.0, 'AGE_PROLIF_TUMOUR_G2': -1.0, 
              'O2_QUISC_TUMOUR_LIVE_CONSUMPTION': -1.0, 'O2_QUISC_TO_PROLIF_LIMIT': -1.0, 'O2_QUISC_TUMOUR_NECR_LIMIT': -1.0, 'H_QUISC_TO_PROLIF_LIMIT': -1.0, 'H_QUISC_TUMOUR_RELEASE_COUNT': -1.0, 'H_QUISC_TUMOUR_NECR_COUNT': 6.5e-15}
parameters["FIELD_SIZE"] = (parameters['FIELD_HEIGHT'], parameters['FIELD_WIDTH'])
parameters["DT_S"] = 3600 * parameters['DT']
parameters["O2_LIVE_CONSUMPTION"] = parameters['DT'] * np.array([0.0, parameters['O2_HEALTHY_LIVE_CONSUMPTION'], parameters['O2_PROLIF_TUMOUR_LIVE_CONSUMPTION'], parameters['O2_QUISC_TUMOUR_LIVE_CONSUMPTION']])
parameters["H_NECR_COUNT"] = np.array([0.0, parameters['H_HEALTHY_NECR_COUNT'], 0.0, parameters['H_QUISC_TUMOUR_NECR_COUNT']])
parameters["H_RELEASE_COUNT"] = parameters['DT'] * np.array([0.0, 0.0, parameters['H_PROLIF_TUMOUR_RELEASE_COUNT'], parameters['H_QUISC_TUMOUR_RELEASE_COUNT']])
parameters["AGE_ADULT"] = np.array([0.0, parameters['AGE_HEALTHY_ADULT'], parameters['AGE_PROLIF_TUMOUR_ADULT'], 0.0])
parameters["AGE_G2"] = np.array([0.0, parameters['AGE_HEALTHY_G2'], parameters['AGE_PROLIF_TUMOUR_G2'], 0.0])
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
        params["O2_LIVE_CONSUMPTION"] = parameters['DT'] * np.array([0.0, params['O2_HEALTHY_LIVE_CONSUMPTION'], params['O2_PROLIF_TUMOUR_LIVE_CONSUMPTION'], params['O2_QUISC_TUMOUR_LIVE_CONSUMPTION']])
        params["H_NECR_COUNT"] = np.array([0.0, params['H_HEALTHY_NECR_COUNT'], 0.0, params['H_QUISC_TUMOUR_NECR_COUNT']])
        params["H_RELEASE_COUNT"] = parameters['DT'] * np.array([0.0, 0.0, params['H_PROLIF_TUMOUR_RELEASE_COUNT'], params['H_QUISC_TUMOUR_RELEASE_COUNT']])
        params["AGE_ADULT"] = np.array([0.0, params['AGE_HEALTHY_ADULT'], params['AGE_PROLIF_TUMOUR_ADULT'], 0.0])
        params["AGE_G2"] = np.array([0.0, params['AGE_HEALTHY_G2'], params['AGE_PROLIF_TUMOUR_G2'], 0.0])
        
    return dict
import_parameters("parameters.txt", parameters)

def save_data(file_name, fields, format="%d", filter=None):
    np.savetxt(file_name, fields['cells'], fmt=format)
# 1.35e-14 - H_HEAL_LIM
# H_PROLIF_TUMOUR_RELEASE_COUNT 6.0e-15


fields =  {"cells":  np.ones(parameters['FIELD_SIZE'], dtype=np.uint8),
           "O2":     np.zeros(parameters['FIELD_SIZE'], dtype=np.float64), 
           "O2_dif": FFT_Diffusion(parameters['FIELD_SIZE'], parameters['DX'], parameters['DT_S'], parameters['O2_DIFFUSION_K']),
           "H":      np.zeros(parameters['FIELD_SIZE'], dtype=np.float64),
           "H_dif":  FFT_Diffusion(parameters['FIELD_SIZE'], parameters['DX'], parameters['DT_S'], parameters['H_DIFFUSION_K']),
           "age":    (100 * np.random.random(parameters['FIELD_SIZE'])) % 4,
           "G2":     np.zeros(parameters['FIELD_SIZE'], dtype=np.float64)
           }


### Buffers

buffer_mask = np.empty(parameters['FIELD_SIZE'], dtype=bool)
buffer_mask2 = np.empty(parameters['FIELD_SIZE'], dtype=bool)
buffer_mask3 = np.empty(parameters['FIELD_SIZE'], dtype=bool)
buffer_mask_4d = [np.empty(parameters['FIELD_SIZE'], dtype=bool) for _ in range(4)]
# i_indices, j_indices = np.ogrid[:parameters['FIELD_HEIGHT'], :parameters['FIELD_WIDTH']]
buffer_float = np.empty(parameters['FIELD_SIZE'], dtype=np.float64)
buffer_float2 = np.empty(parameters['FIELD_SIZE'], dtype=np.float64)
buffer_int_1 = np.empty(parameters['FIELD_SIZE'], dtype=int)
buffer_int_2 = np.empty(parameters['FIELD_SIZE'], dtype=int)
buffer_int16_1 = np.empty(parameters['FIELD_SIZE'], dtype=np.int16)
buffer_int16_2 = np.empty(parameters['FIELD_SIZE'], dtype=np.int16)
buffer_int8 = np.empty(parameters['FIELD_SIZE'], dtype=np.int8)
# buffer_shift_fields = np.empty()
rand_generator = np.random.default_rng()



### Functions

# Math
@numba.njit(parallel=True, fastmath=True, cache=True)
def conv2d_calc(field, kernel, pad_field, out): 
    k_h, k_w = kernel.shape 
    f_h, f_w = field.shape 
    c_i, c_j = k_h//2, k_w//2 

    pad_field[ c_i:-c_i, c_j:-c_j] = field 
    pad_field[ :c_i, c_j:-c_j] = field[-c_i:, :] 
    pad_field[-c_i:, c_j:-c_j] = field[ :c_i, :] 
    pad_field[ :, :c_j] = pad_field[:, -2*c_j:-c_j] 
    pad_field[ :, -c_j:] = pad_field[:, c_j:2*c_j] 

    out[:] = 0
    for f_i in numba.prange(f_h): 
        for k_i in range(k_h): 
            for k_j in range(k_w): 
                for f_j in range(f_w): 
                    out[f_i, f_j] += pad_field[f_i+k_i, f_j+k_j] * kernel[k_i, k_j] 
    return out

cache = {}
def conv2d(field, kernel, out=None):
    k_h, k_w = kernel.shape
    f_h, f_w =  field.shape
    c_i, c_j = k_h//2, k_w//2
    if out == None:
        out = np.empty_like(field)
    if (f_h, f_w, k_h, k_w) in cache:
        pad_field, res = cache[(f_h, f_w, k_h, k_w)]
    else:
        pad_field = np.empty((f_h+2*c_i, f_w+2*c_j), dtype=field.dtype)  
        res = np.zeros_like(field)
        cache[(f_h, f_w, k_h, k_w)] = (pad_field, res)
    return conv2d_calc(field, kernel, pad_field, out)
_ = conv2d(np.ones((3, 3), dtype=np.float64), np.ones((3, 3), dtype=np.float64))
_ = conv2d(np.ones((3, 3), dtype=np.int64), np.ones((3, 3), dtype=np.uint8))

@numba.njit(fastmath=True, cache=True)
def run_mitosis_loop_numba(
    cells, age, g2, ready_parents_mask, 
    buffer_y, buffer_x, buffer_dir,
    num_empty, field_h, field_w, mode
    ):

    if num_empty == 0:
        return

    dy_arr = (0, -1,  0,  1)
    dx_arr = (1,  0, -1,  0)
    for step in range(4):
        write_idx = 0
        any_success = False

        for i in range(num_empty):
            curr_y = buffer_y[i]
            curr_x = buffer_x[i]
            
            direction = (buffer_dir[i] + step) & 3
            
            parent_y = (curr_y + dy_arr[direction]) % field_h
            parent_x = (curr_x + dx_arr[direction]) % field_w
            
            if ready_parents_mask[parent_y, parent_x]:
                cells[curr_y, curr_x] = mode
                age[curr_y, curr_x] = 0.0
                g2[curr_y, curr_x] = 0.0
                
                age[parent_y, parent_x] = 0.0
                g2[parent_y, parent_x] = 0.0
                
                ready_parents_mask[parent_y, parent_x] = False
                
                any_success = True
            else:
                buffer_y[write_idx] = curr_y
                buffer_x[write_idx] = curr_x
                buffer_dir[write_idx] = buffer_dir[i]
                write_idx += 1
                
        num_empty = write_idx
        if num_empty == 0:
            break

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

    if O2_max <= 0: O2_max = 1.0
    if H_max <= 0: H_max = 1.0

    O2_max = 155 / O2_max
    H_max = 105 / H_max

    np.multiply(fields['O2'], O2_max, out=render_arr[..., 0], casting='unsafe')
    np.multiply(fields['H'], H_max, out=render_arr[..., 1], casting='unsafe')
    np.multiply(fields['cells'], 80, out=render_arr[..., 2], casting='unsafe')

    pygame.surfarray.blit_array(surface, render_arr.view(np.uint32).reshape(parameters['FIELD_SIZE']).T)

# Logic

def O2_live_consumption(fields, params=parameters):
    global buffer_mask, buffer_mask2, buffer_float
    params['O2_LIVE_CONSUMPTION'].take(fields['cells'], out=buffer_float)
    fields['O2'] -= buffer_float
    np.less(fields['O2'], 0.0, out=buffer_mask)
    fields['O2'][buffer_mask] = 0.0

    # Quiscent hypoxia death
    np.equal(fields['cells'], 3, out=buffer_mask2)
    np.logical_and(buffer_mask, buffer_mask2, out=buffer_mask2)
    fields['cells'][buffer_mask2] = 0
    fields['H'][buffer_mask2] += params['H_QUISC_TUMOUR_NECR_COUNT']

    # Prolifirated hypoxia transformation
    np.equal(fields['cells'], 2, out=buffer_mask2)
    np.logical_and(buffer_mask, buffer_mask2, out=buffer_mask2)
    fields['cells'][buffer_mask2] = 3

    # Healthy hypoxia death
    np.equal(fields['cells'], 1, out=buffer_mask2)
    np.logical_and(buffer_mask, buffer_mask2, out=buffer_mask2)
    fields['cells'][buffer_mask2] = 0
    fields['H'][buffer_mask2] += params['H_HEALTHY_NECR_COUNT']

def H_healthy_consumption(fields, params=parameters):
    global buffer_mask, buffer_float
    np.equal(fields['cells'], 1, out=buffer_mask)
    np.multiply(buffer_mask, params['H_HEALTHY_LIVE_CONSUMPTION'], out=buffer_float)
    buffer_float *= params['DT']
    fields['H'] -= buffer_float
    np.less(fields['H'], 0.0, out=buffer_mask)
    fields['H'][buffer_mask] = 0.0

def H_cells_lim(fields, params=parameters):
    global buffer_mask, buffer_mask2, buffer_float
    np.greater(fields['H'], params['H_HEALTHY_DEATH_LIMIT'], out=buffer_mask)
    np.equal(fields['cells'], 1, out=buffer_mask2)
    np.logical_and(buffer_mask, buffer_mask2, out=buffer_mask2)
    fields['cells'][buffer_mask2] = 0
    fields['H'][buffer_mask2] -= params['H_HEALTHY_APOPTOSIS_CONSUMPTION']

    np.greater(fields['H'], params['H_PROLIF_TO_QUISC_LIMIT'], out=buffer_mask)
    np.equal(fields['cells'], 2, out=buffer_mask2)
    np.logical_and(buffer_mask, buffer_mask2, out=buffer_mask2)
    fields['cells'][buffer_mask2] = 3

def healthy_mitosis(fields, params=parameters):
    global buffer_mask, buffer_mask2
    global buffer_int16_1, buffer_int16_2, buffer_int8
    #, buffer_mask_4d, i_indices, j_indices
    np.equal(fields['cells'], 1, out=buffer_mask)
    np.greater_equal(fields['G2'], params['AGE_HEALTHY_G2'], out=buffer_mask2)
    np.logical_and(buffer_mask, buffer_mask2, out=buffer_mask)
    # buffer_mask - healthy cells that ready to mitosis

    np.equal(fields['cells'], 0, out=buffer_mask2)
    y, x = np.nonzero(buffer_mask2)

    num_empty = len(y)
    if num_empty == 0:
        return

    buffer_int8.ravel()[:num_empty] = np.random.randint(0, 4, size=num_empty, dtype=np.int8)
    run_mitosis_loop_numba(fields['cells'], fields['age'], fields['G2'], 
                           buffer_mask, y, x, buffer_int8.ravel(), 
                           num_empty, params['FIELD_HEIGHT'], params['FIELD_WIDTH'], 1)


def prolif_mitosis(fields, params=parameters):
    global buffer_mask, buffer_mask2
    global buffer_int16_1, buffer_int16_2, buffer_int8
    #, buffer_mask_4d, i_indices, j_indices
    np.equal(fields['cells'], 2, out=buffer_mask)
    np.greater_equal(fields['G2'], parameters['AGE_PROLIF_TUMOUR_G2'], out=buffer_mask2)
    np.logical_and(buffer_mask, buffer_mask2, out=buffer_mask)
    # buffer_mask - healthy cells that ready to mitosis

    np.less(fields['cells'], 2, out=buffer_mask2)
    y, x = np.nonzero(buffer_mask2)
    
    num_empty = len(y)
    if num_empty == 0:
        return
    
    buffer_int8.ravel()[:num_empty] = np.random.randint(0, 4, size=num_empty, dtype=np.int8)
    run_mitosis_loop_numba(fields['cells'], fields['age'], fields['G2'], 
                           buffer_mask, y, x, buffer_int8.ravel(), 
                           num_empty, params['FIELD_HEIGHT'], params['FIELD_WIDTH'], 2)

def H_O2_quiscent_to_prolif(fields, params=parameters):
    global buffer_mask, buffer_mask2, buffer_float
    np.greater(fields['O2'], params['O2_QUISC_TO_PROLIF_LIMIT'], out=buffer_mask)
    np.less(fields['H'], params['H_QUISC_TO_PROLIF_LIMIT'], out=buffer_mask2)
    np.logical_and(buffer_mask, buffer_mask2, out=buffer_mask)
    np.equal(fields['cells'], 3, out=buffer_mask2)
    np.logical_and(buffer_mask, buffer_mask2, out=buffer_mask)
    fields['cells'][buffer_mask] = 2

def O2_G2_cells_consumption(fields, params=parameters):
    global buffer_mask, buffer_mask2, buffer_float, buffer_float2
    params['O2_LIVE_CONSUMPTION'].take(fields['cells'], out=buffer_float)
    np.greater_equal(fields['O2'], buffer_float, out=buffer_mask)
    params['AGE_ADULT'].take(fields['cells'], out=buffer_float2)
    np.greater_equal(fields['age'], buffer_float2, out=buffer_mask2)
    np.logical_and(buffer_mask, buffer_mask2, out=buffer_mask)
    fields['O2'][buffer_mask] -= buffer_float[buffer_mask]
    fields['G2'][buffer_mask] += params['DT']

def age_cells_inc(fields, params=parameters):
    fields['age'] += params['DT']

def O2_production(fields, params=parameters):
    fields['O2'][:, #params['FIELD_HEIGHT']//2-5:params['FIELD_HEIGHT']//2+5, 
                 params['FIELD_WIDTH']//4-5:params['FIELD_WIDTH']//4+5] += 3.0e-14*parameters['DT']

def H_production(fields, params=parameters):
    global buffer_float
    params['H_RELEASE_COUNT'].take(fields['cells'], out=buffer_float)
    fields['H'] += buffer_float

def O2_diffusion(fields):
    fields['O2'] = fields['O2_dif'].diffuse(fields['O2'])

def H_diffusion(fields):
    fields['H'] = fields['H_dif'].diffuse(fields['H'])
    
    



pygame.init()
scr = pygame.display.set_mode((parameters['FIELD_WIDTH']+100, parameters['FIELD_HEIGHT']+100))
font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

field_pixels = pygame.Surface((parameters['FIELD_WIDTH'], parameters['FIELD_HEIGHT']))
point_state = {'state': -1, 'O2': -1, 'H': -1, 'age': -1, 'G2': -1}
step = 0
max_step = int(parameters['MAX_TIME']/parameters['DT'])
print(f"max_step = {max_step}")
timer = 0.0

fields['O2'] = 1e-13*np.random.random(parameters['FIELD_SIZE'])
# fields['H'] = 1.5e-14*np.random.random(parameters['FIELD_SIZE'])
fields['cells'][:, :parameters['FIELD_WIDTH']//2] = 0 

fields['cells'][parameters['FIELD_HEIGHT']//8*6:parameters['FIELD_HEIGHT']//8*7, parameters['FIELD_WIDTH']//2:] = 2 
# fields['cells'][parameters['FIELD_HEIGHT']//2:, parameters['FIELD_WIDTH']//4*3:] = 3



scr.fill((25, 25, 25))
render_field(field_pixels, fields, cell_cols)
is_simulating = True

running = True
while running:
    # step += 1
    clock.tick()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            j, i = pygame.mouse.get_pos()
            i -= 50
            j -= 50
            if event.button == 1:
                fields['O2'][i-5:i+5, j-5:j+5] += 1.0e-11
            elif event.button == 3:
                fields['H'][i-5:i+5, j-5:j+5] += 3.0e-12
            render_field(field_pixels, fields, cell_cols)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                is_simulating = not is_simulating
            if event.key == pygame.K_i:
                save_data('output_sameple.txt', fields)
            # print(fields[0]['cells'])

    ### Make_step
    if is_simulating: 
        start_time = time.perf_counter_ns()
        
        O2_live_consumption(fields, parameters)
        H_healthy_consumption(fields, parameters)
        H_cells_lim(fields, parameters)
        healthy_mitosis(fields, parameters)
        prolif_mitosis(fields, parameters)
        H_O2_quiscent_to_prolif(fields, parameters)
        O2_G2_cells_consumption(fields, parameters)
        age_cells_inc(fields, parameters)
        O2_production(fields, parameters)
        H_production(fields, parameters)
        O2_diffusion(fields)
        H_diffusion(fields)

        end_time = time.perf_counter_ns()

        # start_time = time.perf_counter_ns()
        timer += end_time - start_time
        step += 1
        
        render_field(field_pixels, fields, cell_cols)
        # is_simulating = False

    # pygame.surfarray.blit_array(field_pixels, cell_cols[fields[0]['cells']])
    scr.blit(field_pixels, (50, 50))    
    pygame.draw.rect(scr, col.BLACK, (0, 0, parameters['FIELD_WIDTH']+200, 40))
    j, i = pygame.mouse.get_pos()
    i -= 50
    j -= 50
    if (0 <= i <= parameters['FIELD_HEIGHT']-1) and (0 <= j <= parameters['FIELD_HEIGHT']-1):
        point_state['state'] = fields['cells'][i, j]
        point_state['O2'] = fields['O2'][i, j]
        point_state['H'] = fields['H'][i, j]
        point_state['age'] = fields['age'][i, j]
        point_state['G2'] = fields['G2'][i, j]
        # circ = g.Circle(scr, (j+50, i+50), 5, 0)
        # circ.evaluate()
    scr.blit(font.render(f"t: {step*parameters['DT']:.2f}    av_stp_t: {(timer)/max(1, step)/10**(9-3):.3f}    O2: {fields['O2'].sum():.5e}    H: {fields['H'].sum():.3e}    FPS: {clock.get_fps():.1f}", True, (255, 255, 255)), (0, 0))
    scr.blit(font.render(f"state: {point_state['state']}  O2: {point_state['O2']:.3e}  H: {point_state['H']:.3e} age: {point_state['age']:.2f} G2: {point_state['G2']:.2f}",  True, (255, 255, 255)), (0, 20))
    pygame.display.flip()
    if step >= max_step:
        running = False

print(f"t = {(timer)/max(1, step)/10**(9-3)} ms, shape = {fields['O2'].shape} steps = {step}")

pygame.quit()