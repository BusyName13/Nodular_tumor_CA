import numpy as np
import numba # taichi
import pygame
import graphic as g
import colors as col
import time

### Consts
cell_cols = np.array([
    0, #Empty_cell
    col.col_to_num((240, 160, 117)), # Healthy_cell
    col.col_to_num((242, 39, 39)), # Proliferating tumour cell
    col.col_to_num((102, 17, 17)), # Quiscent tumour cell
])

parameters = {'FIELD_WIDTH': 750, 'FIELD_HEIGHT': 750, 
              'DT': 1.0, 'DX': 3e-05, 'DN_O2': -1.0, 'DN_H': -1.0, 
              'EMPTY_O2_DIFFUSION_LIMIT': -1.0, 'EMPTY_O2_DIFFUSION_K': 2e-9, 'EMPTY_H_DIFFUSION_LIMIT': -1.0, 'EMPTY_H_DIFFUSION_K': -1.0, 'CELLULAR_O2_DIFFUSION_LIMIT': -1.0, 'CELLULAR_O2_DIFFUSION_K': -1.0, 'CELLULAR_H_DIFFUSION_LIMIT': -1.0, 'CELLULAR_H_DIFFUSION_K': -1.0, 
              'O2_HEALTHY_LIVE_CONSUMPTION': -1.0, 'O2_HEALTHY_MITOSIS_CONSUMPTION': -1.0, 'O2_HEALTHY_HYPOXIA_LIMIT': -1.0, 'H_HEALTHY_NECR_COUNT': -1.0, 'H_HEALTHY_LIVE_CONSUMPTION': -1.0, 'H_HEALTHY_DEATH_LIMIT': -1.0, 'H_HEALTHY_APOPTOSIS_CONSUMPTION': -1.0, 'AGE_HEALTHY_MITOSIS': -1.0, 
              'O2_PROLIF_TUMOUR_LIVE_CONSUMPTION': -1.0, 'O2_PROLIF_TUMOUR_MITOSIS_CONSUMPTION': -1.0, 'O2_PROLIF_TUMOUR_QUISC_LIMIT': -1.0, 'H_PROLIF_TUMOUR_RELEASE_COUNT': -1.0, 'AGE_PROLIF_TUMOUR_MITOSIS': -1.0, 
              'O2_QUISC_TUMOUR_LIVE_CONSUMPTION': -1.0, 'O2_QUISC_TUMOUR_PROLIF_LIMIT': -1.0, 'O2_QUISC_TUMOUR_NECR_LIMIT': -1.0, 'H_QUISC_TUMOUR_RELEASE_COUNT': -1.0}
def import_parameters(file_name):
    dict = {}
    with open(file_name, 'r', encoding='utf-8') as f:
        for _ in range(2):
            line = f.readline()
            key, val = line.split()
            dict[key] = int(val)
        dict.update({key: float(val) for line in f for key, val in [line.split()]})
    return dict
parameters = import_parameters("parameters.txt")

fields = [{"cells": np.ones((parameters['FIELD_HEIGHT'], parameters['FIELD_WIDTH']), dtype=np.uint8),
           "O2":    np.zeros((parameters['FIELD_HEIGHT'], parameters['FIELD_WIDTH']), dtype=np.float64), 
           "H":     np.zeros((parameters['FIELD_HEIGHT'], parameters['FIELD_WIDTH']), dtype=np.float64),
           "age":   parameters['DT'] * np.random.randint(0, 100, (parameters['FIELD_HEIGHT'], parameters['FIELD_WIDTH']), dtype=np.uint8)},
          {"cells": np.ones((parameters['FIELD_HEIGHT'], parameters['FIELD_WIDTH']), dtype=np.uint8),
           "O2":    np.zeros((parameters['FIELD_HEIGHT'], parameters['FIELD_WIDTH']), dtype=np.float64),
           "H":     np.zeros((parameters['FIELD_HEIGHT'], parameters['FIELD_WIDTH']), dtype=np.float64),
           "age":   0}]
fields[1]['age'] = fields[0]['age']

### Functions

# Math
@numba.njit(parallel=True, fastmath=True)
def conv2d(field, kernel):
    k_h, k_w = kernel.shape
    f_h, f_w =  field.shape
    c_i, c_j = k_h//2, k_w//2
    
    res = np.zeros_like(field)
    for i in numba.prange(f_h*f_w):
        f_i = i // f_w
        f_j = i % f_w
        for k_i in range(-c_i, k_h-c_i):
            for k_j in range(-c_j, k_w-c_j):
                res[f_i, f_j] += field[(f_i+k_i)%f_h, (f_j+k_j)%f_w] * kernel[k_i+c_i, k_j+c_j]
    return res
conv2d(np.ones((3, 3)), np.ones((3, 3)))

# Graphic
def render_field(surface, fields, cols, filter=None):
    if filter == None or filter == 'cells':
        # pygame.surfarray.blit_array(surface, cols[fields[0]['cells']])
        return True

    # O2 gradient
    O2_max = fields[0]['O2'].max()
    res = (fields[0]['O2']/O2_max*255).astype(np.int64)
    pygame.surfarray.blit_array(surface, res)

# Logic
def O2_diffushion(fields, params):
    kernel = np.array([
        [1/12, 1/6, 1/12],
        [1/6, -1, 1/6],
        [1/12, 1/6, 1/12]
    ], dtype=np.float64)
    
    



pygame.init()
scr = pygame.display.set_mode((parameters['FIELD_WIDTH']+100, parameters['FIELD_HEIGHT']+100))
font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

field_pixels = pygame.Surface((parameters['FIELD_WIDTH'], parameters['FIELD_HEIGHT']))
step = 0
max_step = 5000
start_time = time.perf_counter_ns()



running = True
while running:
    step += 1
    clock.tick()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pass
            # print(fields[0]['cells'])

    # pygame.surfarray.blit_array(field_pixels, cell_cols[fields[0]['cells']]) 
    fields[0]['O2'] = 10*np.random.random((parameters['FIELD_HEIGHT'], parameters['FIELD_WIDTH']))
    render_field(field_pixels, fields, cell_cols)
    scr.blit(field_pixels, (50, 50))    
    pygame.draw.rect(scr, col.BLACK, (0, 0, 100, 40))
    scr.blit(font.render(f"{clock.get_fps():.1f} {step}", True, (255, 255, 255)), (0, 0))
    pygame.display.flip()
    if step >= max_step:
        running = False

end_time = time.perf_counter_ns()
print((end_time-start_time)/step/10**(9-3))

pygame.quit()