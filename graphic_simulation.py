import numpy as np
import Nodular_tumour_2 as sim
import pygame
import graphic as g
import colors as col

sim.import_parameters("parameters/parameters_1.txt")
print((sim.parameters['FIELD_HEIGHT'], sim.parameters['FIELD_WIDTH'], 4))
# Graphic
render_arr = np.zeros((sim.parameters['FIELD_HEIGHT'], sim.parameters['FIELD_WIDTH'], 4), dtype=np.uint8) # BGRA
render_arr[..., 3] = 255
def render_field(surface, fields, filter=None):
    O2_max = fields['O2'].max()
    H_max = fields['H'].max()

    if O2_max <= 0: O2_max = 1.0 
    if H_max <= 0: H_max = 1.0

    O2_max = 155 / O2_max
    H_max = 105 / H_max

    np.multiply(fields['O2'], O2_max, out=render_arr[..., 0], casting='unsafe')
    np.multiply(fields['H'], H_max, out=render_arr[..., 1], casting='unsafe')
    np.multiply(fields['cells'], 80, out=render_arr[..., 2], casting='unsafe')

    pygame.surfarray.blit_array(surface, render_arr.view(np.uint32).reshape(sim.parameters['FIELD_SIZE']).T)



pygame.init()
scr = pygame.display.set_mode((sim.parameters['FIELD_WIDTH']+100, sim.parameters['FIELD_HEIGHT']+100))
font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

field_pixels = pygame.Surface((sim.parameters['FIELD_WIDTH'], sim.parameters['FIELD_HEIGHT']))
point_state = {'state': -1, 'O2': -1, 'H': -1, 'age': -1, 'G2': -1}
step = 0
max_step = int(sim.parameters['MAX_TIME']/sim.parameters['DT'])
print(f"max_step = {max_step}")
timer = 0.0

sim.fields['O2'][:] = 6 * sim.parameters['O2_HEALTHY_LIVE_CONSUMPTION'] * sim.parameters['DT']
d = 5
sim.fields['cells'][sim.parameters['FIELD_HEIGHT']//2-d:sim.parameters['FIELD_HEIGHT']//2+d, 
                sim.parameters['FIELD_WIDTH']//2-d:sim.parameters['FIELD_WIDTH']//2+d] = 2 


scr.fill((25, 25, 25))
render_field(field_pixels, sim.fields)
is_simulating = True

running = True
while running:
    clock.tick()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            j, i = pygame.mouse.get_pos()
            i -= 50
            j -= 50
            if event.button == 1:
                sim.fields['O2'][i-5:i+5, j-5:j+5] += sim.parameters['O2_HEALTHY_LIVE_CONSUMPTION'] * 10
            elif event.button == 3:
                sim.fields['H'][i-5:i+5, j-5:j+5] += sim.parameters['H_HEALTHY_DEATH_LIMIT'] * 3
            render_field(field_pixels, sim.fields)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                is_simulating = not is_simulating
            if event.key == pygame.K_i:
                sim.save_data('output_sameple.txt', sim.fields)
            # print(sim.fields[0]['cells'])

    ### Make_step
    if is_simulating: 
        timer += sim.make_step(sim.fields, sim.parameters)
        step += 1
        
        render_field(field_pixels, sim.fields)
        # is_simulating = False

    scr.blit(field_pixels, (50, 50))
    pygame.draw.rect(scr, col.BLACK, (0, 0, sim.parameters['FIELD_WIDTH']+200, 40))
    j, i = pygame.mouse.get_pos()
    i -= 50
    j -= 50
    if (0 <= i <= sim.parameters['FIELD_HEIGHT']-1) and (0 <= j <= sim.parameters['FIELD_HEIGHT']-1):
        point_state['state'] = sim.fields['cells'][i, j]
        point_state['O2'] = sim.fields['O2'][i, j]
        point_state['H'] = sim.fields['H'][i, j]
        point_state['age'] = sim.fields['age'][i, j]
        point_state['G2'] = sim.fields['G2'][i, j]

    scr.blit(font.render(f"t: {step*sim.parameters['DT']:.2f}    av_stp_t: {(timer)/max(1, step)/10**(9-3):.3f}    O2: {sim.fields['O2'].sum():.5e}    H: {sim.fields['H'].sum():.3e}    FPS: {clock.get_fps():.1f}", True, (255, 255, 255)), (0, 0))
    scr.blit(font.render(f"state: {point_state['state']}  O2: {point_state['O2']:.3e}  H: {point_state['H']:.3e} age: {point_state['age']:.2f} G2: {point_state['G2']:.2f}",  True, (255, 255, 255)), (0, 20))
    pygame.display.flip()
    if step >= max_step:
        # running = False
        is_simulating = False

print(f"t = {(timer)/max(1, step)/10**(9-3)} ms, shape = {sim.fields['O2'].shape} steps = {step}")

pygame.quit()