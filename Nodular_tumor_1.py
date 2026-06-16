import numpy as np
from scipy.signal import convolve2d
import pygame
from graphic import *
from colors import *

FIELD_SIZE = (750, 750) # in cells
CELL_SIZE = 1 # in px

SIM_SPEED = 6000
FPS = 6000

MAIN_COLOR = WHITE
BACKGROUND_COLOR = BLACK

HEALTHY = BLACK
CANCERED = (138, 74, 91)
DEAD = GRAY
STATES_COLOR = [HEALTHY, CANCERED, DEAD]


class Game:
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.field = np.zeros((self.height, self.width), dtype=int)
        self.start_simulation = False
        self.step = 0
        self.kernel = np.array([
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ])
        # self.kernel = np.array([
        #     [0, 0, 1, 0, 0],
        #     [0, 1.4142, 2, 1.4142, 0],
        #     [1, 2, 0, 2, 1],
        #     [0, 1.4142, 2, 1.4142, 0],
        #     [0, 0, 1, 0, 0]
        # ])

    def make_tick(self):
        self.step += 1
        bound = "wrap"
        # conv_x: type = np.array,  conv_x.shape=self.field,    it is an array that for each cell counts number of near cells of type x (using a kernel)
        conv_0 = convolve2d(self.field==0, self.kernel, mode="same", boundary=bound)
        conv_1 = convolve2d(self.field==1, self.kernel, mode="same", boundary=bound)
        conv_2 = convolve2d(self.field==2, self.kernel, mode="same", boundary=bound)
        
        new_field = np.zeros_like(self.field)

        to_cancer_healthy = (self.field == 0) & ((0.00_000_0 + 0.2*conv_1 - np.random.random((self.height, self.width))) > 0) | (self.field == 1)
        new_field[to_cancer_healthy] = 1
        del to_cancer_healthy

        to_kill_healthy = (self.field == 0) & ((0.01_5*conv_2*conv_2 - np.random.random((self.height, self.width))) > 0) | (self.field == 2)
        new_field[to_kill_healthy] = 2
        del to_kill_healthy

        to_cure_cancered = (self.field == 1) & ((0.05*conv_0 - np.random.random((self.height, self.width))) > 0)
        new_field[to_cure_cancered] = 0
        del to_cure_cancered

        to_kill_cancered = (self.field == 1) & ((0.00_0001*conv_1 + 0.05*conv_2 - np.random.random((self.height, self.width))) > 0)
        new_field[to_kill_cancered] = 2
        del to_kill_cancered

        to_cure_dead = ( (self.field == 2) & ((0.03*conv_0 - np.random.random((self.height, self.width))) > 0) )
        new_field[to_cure_dead] = 0
        del to_cure_dead

        self.field = new_field
        
        # print(np.count_nonzero(self.field == 0), np.count_nonzero(self.field == 1), np.count_nonzero(self.field == 2), sep='\t')


class Canvas:
    def __init__(self, left, top, width, height, cell_size, colors):
        self.left = left
        self.top = top
        self.width = width*cell_size
        self.height = height*cell_size
        self.game = Game(width, height, cell_size)
        self.fps = FPS
    
    def get_elems(self, elem):
        return np.where(self.game.field == elem)
    
    def make_tick(self):
        return self.game.make_tick()
    
    def get_start_simulation(self):
        return self.game.start_simulation
    
    def get_field_params(self):
        return (self.game.width, self.game.height, self.game.cell_size)
    
    def get_field(self):
        return self.game.field

SCREEN = type("Screen", (), {"size": (FIELD_SIZE[0]*CELL_SIZE + 20, FIELD_SIZE[1]*CELL_SIZE + 125)})
SCREEN.width, SCREEN.height = SCREEN.size
CANVAS = Canvas(10, 25, FIELD_SIZE[0], FIELD_SIZE[1], CELL_SIZE, STATES_COLOR) # 300x300

pygame.init()
clock = pygame.time.Clock()
fps = clock.tick()
scr = pygame.display.set_mode(SCREEN.size)
font = pygame.font.Font(None, 24)


def create_agent(i_j_state, states=STATES_COLOR, canvas=CANVAS, surface=scr):
    i, j, state = i_j_state
    if state == -1:
        return None
    pos = (canvas.left + j*canvas.game.cell_size, 
           canvas.top  + i*canvas.game.cell_size)
    return Rectangle(surface, pos, canvas.game.cell_size, canvas.game.cell_size, states[state])

def condition(prev_field, current_field):
    return prev_field != current_field
# condition = np.vectorize(condition)

def make_agents(field, prev_feld, func):
    cond_field = np.array(np.where(condition(prev_feld, field))).T
    return [func((pos[0], pos[1], field[tuple(pos)])) for pos in cond_field]

# def make_agents(field, prev_field, func):
#     field_copy = field.copy()
#     i, j = np.meshgrid(np.arange(field.shape[1]), np.arange(field.shape[0]))
#     field_copy = np.stack((j, i, field_copy), axis=-1).reshape((field.shape[0]*field.shape[1], 3))
#     dt = np.dtype([('i', np.int64), ('j', np.int64), ('state', np.int64)])
#     field_copy = np.array(list(map(tuple, field_copy)), dtype=dt)

#     if len(prev_field) != 0:
#         field_copy = field_copy[condition(field, prev_field).reshape(field.shape[0]*field.shape[1])]
#     if len(field_copy) == 0:
#         return []
#     func = np.vectorize(func)
#     field_copy = list(func(field_copy))
#     return field_copy



to_render = [Fill(scr, BACKGROUND_COLOR)]
to_render = [Rectangle(scr, (0, 0), 300, 25, BACKGROUND_COLOR)]

# # Lines
# for i in range(CANVAS.game.height+1):
#     to_render.append(Line(scr, (CANVAS.left, CANVAS.top+i*CANVAS.game.cell_size), (CANVAS.left+CANVAS.width, CANVAS.top+i*CANVAS.game.cell_size), MAIN_COLOR))
# for j in range(CANVAS.game.width+1):
#     to_render.append(Line(scr, (CANVAS.left+j*CANVAS.game.cell_size, CANVAS.top), (CANVAS.left+j*CANVAS.game.cell_size, CANVAS.top+CANVAS.height), MAIN_COLOR))

mouse_button_pressed = False

def clear(CANVAS: Canvas):
    CANVAS.game.field = np.zeros((CANVAS.game.height, CANVAS.game.width), dtype=int)
    CANVAS.game.step = 0

def start_simulation(CANVAS: Canvas, button: Button):
    if not CANVAS.game.start_simulation:
        CANVAS.fps = SIM_SPEED
        button.text = "Stop"
    else:
        CANVAS.fps = FPS
        button.text = "Start"
    CANVAS.game.start_simulation = not CANVAS.game.start_simulation

tmp_button = Button(scr, font, (10, CANVAS.top+CANVAS.height + 50), SCREEN.width - 20, 30, "Start/Stop")
tmp_button.event = lambda: start_simulation(CANVAS, tmp_button)
Buttons = [
    Button(scr, font, (10, CANVAS.top+CANVAS.height + 10), SCREEN.width//2 - 15, 30, "Step", CANVAS.make_tick),
    Button(scr, font, (SCREEN.width//2 + 5, CANVAS.top+CANVAS.height + 10), SCREEN.width - 15, 30, "Clear", lambda: clear(CANVAS)),
    tmp_button
]

to_render += Buttons.copy()

CANVAS.game.field[CANVAS.game.height//2, CANVAS.game.width//2] = 1  

prev_field = -1*np.ones_like(CANVAS.get_field())

# print(condition(prev_field, ))
# raise

running = True
while running: 
    this_tick = to_render.copy()
    for button in Buttons:
        button.state = 0
        button.hover(pygame.mouse.get_pos())
    clock.tick(CANVAS.fps)

    # Event compilation
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_button_pressed = True
            field_pos = tuple((np.flip(np.array(pygame.mouse.get_pos()))-(CANVAS.top, CANVAS.left))//CANVAS.game.cell_size)
            if (0 <= field_pos[0] < CANVAS.game.height) and (0 <= field_pos[1] <= CANVAS.game.width):
                CANVAS.game.field[field_pos] = (CANVAS.game.field[field_pos] + 1) % 2
            for button in Buttons:
                button.click(event.pos)

        if event.type == pygame.MOUSEBUTTONUP:
            mouse_button_pressed = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                CANVAS.game.make_tick()
            if event.key == pygame.K_c:
                clear(CANVAS)
        
    # Logic
    if CANVAS.game.start_simulation:
        CANVAS.game.make_tick()
    
    this_tick += make_agents(CANVAS.game.field, prev_field, create_agent)
    # this_tick += list(map(lambda pos: Rectangle(scr, pos, CANVAS.game.cell_size, CANVAS.game.cell_size, STATES_COLOR[0]), np.flip(np.array(CANVAS.get_elems(0)).T, 1)*CANVAS.game.cell_size  + (CANVAS.left, CANVAS.top)))
    # this_tick += list(map(lambda pos: Rectangle(scr, pos, CANVAS.game.cell_size, CANVAS.game.cell_size, STATES_COLOR[1]), np.flip(np.array(CANVAS.get_elems(1)).T, 1)*CANVAS.game.cell_size  + (CANVAS.left, CANVAS.top)))
    # this_tick += list(map(lambda pos: Rectangle(scr, pos, CANVAS.game.cell_size, CANVAS.game.cell_size, STATES_COLOR[2]), np.flip(np.array(CANVAS.get_elems(2)).T, 1)*CANVAS.game.cell_size  + (CANVAS.left, CANVAS.top)))

    this_tick.append(Text(scr, font, f"{clock.get_fps():.1f} {len(this_tick)} ({np.count_nonzero(CANVAS.get_field() == 0)}, {np.count_nonzero(CANVAS.get_field() == 1)}, {np.count_nonzero(CANVAS.get_field() == 2)}) {CANVAS.game.step}", (0, 0), RED))
    # this_tick.append(Text(scr, font, str(CANVAS.field), (CANVAS.left, CANVAS.top+CANVAS.height*CANVAS.game.cell_size), WHITE))
    for command in this_tick:
        command.evaluate()
    pygame.display.update()

    prev_field = CANVAS.game.field.copy()

pygame.quit()
