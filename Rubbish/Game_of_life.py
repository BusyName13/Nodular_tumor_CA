import numpy as np
from scipy.signal import convolve2d
import pygame

TRANSPARENT = (0, 0, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

class Fill:
    def __init__(self, surface, color):
        self.color = color
        self.surface = surface
    def evaluate(self):
        self.surface.fill(self.color)

class Text:
    def __init__(self, surface, font, text, pos, color):
        self.text = text
        self.pos = pos
        self.color = color
        self.surface = surface
    def evaluate(self):
        self.text_render = font.render(self.text, True, self.color)
        self.surface.blit(self.text_render, self.pos)

class Circle:
    def __init__(self, surface, pos, radius, color):
        self.surface = surface
        self.pos = pos
        self.radius = radius
        self.color = color
    def evaluate(self):
        pygame.draw.circle(self.surface, self.color, self.pos, self.radius)

class Rectangle:
    def __init__(self, surface, pos, width, height, color):
        self.surface = surface
        self.pos = pos
        self.width = width
        self.height = height
        self.color = color
    def evaluate(self):
        pygame.draw.rect(self.surface, self.color, (self.pos[0], self.pos[1], self.width, self.height))

class Line:
    def __init__(self, surface, start_pos, end_pos, color):
        self.surface = surface
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.color = color
    def evaluate(self):
        pygame.draw.line(self.surface, self.color, self.start_pos, self.end_pos)

class Button:
    def __init__(self, surface, font, pos, width, height, text, event=None, color=BLACK, text_color=WHITE, hover_color=(150, 150, 150), click_color=(50, 50, 50)):
        self.surface = surface
        self.font = font
        self.pos = pos
        self.width = width
        self.height = height
        self.color = color
        self.text_color = text_color
        self.hover_color = hover_color
        self.click_color = click_color
        self.hoverable = True if hover_color != None else False
        self.text = text
        self.event = event
        self.state = 0

    def hover(self, mouse_pos: tuple[int, int]):
        self.text_render = self.font.render(self.text, True, self.text_color)
        if  (0 < mouse_pos[0] - self.pos[0] < self.width) and (0 < mouse_pos[1] - self.pos[1] < self.height):
            if self.hoverable and self.state != 2:
                self.state = 1
            return True
        return False
    
    def click(self, mouse_pos):
        if  (0 < mouse_pos[0] - self.pos[0] < self.width) and (0 < mouse_pos[1] - self.pos[1] < self.height):
                self.state = 2
                self.event()
                return True
        return False
    
    def evaluate(self):
        match self.state:
            case 0: color = self.color
            case 1: color = self.hover_color
            case 2: color = self.click_color
            case _: raise ValueError("Impossible button state")

        rect = pygame.draw.rect(self.surface, color, (self.pos[0], self.pos[1], self.width, self.height), border_radius=5)
        text_rect = self.text_render.get_rect(center=rect.center)
        self.surface.blit(self.text_render, text_rect)



class Canvas:
    def __init__(self, left, top, width, height, cell_size):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.field = np.zeros((self.height, self.width), dtype=int)
        self.start_simulation = False
        self.fps = 10000
    
    def get_elems(self, elem):
        return np.where(self.field == elem)

    def make_tick(self):
        kernel = np.array([
                [1, 1, 1],
                [1, 10, 1],
                [1, 1, 1],
            ])
        self.field = np.isin(convolve2d(self.field, kernel, mode="same", boundary="wrap"), [3, 12, 13]).astype(int)



MAIN_COLOR = WHITE
BACKGROUND_COLOR = BLACK

SCREEN = type("Screen", (), {"size": (600 + 20, 600 + 125)})
SCREEN.width, SCREEN.height = SCREEN.size
CANVAS = Canvas(10, 25, 60, 60, 10) # 600x600




pygame.init()
clock = pygame.time.Clock()
fps = clock.tick()
scr = pygame.display.set_mode(SCREEN.size)
font = pygame.font.Font(None, 24)

to_render = [Fill(scr, BACKGROUND_COLOR)]

for i in range(CANVAS.height+1):
    to_render.append(Line(scr, (CANVAS.left, CANVAS.top+i*CANVAS.cell_size), (CANVAS.left+CANVAS.width*CANVAS.cell_size, CANVAS.top+i*CANVAS.cell_size), MAIN_COLOR))
for j in range(CANVAS.width+1):
    to_render.append(Line(scr, (CANVAS.left+j*CANVAS.cell_size, CANVAS.top), (CANVAS.left+j*CANVAS.cell_size, CANVAS.top+CANVAS.height*CANVAS.cell_size), MAIN_COLOR))

mouse_button_pressed = False

def clear(CANVAS: Canvas):
    CANVAS.field = np.zeros((CANVAS.height, CANVAS.width), dtype=int)

def start_simulation(CANVAS: Canvas, button: Button):
    if not CANVAS.start_simulation:
        CANVAS.fps = 10
        button.text = "Stop"
    else:
        CANVAS.fps = 10000
        button.text = "Start"
    CANVAS.start_simulation = not CANVAS.start_simulation

tmp_button = Button(scr, font, (10, CANVAS.top+CANVAS.height*CANVAS.cell_size + 50), SCREEN.width - 20, 30, "Start/Stop")
tmp_button.event = lambda: start_simulation(CANVAS, tmp_button)
Buttons = [
    Button(scr, font, (10, CANVAS.top+CANVAS.height*CANVAS.cell_size + 10), SCREEN.width//2 - 15, 30, "Step", CANVAS.make_tick),
    Button(scr, font, (SCREEN.width//2 + 5, CANVAS.top+CANVAS.height*CANVAS.cell_size + 10), SCREEN.width - 15, 30, "Clear", lambda: clear(CANVAS)),
    tmp_button
]

to_render += Buttons.copy()


CANVAS.field[1:4, 2] = 1

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
            field_pos = tuple((np.flip(np.array(pygame.mouse.get_pos()))-(CANVAS.top, CANVAS.left))//CANVAS.cell_size)
            if (0 <= field_pos[0] < CANVAS.height) and (0 <= field_pos[1] <= CANVAS.width):
                CANVAS.field[field_pos] = (CANVAS.field[field_pos] + 1) % 2
            for button in Buttons:
                button.click(event.pos)

        if event.type == pygame.MOUSEBUTTONUP:
            mouse_button_pressed = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                CANVAS.make_tick()
        
    # Logic
    if CANVAS.start_simulation:
        CANVAS.make_tick()

    this_tick += list(map(lambda pos: Rectangle(scr, pos, CANVAS.cell_size, CANVAS.cell_size, WHITE), np.flip(np.array(CANVAS.get_elems(1)).T, 1)*CANVAS.cell_size  + (CANVAS.left, CANVAS.top)))

    this_tick.append(Text(scr, font, f"{clock.get_fps():.1f} {len(this_tick)}", (0, 0), RED))
    # this_tick.append(Text(scr, font, str(CANVAS.field), (CANVAS.left, CANVAS.top+CANVAS.height*CANVAS.cell_size), WHITE))
    for command in this_tick:
        command.evaluate()
    pygame.display.update()
pygame.quit()
