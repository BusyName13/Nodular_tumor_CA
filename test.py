import pygame
#wertyuio
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
        self.text_render = font.render(self.text, True, self.color)
        self.surface = surface
    def evaluate(self):
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
        pygame.draw.rect(self.surface, self.color, self.pos, self.width, self.height)

WHITE = (255, 255, 255)
RED = (255, 0, 0)

MARGIN = 15

pygame.init()
clock = pygame.time.Clock()
fps = clock.tick()
scr = pygame.display.set_mode((300, 300))
font = pygame.font.Font(None, 36)

to_render = [Fill(scr, (0, 0, 0))]

run = True
while run: 
    clock.tick(1000)
    is_moved = False
    this_tick = to_render.copy()
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            run = False
        if e.type == pygame.MOUSEMOTION:
            x, y = e.pos
            dx, dy = e.rel
            this_tick.append(Text(scr, font, f"{x}, {y}", (x+10, y), (200, 200, 200)))
            is_moved = True
        if e.type == pygame.MOUSEBUTTONDOWN:
            if e.button == 1:
                x, y = e.pos
                this_tick.append(Circle(scr, (5, 5), 5, WHITE))
                to_render.append(Text(scr, font, f"{x}, {y}", (x+10, y), RED))
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_q:
                for i in range(10):
                    to_render.append(Text(scr, font, f"{x}, {y+i*MARGIN}", (x+10, y+i*MARGIN), (0, 255, 0)))
            if e.key == pygame.K_w:
                for i in range(10):
                    for j in range(10):
                        to_render.append(Text(scr, font, f"{x+j*MARGIN}, {y+i*MARGIN}", (x+10+j*MARGIN, y+i*MARGIN), (0, 0, 255)))

    if not is_moved:
        x, y = pygame.mouse.get_pos()
        this_tick.append(Text(scr, font, f"{x}, {y}", (x+10, y), WHITE))

    this_tick.append(Text(scr, font, f"{clock.get_fps():.1f} {len(this_tick)}", (0, 0), RED))

    for command in this_tick:
        command.evaluate()
    pygame.display.update()
pygame.quit()
