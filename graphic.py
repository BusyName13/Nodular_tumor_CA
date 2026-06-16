import pygame
import colors

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
        self.font = font
    def evaluate(self):
        self.text_render = self.font.render(self.text, True, self.color)
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
    def __init__(self, surface, font, pos, width, height, text, event=None, color=colors.BLACK, text_color=colors.WHITE, hover_color=(150, 150, 150), click_color=(50, 50, 50)):
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