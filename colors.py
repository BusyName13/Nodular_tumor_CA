import numba
TRANSPARENT = (0, 0, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (148, 148, 148)

@numba.njit
def col_to_num(color: tuple[int, int, int]) -> int:
    return color[0] << 16 | color[1] << 8 | color[2]
col_to_num(WHITE)