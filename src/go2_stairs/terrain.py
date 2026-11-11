"""Analytic solid stairs shared by scene generation and privileged sampling."""

from dataclasses import dataclass
from bisect import bisect_right
import math


@dataclass(frozen=True)
class StairSpec:
    height: float
    depth: float
    steps: int
    direction: str
    width: float = 1.2
    landing: float = 1.2

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in (self.height, self.depth, self.width, self.landing)):
            raise ValueError('stair dimensions must be finite and positive')
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError('steps must be a positive integer')
        if self.direction not in ('up', 'down'):
            raise ValueError('direction must be up or down')

    @property
    def finish_x(self):
        return self.landing + self.depth * self.steps

    @property
    def length(self):
        return self.finish_x + self.landing

    @property
    def start_height(self):
        return self.height * self.steps if self.direction == 'down' else 0.


@dataclass(frozen=True)
class Box:
    name: str
    center: tuple
    half_size: tuple


def height_at(x, y, spec):
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError('terrain coordinates must be finite')
    if x < 0 or x > spec.length or abs(y) > spec.width / 2:
        return 0.
    edges = [spec.landing + i * spec.depth for i in range(spec.steps)]
    level = bisect_right(edges, x)
    if spec.direction == 'down':
        level = spec.steps - level
    return level * spec.height


def build_boxes(spec):
    bounds = [0., spec.landing] + [spec.landing + i * spec.depth for i in range(1, spec.steps + 1)] + [spec.length]
    result = []
    for i, (left, right) in enumerate(zip(bounds, bounds[1:])):
        center = (left + right) / 2
        top = height_at(center, 0., spec)
        if top > 0:
            result.append(Box(f'stair_segment_{i}', (center, 0., top / 2),
                              ((right-left)/2, spec.width/2, top/2)))
    return result
