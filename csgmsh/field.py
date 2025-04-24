from typing import Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math


AsField = Union['Field',int,float]


class Field(ABC):
    'A scalar field over a coordinate system.'

    @abstractmethod
    def gettag(self, ff, fragments):
        ...


def _expr(obj):
    if isinstance(obj, MathEval):
        return obj.expr
    elif isinstance(obj, (int, float)):
        return str(obj)
    else:
        raise ValueError(f'unknown object for MathEval: {obj!r}')


class MathEval(Field):

    def __init__(self, expr: str):
        assert isinstance(expr, str)
        self.expr = expr

    def gettag(self, ff, fragments):
        tag = ff.add('MathEval')
        ff.setString(tag, 'F', self.expr)
        return tag

    def __add__(self, other) -> 'Field':
        return MathEval(f'({self.expr})+({_expr(other)})')

    def __radd__(self, other) -> 'Field':
        return MathEval(f'({_expr(other)})+({self.expr})')

    def __sub__(self, other) -> 'Field':
        return MathEval(f'({self.expr})-({_expr(other)})')

    def __rsub__(self, other) -> 'Field':
        return MathEval(f'({_expr(other)})-({self.expr})')

    def __mul__(self, other) -> 'Field':
        return MathEval(f'({self.expr})*({_expr(other)})')

    def __rmul__(self, other) -> 'Field':
        return MathEval(f'({_expr(other)})*({self.expr})')

    def __truediv__(self, other) -> 'Field':
        return MathEval(f'({self.expr})/({_expr(other)})')

    def __rtruediv__(self, other) -> 'Field':
        return MathEval(f'({_expr(other)})/({self.expr})')

    def __pow__(self, other) -> 'Field':
        return MathEval(f'({self.expr})^({_expr(other)})')


x = MathEval('x')
y = MathEval('y')
z = MathEval('z')


class Distance(Field):
    'Refine elements uniformly inside a circle'

    def __init__(self, *entities, sampling: int = 20):
        self.entities = entities
        self.sampling = sampling

    def gettag(self, ff, fragments):
        tag = ff.add('Distance')
        ff.setNumber(tag, 'NumPointsPerCurve', round(self.sampling)) # gmsh 34a4d3c613
        #ff.setNumber(tag, 'Sampling', round(self.sampling))
        surfaces = set()
        curves = set()
        points = set()
        for entity in self.entities:
            tags = entity.select(fragments)
            if entity.ndims == 2:
                surfaces.update(tags)
            elif entity.ndims == 1:
                curves.update(tags)
            elif entity.ndims == 0:
                points.update(tags)
            else:
                bla
        if surfaces:
            ff.setNumbers(tag, 'SurfacesList', sorted(surfaces))
        if curves:
            ff.setNumbers(tag, 'CurvesList', sorted(curves))
        if points:
            ff.setNumbers(tag, 'PointsList', sorted(points))
        return tag


@dataclass
class Threshold(Field):

    d: Field
    dmin: float
    dmax: float
    vmin: float
    vmax: float
    sigmoid: bool = False

    def gettag(self, ff, fragments):
        tag = ff.add('Threshold')
        ff.setNumber(tag, 'InField', self.d.gettag(ff, fragments))
        ff.setNumber(tag, 'DistMin', self.dmin)
        ff.setNumber(tag, 'DistMax', self.dmax)
        ff.setNumber(tag, 'SizeMin', self.vmin)
        ff.setNumber(tag, 'SizeMax', self.vmax)
        ff.setNumber(tag, 'Sigmoid', self.sigmoid)
        return tag


@dataclass(frozen=True)
class Ball(Field):
    'Refine elements uniformly inside a circle'

    center: Tuple[float,float]
    radius: float
    inside: float
    outside: float
    thickness: float = 0.

    def gettag(self, ff, fragments):
        tag = ff.add('Ball')
        ff.setNumber(tag, 'Radius', self.radius)
        ff.setNumber(tag, 'Thickness', self.thickness)
        ff.setNumber(tag, 'VIn', self.inside)
        ff.setNumber(tag, 'VOut', self.outside)
        ff.setNumber(tag, 'XCenter', self.center[0])
        ff.setNumber(tag, 'YCenter', self.center[1])
        ff.setNumber(tag, 'ZCenter', 0)
        return tag


class Min(Field):

    def __init__(self, *fields):
        assert all(isinstance(field, Field) for field in fields)
        self.fields = fields

    def gettag(self, ff, fragments):
        tags = [field.gettag(ff, fragments) for field in self.fields]
        tag = ff.add('Min')
        ff.setNumbers(tag, 'FieldsList', tags)
        return tag


class Max(Field):

    def __init__(self, *fields):
        assert all(isinstance(field, Field) for field in fields)
        self.fields = fields

    def gettag(self, ff, fragments):
        tags = [field.gettag(ff, fragments) for field in self.fields]
        tag = ff.add('Max')
        ff.setNumbers(tag, 'FieldsList', tags)
        return tag


# vim:sw=4:sts=4:et
