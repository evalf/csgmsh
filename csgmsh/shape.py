from typing import Tuple, Optional, Iterable, Sequence, NamedTuple
from abc import ABC, abstractmethod
import numpy, re, math, functools


class Axes2:
    'representation of 2D, positive, orthonormal axes'

    @classmethod
    def eye(cls) -> 'Axes2':
        return cls(0.)

    @classmethod
    def from_x(cls, xaxis) -> 'Axes2':
        cosθ, sinθ = xaxis
        return cls(math.atan2(sinθ, cosθ))

    def __init__(self, rotation: float):
        self._θ = rotation

    def __len__(self) -> int:
        return 2

    def __getitem__(self, s):
        sin = numpy.sin(self._θ)
        cos = numpy.cos(self._θ)
        return numpy.array([[cos, sin], [-sin, cos]][s])

    def rotate(self, angle: float) -> 'Axes2':
        return Axes2(self._θ + angle)

    def as_3(self, origin) -> Tuple:
        return Axes3.from_rotation_vector((0, 0, self._θ)), (*origin, 0)


class Axes3:
    'representation of 3D, positive, orthonormal axes'
    # immutable representation of a unit quaternion, see https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    @classmethod
    def eye(cls) -> 'Axes3':
        return cls(1., (0., 0., 0.))

    @classmethod
    def from_rotation_vector(cls, v) -> 'Axes3':
        v_ = numpy.asarray(v)
        θ = numpy.linalg.norm(v_) / 2
        return cls(numpy.cos(θ), v_ * (-.5 * numpy.sinc(θ/numpy.pi)))

    @classmethod
    def from_xy(cls, xaxis, yaxis) -> 'Axes3':
        Q = numpy.array([xaxis, yaxis, numpy.cross(xaxis, yaxis)])

        K = numpy.empty([4,4])
        K[1:,1:] = Q + Q.T
        K[0,1:] = K[1:,0] = Q[[2,0,1],[1,2,0]] - Q[[1,2,0],[2,0,1]]
        K[0,0] = 2 * numpy.trace(Q)

        eigval, eigvec = [v.T[-1] for v in numpy.linalg.eigh(K)]
        # is xaxis and yaxis are orthonormal then (eigval-K[0,0]/2)/3 == 1

        return cls(eigvec[0], eigvec[1:])

    def __init__(self, w: float, v):
        self._w = w
        self._v = numpy.array(v)

    def __len__(self) -> int:
        return 3

    def __getitem__(self, s):
        I = numpy.eye(3)[s]
        return 2 * ((.5 - self._v @ self._v) * I + self._v[s, numpy.newaxis] * self._v + numpy.cross(I, self._w * self._v))

    def rotate(self, rotvec) -> 'Axes3':
        other = Axes3.from_rotation_vector(rotvec)
        # hamilton product
        return Axes3(self._w * other._w - self._v @ other._v, self._w * other._v + other._w * self._v + numpy.cross(self._v, other._v))

    def as_3(self, origin) -> Tuple:
        return self, origin

    @property
    def rotation_axis(self):
        return tuple(self._v)

    @property
    def rotation_angle(self) -> float:
        return -2 * math.atan2(numpy.linalg.norm(self._v), self._w)

    @property
    def rotation_vector(self):
        n = numpy.linalg.norm(self._v)
        return self._v * (n and -2 * math.atan2(n, self._w) / n)


class Orientation:

    def __init__(self, origin, **kwargs):
        ndims = len(origin)
        args = ', '.join(sorted(kwargs))
        if args == 'axes':
            axes = kwargs['axes']
            assert len(axes) == ndims
        elif ndims == 2 and not args:
            axes = Axes2.eye()
        elif ndims == 3 and not args:
            axes = Axes3.eye()
        elif ndims == 2 and args == 'rotation':
            axes = Axes2(kwargs['rotation'])
        elif ndims == 2 and args == 'xaxis':
            axes = Axes2.from_x(kwargs['xaxis'])
        elif ndims == 3 and args == 'xaxis, yaxis':
            axes = Axes3.from_xy(kwargs['xaxis'], kwargs['yaxis'])
        elif ndims == 3 and args == 'rotvec':
            axes = Axes3.from_rotation_vector(kwargs['rotvec'])
        else:
            raise ValueError(f'cannot create {ndims}D orientation based on arguments {args}')
        self.origin = numpy.array(origin)
        self.ndims = ndims
        self.axes = axes

    def orient(self, occ, dimtags) -> None:
        'position a shape in 3D space'

        axes, origin = self.axes.as_3(self.origin)
        if any(origin):
            occ.translate(dimtags, *origin)
        if axes.rotation_angle:
            occ.rotate(dimtags, *origin, *axes.rotation_axis, -axes.rotation_angle)


class Affine(NamedTuple):
    xx: float; xy: float; xz: float; dx: float
    yx: float; yy: float; yz: float; dy: float
    zx: float; zy: float; zz: float; dz: float
    u1: float; u2: float; u3: float; u4: float
    @classmethod
    def shift(cls, dx: float, dy: float, dz: float):
        return cls(1., 0., 0., dx, 0., 1., 0., dy, 0., 0., 1., dz, 0., 0., 0., 1.)


def overdimensioned(*args):
    ncoeffs = len(args[0]) - 1
    matrix = []
    values = []
    for v, *coeffs in args:
        assert len(coeffs) == ncoeffs
        if v is not None:
            matrix.append(coeffs)
            values.append(v)
    if len(values) != ncoeffs:
        raise ValueError(f'exactly {ncoeffs} arguments should be specified')
    mat = numpy.linalg.solve(matrix, values).T
    return [mat @ coeffs for v, *coeffs in args]


class Entity(ABC):

    def __init__(self, ndims: int):
        self.ndims = ndims

    @abstractmethod
    def get_shapes(self) -> Iterable['Shape']:
        ...

    @abstractmethod
    def select(self, fragments):
        ...

    def __sub__(self, other):
        return SetOp(self, other, set.__sub__, '-')

    def __and__(self, other):
        return SetOp(self, other, set.__and__, '&')

    def __or__(self, other):
        return SetOp(self, other, set.__or__, '|')

    def __repr__(self):
        return f'{type(self).__name__}({self.ndims}D)'


class SetOp(Entity):

    def __init__(self, a, b, op, sym):
        assert a.ndims == b.ndims
        self.a = a
        self.b = b
        self.op = op
        self.sym = sym
        super().__init__(a.ndims)

    def get_shapes(self):
        yield from self.a.get_shapes()
        yield from self.b.get_shapes()

    def select(self, fragments):
        return self.op(self.a.select(fragments), self.b.select(fragments))

    def __repr__(self):
        return f'{self.a:!r}{self.sym}{self.b:!r}'


class Shape(Entity):

    def __init__(self, ndims: int, nbnd: Optional[int], periodicity: Iterable[Tuple[str,str,Affine]] = ()):
        self.nbnd = nbnd
        self._periodicity = tuple(periodicity)
        super().__init__(ndims)

    def get_shapes(self):
        yield self

    def extruded(self, segments: Sequence[Tuple[float,float,float]], **orientation_kwargs) -> 'Pipe':
        '''Extruded 2D shape along 3D wire.

        The 2D `shape` is positioned in 3D space by translating to `origin` and
        rotating in the directions of `xaxis` and `yaxis`. The shape is
        subsequently extruded via a number of sections, each of which defines a
        length, an x-curvature and a y-curvature. If both curvatures are zero then
        the shape is linearly extruded over a distance of `length`. Otherwise, the
        vector (xcurv, ycurv) defines the axis of rotation in the 2D plane, and its
        length the curvature. Rotation follows the right-hand rule.'''

        orientation_kwargs.setdefault('origin', numpy.zeros(self.ndims+1))
        return Pipe(self, segments, Orientation(**orientation_kwargs))

    def revolved(self, angle: float = 360., **orientation_kwargs) -> 'Revolved':
        ''''Revolve 2D shape.

        The 2D `shape` is positioned in 3D space by translating to `origin` and
        rotating in the directions of `xaxis` and `yaxis`. The shape is
        subsequently rotated over its y-axis to form a (partially) revolved body.
        In case the rotation `angle` is less than 360 degrees then boundary groups
        'front' and 'back' are added to the 2D shape's existing boundaries;
        otherwise the revolved shape defines only the 2D boundary groups.'''

        orientation_kwargs.setdefault('origin', numpy.zeros(self.ndims+1))
        return Revolved(self, angle, Orientation(**orientation_kwargs))

    def select(self, fragments):
        vtags, btags = fragments[self]
        return vtags

    def make_periodic(self, mesh, btags) -> None:
        for a, b, affinetrans in self._periodicity:
            mesh.setPeriodic(self.ndims-1, sorted(btags[a]), sorted(btags[b]), affinetrans)

    @property
    def boundary(self) -> 'Boundary':
        return Boundary(self)

    @abstractmethod
    def add_to(self, occ):
        ...


class Interval(Shape):

    def __init__(self, left: Optional[float] = None, right: Optional[float] = None, center: Optional[float] = None, length: Optional[float] = None, periodic: bool = False):
        self._left, self._right, self._center, self._length = overdimensioned((left, 1, 0), (right, 0, 1), (center, .5, .5), (length, -1, 1))
        if self._length <= 0:
            raise ValueError('negative interval')
        self.periodic = periodic
        super().__init__(ndims=1, nbnd=2)

    def add_to(self, occ):
        p1, p2 = [occ.addPoint(x, 0, 0) for x in (self._left, self._right)]
        return (1, occ.addLine(p1, p2)),

    @property
    def left(self):
        return BoundarySegment(self, 0)

    @property
    def right(self):
        return BoundarySegment(self, 1)


class Point(Shape):

    def __init__(self, *p):
        self.p = p
        super().__init__(ndims=0, nbnd=0)

    def add_to(self, occ):
        p = occ.addPoint(*self.p, *[0]*(3-len(self.p)))
        return (0, p),


class Line(Shape):

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        assert len(p1) == len(p2)
        super().__init__(ndims=1, nbnd=2)

    def add_to(self, occ):
        p1, p2 = [occ.addPoint(*p, *[0]*(3-len(p))) for p in (self.p1, self.p2)]
        return (self.ndims, occ.addLine(p1, p2)),


class Rectangle(Shape):
    'Rectangular domain'

    def __init__(self, x: Interval = Interval(0, 1), y: Interval = Interval(0, 1)):
        self.x = x._left
        self.dx = x._length
        self.y = y._left
        self.dy = y._length
        periodicity = [(b, a, Affine.shift(*d*iv._length)) for d, (a, iv, b) in zip(numpy.eye(3), [(3, x, 1), (0, y, 2)]) if iv.periodic]
        super().__init__(ndims=2, nbnd=4, periodicity=periodicity)

    def add_to(self, occ):
        return (2, occ.addRectangle(x=self.x, y=self.y, z=0., dx=self.dx, dy=self.dy)),

    @property
    def bottom(self):
        return BoundarySegment(self, 0)

    @property
    def right(self):
        return BoundarySegment(self, 1)

    @property
    def top(self):
        return BoundarySegment(self, 2)

    @property
    def left(self):
        return BoundarySegment(self, 3)


class Circle(Shape):
    'Circular domain'

    def __init__(self, center = (0., 0.), radius: float = 1.):
        self.center = center
        self.radius = radius
        super().__init__(ndims=2, nbnd=1)

    def add_to(self, occ):
        return (2, occ.addDisk(*self.center, 0., rx=self.radius, ry=self.radius)),


class Ellipse(Shape):
    'Ellipsoidal domain'

    def __init__(self, center = (0., 0.), width: float = 1., height: float = .5, angle: float = 0.):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        super().__init__(ndims=2, nbnd=1)

    def add_to(self, occ):
        height, width, angle = (self.height, self.width, self.angle) if self.width > self.height \
                          else (self.width, self.height, self.angle + 90)
        tag = occ.addDisk(*self.center, 0., rx=width/2, ry=height/2)
        occ.rotate([(2, tag)], *self.center, 0., ax=0., ay=0., az=1., angle=angle*numpy.pi/180)
        return (2, tag),


class Path(Shape):
    'Arbitrarily shaped domain enclosed by straight and curved boundary segments'

    def __init__(self, vertices, curvatures: Optional[Tuple[float,...]] = None):
        self.vertices = tuple(vertices)
        assert all(len(v) == 2 for v in self.vertices)
        self.curvatures = numpy.array(curvatures) if curvatures else numpy.zeros(len(self.vertices))
        assert len(self.curvatures) == len(self.vertices)
        super().__init__(ndims=2, nbnd=len(vertices))

    def add_to(self, occ):
        points = [(v, occ.addPoint(*v, 0.)) for v in self.vertices]
        points.append(points[0])
        lines = [occ.addLine(p1, p2) if k == 0
            else occ.addCircleArc(p1, occ.addPoint(*self._center(v1, v2, 1/k), 0.), p2)
                for k, (v1, p1), (v2, p2) in zip(self.curvatures, points[:-1], points[1:])]
        loop = occ.addCurveLoop(lines)
        return (2, occ.addPlaneSurface([loop])),

    @staticmethod
    def _center(v1, v2, r: float, nudge: float = 1e-7):
        cx, cy = numpy.add(v1, v2) / 2
        dx, dy = numpy.subtract(v1, v2) / 2
        r2 = dx**2 + dy**2
        D2 = r**2 / r2 - 1 + nudge
        assert D2 > 0, f'invalid arc radius: {r} < {numpy.sqrt(r2)}'
        D = numpy.copysign(numpy.sqrt(D2), r)
        return cx + dy * D, cy - dx * D,

    @property
    def segment(self):
        return [BoundarySegment(self, i) for i in range(len(self.vertices))]


class Box(Shape):
    'Box'

    def __init__(self, x: Interval = Interval(0, 1), y: Interval = Interval(0, 1), z: Interval = Interval(0, 1)):
        self.x = x._left
        self.dx = x._length
        self.y = y._left
        self.dy = y._length
        self.z = z._left
        self.dz = z._length
        periodicity = [(b, a, Affine.shift(*d*iv._length)) for d, (a, iv, b) in zip(numpy.eye(3), [(0, x, 1), (2, y, 3), (4, z, 5)]) if iv.periodic]
        super().__init__(ndims=3, nbnd=6, periodicity=periodicity)

    def add_to(self, occ):
        return (3, occ.addBox(x=self.x, y=self.y, z=self.z, dx=self.dx, dy=self.dy, dz=self.dz)),

    @property
    def left(self):
        return BoundarySegment(self, 0)

    @property
    def right(self):
        return BoundarySegment(self, 1)

    @property
    def bottom(self):
        return BoundarySegment(self, 2)

    @property
    def top(self):
        return BoundarySegment(self, 3)

    @property
    def front(self):
        return BoundarySegment(self, 4)

    @property
    def back(self):
        return BoundarySegment(self, 5)


class Sphere(Shape):
    'Sphere'

    def __init__(self, center = (0., 0., 0., ), radius: float = 1.):
        self.center = center
        self.radius = radius
        super().__init__(ndims=3, nbnd=1)

    def add_to(self, occ):
        return (3, occ.addSphere(*self.center, self.radius)),


class Cylinder(Shape):
    'Cylinder'

    def __init__(self, front = (0.,0.,0.), back = (0.,0.,1.), radius: float = 1., periodic: bool = False):
        self.center = front
        self.axis = back[0] - front[0], back[1] - front[1], back[2] - front[2]
        self.radius = radius
        super().__init__(ndims=3, nbnd=3, periodicity=[(1, 2, Affine.shift(*self.axis))] if periodic else ())

    def add_to(self, occ):
        return (3, occ.addCylinder(*self.center, *self.axis, self.radius)),

    @property
    def side(self):
        return BoundarySegment(self, 0)

    @property
    def back(self):
        return BoundarySegment(self, 1)

    @property
    def front(self):
        return BoundarySegment(self, 2)


class Cut(Shape):

    def __init__(self, shape1: Shape, shape2: Shape):
        self.shape1 = shape1
        self.shape2 = shape2
        assert shape2.ndims == shape1.ndims
        super().__init__(shape1.ndims, nbnd=None)

    def add_to(self, occ):
        return occ.cut(objectDimTags=self.shape1.add_to(occ), toolDimTags=self.shape2.add_to(occ))[0]


class Intersect(Shape):

    def __init__(self, *shapes):
        self.shapes = shapes
        ndims = shapes[0].ndims
        assert all(shape.ndims == ndims for shape in shapes)
        super().__init__(ndims, nbnd=None)

    def add_to(self, occ):
        obj, *tool = [tag for shape in self.shapes for tag in shape.add_to(occ)]
        return occ.intersect(objectDimTags=[obj], toolDimTags=tool)[0]


class Fuse(Shape):

    def __init__(self, *shapes):
        self.shapes = shapes
        ndims = shapes[0].ndims
        assert all(shape.ndims == ndims for shape in shapes)
        super().__init__(ndims, nbnd=None)

    def add_to(self, occ):
        obj, *tool = [tag for shape in self.shapes for tag in shape.add_to(occ)]
        return occ.fuse(objectDimTags=[obj], toolDimTags=tool)[0]


class Revolved(Shape):

    def __init__(self, shape: Shape, angle: float, orientation: Orientation):
        assert orientation.ndims == shape.ndims + 1
        self.shape = shape
        self.orientation = orientation
        self.angle = float(angle) * numpy.pi / 180
        self.partial = self.angle < 2 * numpy.pi
        super().__init__(ndims=orientation.ndims, nbnd=None if shape.nbnd is None else shape.nbnd + 2 if self.partial else shape.nbnd)

    def add_to(self, occ):
        front = self.shape.add_to(occ)
        self.orientation.orient(occ, front)
        axes, origin = self.orientation.axes.as_3(self.orientation.origin)
        iaxis = {2: 2, 3: 1}[self.ndims] # TODO: allow variation of revolution axis in 3D
        dimtags = occ.revolve(front, *origin, *axes[iaxis], -self.angle)
        return [(dim, tag) for dim, tag in dimtags if dim == self.ndims]

    @property
    def front(self):
        assert self.partial
        return BoundarySegment(self, -1)

    @property
    def back(self):
        assert self.partial
        return BoundarySegment(self, 0)

    @property
    def side(self):
        s = slice(1, -1) if self.partial else slice(None)
        return WrappedBoundary(self, self.shape, lambda btags: btags[s])


class Pipe(Shape):

    def __init__(self, shape: Shape, segments: Sequence[Tuple[float,float,float]], orientation: Orientation):
        assert orientation.ndims == shape.ndims + 1
        self.shape = shape
        self.nsegments = len(segments)

        self.front_orientation = orientation
        vertices = [orientation.origin]
        midpoints = []
        for length, *curvature in segments:
            if not any(curvature):
                midpoints.append(None)
                orientation = Orientation(orientation.origin + length * orientation.axes[-1], axes=orientation.axes)
            else:
                if shape.ndims == 1:
                    kx, = curvature
                    radius = numpy.array([-1/kx])
                    rotation = kx * length
                elif shape.ndims == 2:
                    kx, ky = curvature
                    radius = numpy.divide([ky, -kx], kx**2 + ky**2)
                    rotation = numpy.multiply(curvature, length) @ orientation.axes[:-1]
                else:
                    raise NotImplementedError
                center = orientation.origin + radius @ orientation.axes[:-1]
                midpoints.append(center)
                axes = orientation.axes.rotate(rotation)
                orientation = Orientation(center - radius @ axes[:-1], axes=axes)
            vertices.append(orientation.origin)
        self.midpoints = tuple(midpoints)
        self.vertices = tuple(vertices)
        self.back_orientation = orientation

        super().__init__(ndims=orientation.ndims, nbnd=None if shape.nbnd is None else 2 + shape.nbnd * self.nsegments)

    def add_to(self, occ):
        z = (0,) * (3 - self.ndims)
        points = [occ.addPoint(*v, *z) for v in self.vertices]
        segments = [occ.addLine(p1, p2) if v is None
               else occ.addCircleArc(p1, occ.addPoint(*v, *z), p2) for p1, v, p2 in zip(points, self.midpoints, points[1:])]
        wire_tag = occ.addWire(segments)
        front = self.shape.add_to(occ)
        self.front_orientation.orient(occ, front)
        return occ.addPipe(front, wire_tag)

    @property
    def front(self):
        return BoundarySegment(self, 0 if self.ndims == 3 else 1)

    @property
    def back(self):
        return BoundarySegment(self, -1)

    @property
    def segment(self):
        if self.ndims == 3:
            def subset(i, btags):
                btags = btags[1:-1]
                assert len(btags) % self.nsegments == 0
                return btags[i::self.nsegments]
        else:
            def subset(i, btags):
                btags = btags[:1] + btags[2:-1]
                n, m = divmod(len(btags), self.nsegments)
                assert m == 0
                return btags[i*n:][:n]
        return [WrappedBoundary(self, self.shape, functools.partial(subset, i)) for i in range(self.nsegments)]


class Boundary(Entity):

    def __init__(self, parent: Shape):
        self.parent = parent
        super().__init__(parent.ndims - 1)

    def get_shapes(self):
        return self.parent.get_shapes()

    def select(self, fragments):
        vtags, btags = fragments[self.parent]
        return set.union(*btags)


class BoundarySegment(Entity):

    def __init__(self, parent: Shape, item: str):
        self.parent = parent
        self.item = item
        super().__init__(parent.ndims - 1)

    def get_shapes(self):
        return self.parent.get_shapes()

    def select(self, fragments):
        vtags, btags = fragments[self.parent]
        return btags[self.item]


class WrappedBoundary(Entity):

    def __init__(self, parent: Shape, wrapped: Shape, subset):
        self.parent = parent
        self.wrapped = wrapped
        self.subset = subset
        super().__init__(parent.ndims - 1)

    def __getattr__(self, attr):
        return WrappedBoundarySegment(self.parent, getattr(self.wrapped, attr), self.subset)

    def get_shapes(self):
        return self.parent.get_shapes()

    def select(self, fragments):
        vtags, btags = fragments[self.parent]
        return set.union(*self.subset(btags))


class WrappedBoundarySegment(Entity):

    def __init__(self, parent: Shape, wrapped: Shape, subset):
        self.parent = parent
        self.wrapped = wrapped
        self.subset = subset
        super().__init__(parent.ndims - 1)

    def get_shapes(self):
        return self.parent.get_shapes()

    def select(self, fragments):
        vtags, btags = fragments[self.parent]
        fragments = {self.wrapped.parent: (None, self.subset(btags))}
        return self.wrapped.select(fragments)


class Skeleton(Entity):

    def get_shapes(self):
        return ()

    def select(self, fragments):
        return set.union(*[btag for vtags, btags in fragments.values() for btag in btags])


# vim:sw=4:sts=4:et
