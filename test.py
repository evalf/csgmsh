from nutils import function
from csgmsh import shape, mesh
import unittest
import numpy
import tempfile
import os


def nutils_mesh(**kwargs):
    from nutils.mesh import gmsh
    fid, fname = tempfile.mkstemp(suffix='.msh')
    try:
        os.close(fid) # release file for writing by gmsh (windows)
        mesh.write(output_path=fname, **kwargs)
        return gmsh(fname)
    finally:
        os.unlink(fname)


class TestShape(unittest.TestCase):

    def assertAreaCentroid(self, topo, geom, *, mass, centroid, places, degree=1):
        J = function.J(geom)
        _mass, _moment = topo.sample('gauss', degree*2).integrate([J, geom * J])
        self.assertAlmostEqual(_mass, mass, places=places)
        numpy.testing.assert_almost_equal(actual=_moment/mass, desired=centroid, decimal=places)

    def test_rectangle(self):
        'Square with 1<x<2 and 3<y<4'
        for periodic in ('', 'x', 'y', 'xy'):
            rect = shape.Rectangle(shape.Interval(1, 2, periodic='x' in periodic), shape.Interval(3, 4, periodic='y' in periodic))
            shapes = dict(dom=rect, left=rect.left, right=rect.right, bottom=rect.bottom, top=rect.top)
            topo, geom = nutils_mesh(physical_groups=shapes, mesh_size=.1)
            with self.subTest(f'interior(periodic={periodic})'):
                self.assertAreaCentroid(topo, geom, mass=1, centroid=(1.5,3.5), places=10)
            volume = 0
            if 'x' not in periodic:
                with self.subTest(f'left,right(periodic={periodic})'):
                    self.assertAreaCentroid(topo.boundary['left'], geom, mass=1, centroid=(1,3.5), places=10)
                    self.assertAreaCentroid(topo.boundary['right'], geom, mass=1, centroid=(2,3.5), places=10)
                volume += 2
            if 'y' not in periodic:
                with self.subTest(f'bottom,top(periodic={periodic})'):
                    self.assertAreaCentroid(topo.boundary['bottom'], geom, mass=1, centroid=(1.5,3), places=10)
                    self.assertAreaCentroid(topo.boundary['top'], geom, mass=1, centroid=(1.5,4), places=10)
                volume += 2
            with self.subTest(f'boundary(periodic={periodic})'):
                if volume:
                    self.assertAreaCentroid(topo.boundary, geom, mass=volume, centroid=(1.5,3.5), places=10)
                else:
                    self.assertEqual(len(topo.boundary), 0)

    def test_circle(self):
        'Circle with radius .5 centered around x=1 y=2'
        circ = shape.Circle(center=(1.,2.),radius=.5)
        topo, geom = nutils_mesh(physical_groups=dict(dom=circ), mesh_size=.1, element_order=2)
        with self.subTest('interior'):
            self.assertAreaCentroid(topo, geom, mass=.25*numpy.pi, centroid=(1,2), degree=2, places=5)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary, geom, mass=numpy.pi, centroid=(1,2), degree=2, places=5)

    def test_ellipse(self):
        'Ellipse with width .5, height 1, centered around x=1 y=2 and rotated by 30deg'
        ellipse = shape.Ellipse(center=(1.,2.),width=1.,height=2.,angle=30.)
        topo, geom = nutils_mesh(physical_groups=dict(dom=ellipse), mesh_size=.1, element_order=2)
        with self.subTest('interior'):
            self.assertAreaCentroid(topo, geom, mass=.5*numpy.pi, centroid=(1,2), degree=2, places=5)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary, geom, mass=4.84422411, centroid=(1,2), degree=2, places=4)

    def test_path(self):
        'Quarter wedge with radius 2 centered around x=3 y=2'
        path = shape.Path(vertices=((1,2),(3,4)), angles=(0.,90.))
        shapes = dict(dom=path, line=path.segment[0], arc=path.segment[1])
        topo, geom = nutils_mesh(physical_groups=shapes, mesh_size=.1, element_order=2)
        with self.subTest('interior'):
            area = numpy.pi - 2
            self.assertAreaCentroid(topo, geom, mass=area, centroid=(3-4/3/area, 2+4/3/area), degree=2, places=6)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary['line'], geom, mass=2**1.5, centroid=(2,3), places=7)
            area = numpy.pi
            self.assertAreaCentroid(topo.boundary['arc'], geom, mass=area, centroid=(3-4/area,2+4/area), places=7)

    def test_box(self):
        'Cube with 1<x<2 and 3<y<4 and 5<z<6'
        for periodic in ('', 'x', 'y', 'z', 'yz', 'xz', 'yz', 'xyz'):
            box = shape.Box(shape.Interval(1, 2, periodic='x' in periodic), shape.Interval(3, 4, periodic='y' in periodic), shape.Interval(5, 6, periodic='z' in periodic))
            shapes = dict(dom=box, left=box.left, right=box.right, bottom=box.bottom, top=box.top, front=box.front, back=box.back)
            topo, geom = nutils_mesh(physical_groups=shapes, mesh_size=.2)
            with self.subTest(f'interior(periodic={periodic})'):
                self.assertAreaCentroid(topo, geom, mass=1, centroid=(1.5,3.5,5.5), places=10)
            volume = 0
            if 'x' not in periodic:
                with self.subTest(f'left,right(periodic={periodic})'):
                    self.assertAreaCentroid(topo.boundary['left'], geom, mass=1, centroid=(1,3.5,5.5), places=10)
                    self.assertAreaCentroid(topo.boundary['right'], geom, mass=1, centroid=(2,3.5,5.5), places=10)
                volume += 2
            if 'y' not in periodic:
                with self.subTest(f'bottom,top(periodic={periodic})'):
                    self.assertAreaCentroid(topo.boundary['bottom'], geom, mass=1, centroid=(1.5,3,5.5), places=10)
                    self.assertAreaCentroid(topo.boundary['top'], geom, mass=1, centroid=(1.5,4,5.5), places=10)
                volume += 2
            if 'z' not in periodic:
                with self.subTest(f'front,back(periodic={periodic})'):
                    self.assertAreaCentroid(topo.boundary['front'], geom, mass=1, centroid=(1.5,3.5,5), places=10)
                    self.assertAreaCentroid(topo.boundary['back'], geom, mass=1, centroid=(1.5,3.5,6), places=10)
                volume += 2
            with self.subTest(f'boundary(periodic={periodic})'):
                if volume:
                    self.assertAreaCentroid(topo.boundary, geom, mass=volume, centroid=(1.5,3.5,5.5), places=10)
                else:
                    self.assertEqual(len(topo.boundary), 0)

    def test_sphere(self):
        'Sphere with radius .5 centered at (1,2,3)'
        sphere = shape.Sphere(center=(1,2,3), radius=.5)
        shapes = dict(dom=sphere, wall=sphere.boundary)
        topo, geom = nutils_mesh(physical_groups=shapes, mesh_size=.095, element_order=2)
        with self.subTest(f'interior'):
            self.assertAreaCentroid(topo, geom, mass=numpy.pi/6, centroid=(1,2,3), degree=2, places=4)
        with self.subTest(f'boundary'):
            self.assertAreaCentroid(topo.boundary, geom, mass=numpy.pi, centroid=(1,2,3), degree=2, places=4)

    def test_cylinder(self):
        'Cylinder with radius .5, with front plane in (1,2,3) and back plane in (4,5,6)'
        for periodic in False, True:
            cyl = shape.Cylinder(front=(1,2,3), back=(4,5,6), radius=.5, periodic=periodic)
            shapes = dict(dom=cyl, side=cyl.side, front=cyl.front, back=cyl.back)
            topo, geom = nutils_mesh(physical_groups=shapes, mesh_size=.1, element_order=2)
            L = 3 * numpy.sqrt(3)
            A = numpy.pi * .5**2
            with self.subTest(f'interior(periodic={periodic})'):
                self.assertAreaCentroid(topo, geom, mass=L*A, centroid=(2.5,3.5,4.5), degree=2, places=4)
            area = numpy.pi*L
            with self.subTest(f'side(periodic={periodic})'):
                self.assertAreaCentroid(topo.boundary['side'], geom, mass=area, centroid=(2.5,3.5,4.5), degree=2, places=4)
            if not periodic:
                with self.subTest(f'front,back(periodic={periodic})'):
                    self.assertAreaCentroid(topo.boundary['front'], geom, mass=A, centroid=(1,2,3), degree=2, places=4)
                    self.assertAreaCentroid(topo.boundary['back'], geom, mass=A, centroid=(4,5,6), degree=2, places=4)
                area += 2*A
            with self.subTest('boundary'):
                self.assertAreaCentroid(topo.boundary, geom, mass=area, centroid=(2.5,3.5,4.5), degree=2, places=4)

    def test_cut(self):
        'Unit square with circular cutout'
        rect = shape.Rectangle()
        circ = shape.Circle(center=(1.,1.), radius=.5)
        for simple in True, False:
            cut = rect - circ if simple else shape.Cut(rect, circ)
            shapes = dict(dom=cut, circ=circ.boundary, left=rect.left, right=rect.right, bottom=rect.bottom, top=rect.top)
            topo, geom = nutils_mesh(physical_groups=shapes, mesh_size=.1, element_order=2)
            with self.subTest('simple interior' if simple else 'occ interior'):
                volume = 1 - numpy.pi/16
                c = 1 - 11/24/volume
                self.assertAreaCentroid(topo, geom, mass=volume, centroid=(c,c), degree=2, places=5)
            with self.subTest('simple boundary' if simple else 'occ boundary'):
                self.assertAreaCentroid(topo.boundary['left'], geom, mass=1, centroid=(0,.5), places=10)
                self.assertAreaCentroid(topo.boundary['right'], geom, mass=.5, centroid=(1,.25), places=10)
                self.assertAreaCentroid(topo.boundary['bottom'], geom, mass=1, centroid=(.5,0), places=10)
                self.assertAreaCentroid(topo.boundary['top'], geom, mass=.5, centroid=(.25,1), places=10)
                self.assertAreaCentroid(topo.boundary['circ'], geom, mass=numpy.pi*.5*2/4, centroid=(1-1/numpy.pi,1-1/numpy.pi), places=6)

    def test_fuse(self):
        'Unit square fused with a circle'
        rect = shape.Rectangle()
        circ = shape.Circle(center=(1.,.5), radius=.5)
        for simple in True, False:
            fuse = rect | circ if simple else shape.Fuse(rect, circ)
            topo, geom = nutils_mesh(physical_groups=dict(dom=fuse), mesh_size=.1, element_order=2)
            with self.subTest('simple interior' if simple else 'occ interior'):
                volume = 1 + numpy.pi/8
                self.assertAreaCentroid(topo, geom, mass=volume, centroid=((7/12+numpy.pi/8)/volume,.5), degree=2, places=5)
            with self.subTest('simple boundary' if simple else 'icc boundary'):
                area = 3 + numpy.pi/2
                self.assertAreaCentroid(topo.boundary, geom, mass=area, centroid=((3/2+numpy.pi/2)/area,.5), places=5)

    def test_intersect(self):
        'Semi-circle'
        rect = shape.Rectangle()
        circ = shape.Circle(center=(1.,.5), radius=.5)
        for simple in True, False:
            intersect = rect & circ if simple else shape.Intersect(rect, circ)
            topo, geom = nutils_mesh(physical_groups=dict(dom=intersect), mesh_size=.1, element_order=2)
            with self.subTest('simple interior' if simple else 'occ interior'):
                volume = numpy.pi/8
                self.assertAreaCentroid(topo, geom, mass=volume, centroid=((numpy.pi/8-1/12)/volume,.5), degree=2, places=5)
            with self.subTest('simple boundary' if simple else 'occ boundary'):
                area = 1 + numpy.pi/2
                self.assertAreaCentroid(topo.boundary, geom, mass=area, centroid=((numpy.pi/2+.5)/area,.5), places=5)

    def test_revolved2(self):
        orig = numpy.array([2., 3.])
        line = shape.Interval(1,2)
        rev = line.revolved(origin=orig, xaxis=(1,-1), angle=90)
        # xaxis is rotated from (1,-1) to (1,1)
        shapes = dict(dom=rev, front=rev.front, back=rev.back, left=rev.side.left, right=rev.side.right)
        topo, geom = nutils_mesh(physical_groups=shapes, mesh_size=.1, element_order=2)
        with self.subTest('interior'):
            volume = numpy.pi * 3 / 4
            self.assertAreaCentroid(topo, geom, mass=volume, centroid=orig+((7/3)*numpy.sqrt(2)/volume,0), degree=2, places=6)
        with self.subTest('boundary'):
            area = numpy.pi * 3 / 2 + 2
            self.assertAreaCentroid(topo.boundary, geom, mass=area, centroid=orig+((13/2)*numpy.sqrt(2)/area,0), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['front'], geom, mass=1, centroid=orig+(numpy.sqrt(2)*3/4,-numpy.sqrt(2)*3/4), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['back'], geom, mass=1, centroid=orig+(numpy.sqrt(2)*3/4,numpy.sqrt(2)*3/4), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['left'], geom, mass=numpy.pi/2, centroid=orig+(2/numpy.pi*numpy.sqrt(2),0), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['right'], geom, mass=numpy.pi, centroid=orig+(4/numpy.pi*numpy.sqrt(2),0), degree=2, places=6)

    def test_revolved3(self):
        'Quarter annular sector with inner radius 1 and outer radius 2, symmetric in z=0'
        orig = numpy.array([2., 3., 4.])
        rect = shape.Rectangle(x=shape.Interval(1,2))
        rev = rect.revolved(origin=orig, xaxis=(-1,0,1), yaxis=(0,1,0), angle=-90)
        # x-axis is rotated from (-1,0,1) to (-1,0,-1)
        shapes = dict(dom=rev, front=rev.front, back=rev.back, left=rev.side.left, right=rev.side.right, bottom=rev.side.bottom, top=rev.side.top)
        topo, geom = nutils_mesh(physical_groups=shapes, mesh_size=.1, element_order=2)
        with self.subTest('interior'):
            volume = numpy.pi * 3 / 4
            cx = -(7/3)*numpy.sqrt(2)/volume
            self.assertAreaCentroid(topo, geom, mass=volume, centroid=orig+(cx,.5,0), degree=2, places=6)
        with self.subTest('boundary'):
            area = numpy.pi * 3 + 2
            self.assertAreaCentroid(topo.boundary, geom, mass=area, centroid=orig+(-(67/6)*numpy.sqrt(2)/area,.5,0), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['front'], geom, mass=1, centroid=orig+(-(3/4)*numpy.sqrt(2),.5,(3/4)*numpy.sqrt(2)), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['back'], geom, mass=1, centroid=orig+(-(3/4)*numpy.sqrt(2),.5,-(3/4)*numpy.sqrt(2)), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['left'], geom, mass=.5*numpy.pi, centroid=orig+(-2/numpy.pi*numpy.sqrt(2),.5,0), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['right'], geom, mass=numpy.pi, centroid=orig+(-4/numpy.pi*numpy.sqrt(2),.5,0), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['top'], geom, mass=volume, centroid=orig+(cx,1,0), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['bottom'], geom, mass=volume, centroid=orig+(cx,0,0), degree=2, places=6)

    def test_revolved_full(self):
        'Annular ring with inner radius 1 and outer radius 2'
        orig = numpy.array([2., 3., 4.])
        rect = shape.Rectangle(x=shape.Interval(1,2))
        rev = rect.revolved(origin=orig, xaxis=(-1,0,1), yaxis=(0,1,0), angle=360)
        shapes = dict(dom=rev, left=rev.side.left, right=rev.side.right, bottom=rev.side.bottom, top=rev.side.top)
        topo, geom = nutils_mesh(physical_groups=shapes, mesh_size=.1, element_order=2)
        with self.subTest('interior'):
            self.assertAreaCentroid(topo, geom, mass=3*numpy.pi, centroid=orig+(0,.5,0), degree=2, places=6)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary, geom, mass=12*numpy.pi, centroid=orig+(0,.5,0), degree=2, places=5)
            self.assertAreaCentroid(topo.boundary['left'], geom, mass=2*numpy.pi, centroid=orig+(0,.5,0), degree=2, places=5)
            self.assertAreaCentroid(topo.boundary['right'], geom, mass=4*numpy.pi, centroid=orig+(0,.5,0), degree=2, places=5)
            self.assertAreaCentroid(topo.boundary['bottom'], geom, mass=3*numpy.pi, centroid=orig+(0,0,0), degree=2, places=5)
            self.assertAreaCentroid(topo.boundary['top'], geom, mass=3*numpy.pi, centroid=orig+(0,1,0), degree=2, places=5)

    def test_pipe2(self):
        orig = numpy.array([2., 3.])
        line = shape.Interval(0, 2)
        pipe = line.extruded(segments=[[numpy.pi/2,1], [1,0], [2*numpy.pi,-.25]], origin=orig)
        shapes = dict(dom=pipe, front=pipe.front, back=pipe.back,
            left0=pipe.segment[0].left, right0=pipe.segment[0].right,
            left1=pipe.segment[1].left, right1=pipe.segment[1].right,
            left2=pipe.segment[2].left, right2=pipe.segment[2].right)
        topo, geom = nutils_mesh(physical_groups=shapes, mesh_size=.1, element_order=2)
        with self.subTest('interior'):
            V = numpy.pi * 5 + 2
            M = -8 * numpy.pi - 13, 15 * numpy.pi - 6
            self.assertAreaCentroid(topo, geom, mass=V, centroid=orig + numpy.divide(M,V), degree=2, places=6)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary['front'], geom, mass=2, centroid=orig+(1,0), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['left0'], geom, mass=numpy.pi/2, centroid=orig+(-1+2/numpy.pi, 2/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['right0'], geom, mass=numpy.pi*3/2, centroid=orig+(-1+6/numpy.pi, 6/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['left1'], geom, mass=1, centroid=orig+(-1.5,1), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['right1'], geom, mass=1, centroid=orig+(-1.5,3), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['left2'], geom, mass=numpy.pi*2, centroid=orig+(-2-8/numpy.pi, 5-8/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['right2'], geom, mass=numpy.pi, centroid=orig+(-2-4/numpy.pi,5-4/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['back'], geom, mass=2, centroid=orig+(-5,5), degree=1, places=9)

    def test_pipe3(self):
        orig = numpy.array([2., 3., 4.])
        rect = shape.Rectangle(y=shape.Interval(0, 2))
        pipe = rect.extruded(segments=[[numpy.pi/2,1,0], [1,0,0], [numpy.pi,0,.5]], origin=orig)
        shapes = dict(dom=pipe, front=pipe.front, back=pipe.back,
            left0=pipe.segment[0].left, right0=pipe.segment[0].right, bottom0=pipe.segment[0].bottom, top0=pipe.segment[0].top,
            left1=pipe.segment[1].left, right1=pipe.segment[1].right, bottom1=pipe.segment[1].bottom, top1=pipe.segment[1].top,
            left2=pipe.segment[2].left, right2=pipe.segment[2].right, bottom2=pipe.segment[2].bottom, top2=pipe.segment[2].top)
        topo, geom = nutils_mesh(physical_groups=shapes, mesh_size=.1, element_order=2)
        with self.subTest('interior'):
            V = numpy.pi * (7/2) + 2
            M = 4 * numpy.pi - 11/3, 1 - 5 * numpy.pi, 38/3 + numpy.pi * 3
            self.assertAreaCentroid(topo, geom, mass=V, centroid=orig + numpy.divide(M,V), degree=2, places=6)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary['front'], geom, mass=2, centroid=orig+(.5,1,0), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['left0'], geom, mass=numpy.pi*2, centroid=orig+(0,13/3/numpy.pi-1,13/3/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['right0'], geom, mass=numpy.pi*2, centroid=orig+(1,13/3/numpy.pi-1,13/3/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['top0'], geom, mass=numpy.pi*3/2, centroid=orig+(.5,6/numpy.pi-1,6/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['bottom0'], geom, mass=numpy.pi/2, centroid=orig+(.5,2/numpy.pi-1,2/numpy.pi), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['left1'], geom, mass=2, centroid=orig+(0,-1.5,2), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['right1'], geom, mass=2, centroid=orig+(1,-1.5,2), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['top1'], geom, mass=1, centroid=orig+(.5,-1.5,3), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['bottom1'], geom, mass=1, centroid=orig+(.5,-1.5,1), degree=1, places=9)
            self.assertAreaCentroid(topo.boundary['left2'], geom, mass=numpy.pi*2, centroid=orig+(2-4/numpy.pi,-2-4/numpy.pi,2), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['right2'], geom, mass=numpy.pi, centroid=orig+(2-2/numpy.pi,-2-2/numpy.pi,2), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['top2'], geom, mass=numpy.pi*3/4, centroid=orig+(2-28/9/numpy.pi,-2-28/9/numpy.pi,3), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['bottom2'], geom, mass=numpy.pi*3/4, centroid=orig+(2-28/9/numpy.pi,-2-28/9/numpy.pi,1), degree=2, places=6)
            self.assertAreaCentroid(topo.boundary['back'], geom, mass=2, centroid=orig+(2,-3.5,2), degree=1, places=9)

    def test_inclusion(self):
        outer = shape.Box(shape.Interval(-2, 2), shape.Interval(-2, 2), shape.Interval(-2, 2))
        inner = shape.Box(shape.Interval(-1, 1), shape.Interval(-1, 1), shape.Interval(-1, 1))
        shapes = dict(dom=outer-inner,
            innerleft=inner.left, innerright=inner.right, innertop=inner.top, innerbottom=inner.bottom, innerfront=inner.front, innerback=inner.back,
            outerleft=outer.left, outerright=outer.right, outertop=outer.top, outerbottom=outer.bottom, outerfront=outer.front, outerback=outer.back)
        topo, geom = nutils_mesh(physical_groups=shapes, mesh_size=.5, element_order=2)
        with self.subTest('interior'):
            self.assertAreaCentroid(topo, geom, mass=56, centroid=(0,0,0), degree=1, places=12)
        with self.subTest('boundary'):
            self.assertAreaCentroid(topo.boundary['innerleft'], geom, mass=4, centroid=(-1,0,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['innerright'], geom, mass=4, centroid=(1,0,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['innerbottom'], geom, mass=4, centroid=(0,-1,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['innertop'], geom, mass=4, centroid=(0,1,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['innerfront'], geom, mass=4, centroid=(0,0,-1), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['innerback'], geom, mass=4, centroid=(0,0,1), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['outerleft'], geom, mass=16, centroid=(-2,0,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['outerright'], geom, mass=16, centroid=(2,0,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['outerbottom'], geom, mass=16, centroid=(0,-2,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['outertop'], geom, mass=16, centroid=(0,2,0), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['outerfront'], geom, mass=16, centroid=(0,0,-2), degree=1, places=13)
            self.assertAreaCentroid(topo.boundary['outerback'], geom, mass=16, centroid=(0,0,2), degree=1, places=13)


class TestAxes2(unittest.TestCase):

    def test_eye(self):
        numpy.testing.assert_almost_equal(shape.Axes2.eye()[:], numpy.eye(2))

    def test_from_x(self):
        xaxis = numpy.array([-1.,1.])
        axes = shape.Axes2.from_x(xaxis)
        numpy.testing.assert_almost_equal(axes[0], xaxis / numpy.linalg.norm(xaxis))

    def test_rotate(self):
        axes = shape.Axes2.from_x([1,0]).rotate(numpy.pi/2)
        numpy.testing.assert_almost_equal(axes[0], [0,1])


class TestAxes3(unittest.TestCase):

    def test_eye(self):
        numpy.testing.assert_almost_equal(shape.Axes3.eye()[:], numpy.eye(3))

    def test_from_xy(self):
        xaxis = numpy.array([1.,2.,4.])
        yaxis = numpy.array([2.,1.,-1.]) # orthogonal
        axes = shape.Axes3.from_xy(xaxis, yaxis)
        numpy.testing.assert_almost_equal(axes[0], xaxis / numpy.linalg.norm(xaxis))
        numpy.testing.assert_almost_equal(axes[1], yaxis / numpy.linalg.norm(yaxis))

    def test_from_rotation_vector(self):
        v = numpy.array([.3,.4,.5])
        axes = shape.Axes3.from_rotation_vector(v)
        numpy.testing.assert_almost_equal(v, axes.rotation_vector)

    def test_rotate(self):
        eye = shape.Axes3.eye()
        numpy.testing.assert_almost_equal(eye.rotate([0,0,numpy.pi/2])[:], [[0,1,0],[-1,0,0],[0,0,1]])
        numpy.testing.assert_almost_equal(eye.rotate([0,numpy.pi/2,0])[:], [[0,0,-1],[0,1,0],[1,0,0]])
        numpy.testing.assert_almost_equal(eye.rotate([numpy.pi/2,0,0])[:], [[1,0,0],[0,0,1],[0,-1,0]])


class TestInterval(unittest.TestCase):

    def test_left_right(self):
        iv = shape.Interval(left=1, right=3)
        self.assertAlmostEqual(iv._left, 1)
        self.assertAlmostEqual(iv._right, 3)
        self.assertAlmostEqual(iv._center, 2)
        self.assertAlmostEqual(iv._length, 2)

    def test_center_length(self):
        iv = shape.Interval(center=1, length=4)
        self.assertAlmostEqual(iv._left, -1)
        self.assertAlmostEqual(iv._right, 3)
        self.assertAlmostEqual(iv._center, 1)
        self.assertAlmostEqual(iv._length, 4)
