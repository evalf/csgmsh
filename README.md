# Constructive Solid helper for Gmsh

This project is an opinionated helper tool that aims to provide a high level
Python API for 2D/3D Constructive Solid Geometry modelling, which is a very
convenient way of defining geometries using boolean operations. A typical
script looks like this:

```python
from csgmsh import mesh, shape, field

rect = shape.Rectangle()
circ = shape.Circle(center=[1,1], radius=.5)

mesh.write('demo.msh',
    groups = {
        'domain': rect - circ,
        'top-right': rect.top | circ.boundary | rect.right,
        'bottom-left': rect.left | rect.bottom,
    },
    elemsize = field.Threshold(
        d=field.Distance(rect.bottom),
        dmin=0, vmin=.01,
        dmax=.5, vmax=.1),
)
```

Context: gmsh supports GSG modelling via its OpenCASCADE kernel. Unfortunately
it is set up in a way that makes it difficult to keep track of boundary
segments, which is crucial for computational applications. Csgmsh fixes this by
assuming that the (undocumented) order in which boundary entities are provided
is stable, and keeps track of how these boundaries are subdivided via the
`BooleanFragments` operation.

This project is very much a work in progress and the API may change extensively
pre-1.0, so be sure to pin it to a version if you find a use for it.
