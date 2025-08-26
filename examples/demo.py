from csgmsh import mesh, shape, field

rect = shape.Rectangle()
circ = shape.Circle(center=[1, 1], radius=0.5)

mesh.write(
    "demo.msh",
    groups={
        "domain": rect - circ,
        "top-right": rect.top | circ.boundary | rect.right,
        "bottom-left": rect.left | rect.bottom,
    },
    elemsize=field.Threshold(
        d=field.Distance(rect.bottom), dmin=0, vmin=0.01, dmax=0.5, vmax=0.1
    ),
)
