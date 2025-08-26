import os
import treelog


def _tags(dimtags, expect_dim: int):
    assert all(dim == expect_dim for dim, tag in dimtags)
    return {tag for dim, tag in dimtags}


def _generate_mesh(model, physical_groups, mesh_size) -> None:
    if isinstance(physical_groups, dict):
        physical_groups = physical_groups.items()
    else:
        seen = dict()
        for name, entity in physical_groups:
            s = seen.setdefault(entity.ndims, set())
            if name in s:
                raise ValueError(f"{name!r} occurs twice for dimension {entity.ndims}")
            s.add(name)

    shapes = [shape for _, entity in physical_groups for shape in entity.get_shapes()]
    shapes = tuple(dict.fromkeys(shapes))  # stable unique via dict

    ndims = shapes[0].ndims
    if not all(shape.ndims == ndims for shape in shapes):
        raise ValueError("mesh contains shapes of varying dimensions")

    shape_tags = [
        shape.add_to(model.occ) for shape in shapes
    ]  # create all shapes before sync
    model.occ.synchronize()  # required for getBoundary

    objectDimTags = []
    slices = []
    a = 0
    for dimtags in shape_tags:
        objectDimTags.extend(dimtags)
        b = len(objectDimTags)
        vslice = slice(a, b)
        objectDimTags.extend(model.getBoundary(dimtags, oriented=False))
        a = len(objectDimTags)
        bslice = slice(b, a)
        slices.append((vslice, bslice))
    _, fragment_map = model.occ.fragment(
        objectDimTags=objectDimTags, toolDimTags=[], removeObject=False
    )
    assert len(fragment_map) == a

    model.occ.synchronize()

    # setting fragment's removeObject=True has a tendency to remove (boundary)
    # entities that are still in use, so we remove unused entities manually
    # instead
    remove = set(objectDimTags)
    for dimtags in fragment_map:
        remove.difference_update(dimtags)
    if remove:
        model.removeEntities(sorted(remove))

    fragments = {}
    for shape, (vslice, bslice) in zip(shapes, slices):
        vtags = _tags(
            [dimtag for dimtags in fragment_map[vslice] for dimtag in dimtags], ndims
        )
        btags = [_tags(dimtags, ndims - 1) for dimtags in fragment_map[bslice]]
        assert shape.nbnd is None or shape.nbnd == len(btags)
        shape.make_periodic(model.mesh, btags)
        fragments[shape] = vtags, btags

    for name, item in physical_groups:
        tag = model.addPhysicalGroup(item.ndims, sorted(item.select(fragments)))
        model.setPhysicalName(dim=item.ndims, tag=tag, name=name)

    if not isinstance(mesh_size, (int, float)):
        ff = model.mesh.field
        ff.setAsBackgroundMesh(mesh_size.gettag(ff, fragments))

    model.mesh.generate(ndims)


def _write(output_path: str, physical_groups, mesh_size, **mesh_options) -> None:
    import gmsh

    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 1)
    mesh_options.setdefault("binary", 1)
    mesh_options.setdefault("characteristic_length_extend_from_boundary", 0)
    mesh_options.setdefault("characteristic_length_from_points", 0)
    mesh_options.setdefault("characteristic_length_from_curvature", 0)
    if isinstance(mesh_size, (int, float)):
        mesh_options.setdefault("mesh_size_min", mesh_size)
        mesh_options.setdefault("mesh_size_max", mesh_size)
    for name, value in mesh_options.items():
        gmsh.option.setNumber("Mesh." + name.title().replace("_", ""), value)
    _generate_mesh(gmsh.model, physical_groups, mesh_size)
    gmsh.plugin.setNumber("AnalyseMeshQuality", "JacobianDeterminant", 1)
    gmsh.plugin.setNumber("AnalyseMeshQuality", "IGEMeasure", 1)
    gmsh.plugin.setNumber("AnalyseMeshQuality", "ICNMeasure", 1)
    gmsh.plugin.run("AnalyseMeshQuality")
    gmsh.write(output_path)
    gmsh.finalize()


def write(*, fork: bool = hasattr(os, "fork"), **kwargs) -> None:
    """Create .msh file based on Constructive Solid Geometry description.

    Arguments
    ---------
    output_path
        Path of the output .msh file.
    physical_groups
        Dictionary of physical name -> Shape objects.
    mesh_size
        Field object for spatially varying element size, or a float for
        constant element size.
    **
        Any mesh option can be specified as keyword arguments, with the
        original camel case turned to snake case (e.g. element_order instead of
        ElementOrder). See https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options
        for the full list of options.
    """

    if not fork:
        return _write(**kwargs)

    r, w = os.pipe()

    if os.fork():  # parent process
        os.close(w)
        with os.fdopen(r, "r", -1) as lines:
            for line in lines:
                level, sep, msg = line.partition(": ")
                level = level.rstrip().lower()
                if level in ("debug", "info", "warning", "error"):
                    getattr(treelog, level)(msg.rstrip())
        if os.wait()[1]:
            raise RuntimeError(
                "gmsh failed (for more information consider running with fork=False)"
            )

    else:  # child process
        os.close(r)
        os.dup2(w, 1)
        os.dup2(w, 2)
        try:
            _write(**kwargs)
        except Exception as e:
            print("Error:", e)
            os._exit(1)
        except:  # noqa: E722
            os._exit(1)
        else:
            os._exit(0)


# vim:sw=4:sts=4:et
