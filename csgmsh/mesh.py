import os, treelog


def _tags(dimtags, expect_dim: int):
    assert all(dim == expect_dim for dim, tag in dimtags)
    return {tag for dim, tag in dimtags}


def generate_mesh(model, groups, elemsize) -> None:

    if isinstance(groups, dict):
        groups = groups.items()
    else:
        seen = dict()
        for name, entity in groups:
            s = seen.setdefault(entity.ndims, set())
            if name in s:
                raise ValueError(f'{name!r} occurs twice for dimension {entity.ndims}')
            s.add(name)

    shapes = [shape for _, entity in groups for shape in entity.get_shapes()]
    shapes = tuple(dict.fromkeys(shapes)) # stable unique via dict

    ndims = shapes[0].ndims
    if not all(shape.ndims == ndims for shape in shapes):
        raise ValueError('mesh contains shapes of varying dimensions')

    shape_tags = [shape.add_to(model.occ) for shape in shapes] # create all shapes before sync
    model.occ.synchronize() # required for getBoundary

    objectDimTags: List[Tuple[int, Tag]] = []
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
    _, fragment_map = model.occ.fragment(objectDimTags=objectDimTags, toolDimTags=[], removeObject=False)
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
        vtags = _tags([dimtag for dimtags in fragment_map[vslice] for dimtag in dimtags], ndims)
        btags = [_tags(dimtags, ndims-1) for dimtags in fragment_map[bslice]]
        assert shape.nbnd is None or shape.nbnd == len(btags)
        shape.make_periodic(model.mesh, btags)
        fragments[shape] = vtags, btags

    for name, item in groups:
        tag = model.addPhysicalGroup(item.ndims, sorted(item.select(fragments)))
        model.setPhysicalName(dim=item.ndims, tag=tag, name=name)

    if not isinstance(elemsize, (int, float)):
        ff = model.mesh.field
        ff.setAsBackgroundMesh(elemsize.gettag(ff, fragments))

    model.mesh.generate(ndims)


def _write(fname: str, groups, elemsize, order: int) -> None:
    import gmsh
    gmsh.initialize(interruptible=False)
    gmsh.option.setNumber('General.Terminal', 1)
    gmsh.option.setNumber('Mesh.Binary', 1)
    gmsh.option.setNumber('Mesh.ElementOrder', order)
    if isinstance(elemsize, (int, float)):
        gmsh.option.setNumber('Mesh.MeshSizeMin', elemsize)
        gmsh.option.setNumber('Mesh.MeshSizeMax', elemsize)
    else:
        gmsh.option.setNumber('Mesh.CharacteristicLengthExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.CharacteristicLengthFromPoints', 0)
        gmsh.option.setNumber('Mesh.CharacteristicLengthFromCurvature', 0)
    generate_mesh(gmsh.model, groups, elemsize)
    gmsh.write(fname)
    gmsh.finalize()


def write(fname: str, groups, elemsize, order: int = 1, fork: bool = hasattr(os, 'fork')) -> None:
    'Create .msh file based on Constructive Solid Geometry description.'

    if not fork:
        return _write(fname, groups, elemsize, order)

    r, w = os.pipe()

    if os.fork(): # parent process

        os.close(w)
        for line in os.fdopen(r, 'r', -1):
            level, sep, msg = line.partition(': ')
            level = level.rstrip().lower()
            if level in ('debug', 'info', 'warning', 'error'):
                getattr(treelog, level)(msg.rstrip())
        if os.wait()[1]:
            raise RuntimeError('gmsh failed (for more information consider running with fork=False)')

    else: # child process

        os.close(r)
        os.dup2(w, 1)
        os.dup2(w, 2)
        try:
            _write(fname, groups, elemsize, order)
        except Exception as e:
            print('Error:', e)
            os._exit(1)
        except:
            os._exit(1)
        else:
            os._exit(0)


# vim:sw=4:sts=4:et
