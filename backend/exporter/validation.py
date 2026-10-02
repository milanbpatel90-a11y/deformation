"""Independent GLB accessor and world-transform inspection for regression checks."""
import json
import struct
from pathlib import Path
import numpy as np
import trimesh


def read_glb(path):
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack_from('<4sII', data)
    if magic != b'glTF' or version != 2 or length != len(data):
        raise ValueError('Invalid GLB 2 header')
    offset, tree, binary = 12, None, None
    while offset < len(data):
        size, kind = struct.unpack_from('<II', data, offset)
        chunk = data[offset+8:offset+8+size]
        if len(chunk) != size or size % 4:
            raise ValueError('Invalid GLB chunk')
        if kind == 0x4E4F534A:
            tree = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        offset += size + 8
    if tree is None or binary is None:
        raise ValueError('Missing GLB chunks')
    return tree, binary


def accessor(tree, binary, index):
    item = tree['accessors'][index]
    view = tree['bufferViews'][item['bufferView']]
    dtype = np.dtype({5120:'i1',5121:'u1',5122:'<i2',5123:'<u2',5125:'<u4',5126:'<f4'}[item['componentType']])
    width = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[item['type']]
    offset = view.get('byteOffset',0) + item.get('byteOffset',0)
    stride = view.get('byteStride',dtype.itemsize*width)
    if offset + (item['count']-1)*stride + width*dtype.itemsize > len(binary):
        raise ValueError('Accessor exceeds binary buffer')
    return np.ndarray((item['count'],width),dtype=dtype,buffer=binary,offset=offset,
                      strides=(stride,dtype.itemsize)).copy()


def world_primitives(tree, binary):
    def visit(index, parent, ancestors):
        if index in ancestors:
            raise ValueError('Cyclic scene graph')
        node = tree['nodes'][index]
        if 'matrix' in node:
            local = np.asarray(node['matrix']).reshape(4,4,order='F')
        else:
            x,y,z,w = node.get('rotation',[0,0,0,1])
            local = trimesh.transformations.quaternion_matrix([w,x,y,z])
            local[:3,:3] = local[:3,:3] @ np.diag(node.get('scale',[1,1,1]))
            local[:3,3] = node.get('translation',[0,0,0])
        world = parent @ local
        if 'mesh' in node:
            for primitive in tree['meshes'][node['mesh']]['primitives']:
                positions = accessor(tree,binary,primitive['attributes']['POSITION'])
                yield node, primitive, trimesh.transform_points(positions,world)
        for child in node.get('children',[]):
            yield from visit(child,world,ancestors | {index})
    for root in tree['scenes'][tree.get('scene',0)]['nodes']:
        yield from visit(root,np.eye(4),set())


def inspect_glb(path, expected_width_mm=None, tolerance_mm=0.5):
    tree,binary=read_glb(path)
    components={}
    for node,primitive,world in world_primitives(tree,binary):
        indices=accessor(tree,binary,primitive['indices']).ravel()
        if not np.isfinite(world).all() or indices.max() >= len(world):
            raise ValueError('Invalid positions or indices')
        attrs=primitive['attributes']
        normal=accessor(tree,binary,attrs['NORMAL']) if 'NORMAL' in attrs else None
        components[node['name']]={'vertices':len(world),'triangles':len(indices)//3,
            'bounds': [world.min(0).tolist(),world.max(0).tolist()],
            'extent':np.ptp(world,axis=0).tolist(),'attributes':list(attrs),
            'normals_valid':bool(normal is not None and np.isfinite(normal).all() and
                                 np.allclose(np.linalg.norm(normal,axis=1),1,atol=1e-4))}
    width=components['Frame']['extent'][0]
    return {'path':str(path),'bytes':Path(path).stat().st_size,'asset':tree['asset'],
        'vertices':sum(v['vertices'] for v in components.values()),
        'triangles':sum(v['triangles'] for v in components.values()),
        'primitive_count':sum(len(m['primitives']) for m in tree['meshes']),
        'material_count':len(tree.get('materials',[])), 'components':components,
        'frame_width_world':width,'expected_width_mm':expected_width_mm,
        'dimension_passed':None if expected_width_mm is None else bool(abs(width*1000-expected_width_mm)<=tolerance_mm),
        'nodes':tree['nodes'],'scene_extras':tree['scenes'][tree.get('scene',0)].get('extras',{}),
        'materials':tree.get('materials',[]),'textures':tree.get('textures',[])}
