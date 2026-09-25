"""Production Gold Template regressions: no procedural substitutes."""
import json
from pathlib import Path
import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree
from backend.deformer.basis_deformer import BasisDeformer
from backend.exporter.glb_exporter import GLBExporter
from backend.exporter.geometry import world_vertices
from backend.exporter.validation import inspect_glb, read_glb, world_primitives
from backend.materials.pbr import apply_materials
from backend.models import Measurements

ROOT=Path(__file__).resolve().parents[1]/'assets/templates/GT_001'


@pytest.fixture(scope='module')
def real_export(tmp_path_factory):
    measurements=Measurements(frame_width=135,lens_width=56,lens_height=37,
        bridge_width=18,temple_length=130.9,rim_thickness=1.2,color='#d0b7b1')
    engine=BasisDeformer(ROOT)
    scene,quality=engine.deform(measurements)
    scene=apply_materials(scene,measurements)
    scene.metadata['customer_tag']='preserve-this'
    path=tmp_path_factory.mktemp('real-gold')/'gold.glb'
    GLBExporter().export(scene,path,measurements,'GT_001')
    return scene,path,measurements,quality


def test_physical_scale_normals_metadata_and_independent_components(real_export):
    source,path,measurements,quality=real_export
    report=inspect_glb(path,135,tolerance_mm=0.01)
    assert report['dimension_passed']
    assert report['material_count']==2
    materials={m['name']:m for m in report['materials']}
    assert materials['Lens']['alphaMode']=='BLEND'
    assert materials['FrameMetal']['pbrMetallicRoughness']['metallicFactor']==1.0
    assert not materials['FrameMetal'].get('doubleSided',False)
    assert set(report['components'])==set(source.geometry)
    assert all(c['normals_valid'] for c in report['components'].values())
    tree,_=read_glb(path)
    nodes={n['name']:n for n in tree['nodes']}
    assert nodes['LeftLens']['mesh']!=nodes['RightLens']['mesh']
    assert nodes['LeftTemple']['mesh']!=nodes['RightTemple']['mesh']
    extras=report['scene_extras']
    assert extras['coordinate_units']=='m' and extras['measurement_units']=='mm'
    assert extras['customer_tag']=='preserve-this'
    assert 'extras' not in extras and 'vto_anchors' not in extras
    assert 'defirmation' not in json.dumps(extras)
    assert report['asset']['generator'].startswith('deformation/')
    sidecar=json.loads(path.with_suffix('.metadata.json').read_text())
    assert sidecar['anchors']==extras['anchors']
    assert sidecar['frame_width']==135  # Public dimension API stays mm.
    assert quality.passed
    assert abs(report['components']['LeftLens']['extent'][0]*1000-measurements.lens_width)<0.5
    assert abs(report['components']['LeftLens']['extent'][1]*1000-measurements.lens_height)<0.5
    assert abs((report['components']['LeftLens']['bounds'][0][0]-report['components']['RightLens']['bounds'][1][0])*1000-measurements.bridge_width)<0.5


def test_cleanup_preserves_surface_and_source_topology(real_export):
    source,path,_,_=real_export
    tree,binary=read_glb(path)
    for node,primitive,points in world_primitives(tree,binary):
        expected=world_vertices(source,node['name'])*0.001
        # Bidirectional error, not a one-sided subset check.
        assert cKDTree(expected).query(points)[0].max()<2e-8
        assert cKDTree(points).query(expected)[0].max()<2e-8
    report=inspect_glb(path)
    assert report['triangles']==sum(len(m.faces) for m in source.geometry.values())
    assert report['vertices']<sum(len(m.vertices) for m in source.geometry.values())
    with np.load(ROOT/'deformation/basis.npz',allow_pickle=False) as basis:
        assert sum(len(m.vertices) for m in source.geometry.values())==len(basis['V0'])


def test_hinge_rotation_is_local_and_does_not_move_other_components(real_export):
    _,path,_,_=real_export
    scene=trimesh.load(path,force='scene',process=False)
    before={name:world_vertices(scene,name).copy() for name in scene.geometry}
    pivot='LeftTemplePivot'
    matrix,_=scene.graph[pivot]
    point=matrix[:3,3].copy()
    scene.graph.update(frame_to=pivot,frame_from=scene.graph.base_frame,
        matrix=matrix @ trimesh.transformations.rotation_matrix(np.pi/3,[0,1,0]))
    for name in scene.geometry:
        after=world_vertices(scene,name)
        if name in {'LeftTemple','LeftFrameInsert'}:
            assert not np.allclose(after,before[name])
            np.testing.assert_allclose(np.linalg.norm(after-point,axis=1),
                                       np.linalg.norm(before[name]-point,axis=1),atol=1e-9)
        else:
            np.testing.assert_allclose(after,before[name],atol=1e-12)


def test_metre_reexport_does_not_double_scale(real_export,tmp_path):
    _,path,measurements,_=real_export
    scene=trimesh.load(path,force='scene',process=False)
    again=tmp_path/'again.glb'
    GLBExporter().export(scene,again,measurements,'GT_001')
    a=inspect_glb(path);b=inspect_glb(again,135,0.01)
    assert b['dimension_passed']
    np.testing.assert_allclose(a['scene_extras']['anchors']['LeftHinge'],b['scene_extras']['anchors']['LeftHinge'])
    for name in a['components']:
        np.testing.assert_allclose(a['components'][name]['bounds'],b['components'][name]['bounds'],atol=2e-8)


def test_nested_parent_translation_and_world_anchors_scale_once(real_export,tmp_path):
    original,_,measurements,_=real_export
    source=original.copy()
    source.metadata.pop('vto_anchors')
    edges=source.graph.to_edgelist()
    graph=trimesh.scene.transforms.SceneGraph(base_frame=source.graph.base_frame)
    graph.update(frame_to='Assembly',translation=[20,30,40])
    for parent,child,data in edges:
        graph.update(frame_from='Assembly' if parent==source.graph.base_frame else parent,
                     frame_to=child,**data)
    source.graph=graph
    path=tmp_path/'nested.glb'
    GLBExporter().export(source,path,measurements,'GT_001')
    loaded=trimesh.load(path,force='scene',process=False)
    for name in source.geometry:
        before=world_vertices(source,name)*0.001
        after=world_vertices(loaded,name)
        assert cKDTree(before).query(after)[0].max()<2e-8
    hinge=world_vertices(loaded,'LeftHinge')
    expected=(hinge.min(0)+hinge.max(0))/2
    np.testing.assert_allclose(loaded.metadata['anchors']['LeftHinge'],expected,atol=2e-8)


@pytest.mark.parametrize('width,lens_width,length',[(120,48,120),(135,55.79,155.02),(160,55.79,180)])
def test_real_deformation_width_symmetry_and_temple_independence(width,lens_width,length):
    engine=BasisDeformer(ROOT)
    defaults={p['name']:p['default'] for p in engine.parameters}
    defaults.update(frame_width=width,lens_width=lens_width,temple_length=length)
    scene,_=engine.deform(Measurements(**defaults))
    assert np.ptp(world_vertices(scene,'Frame')[:,0])==pytest.approx(width,abs=1e-7)
    left=world_vertices(scene,'LeftLens');right=world_vertices(scene,'RightLens')
    assert cKDTree(left*[-1,1,1]).query(right)[0].max()<0.001
    defaults['temple_length']=min(length+1,180)
    changed,_=engine.deform(Measurements(**defaults))
    for name in ('Frame','LeftLens','RightLens','LeftHinge','RightHinge'):
        np.testing.assert_allclose(world_vertices(scene,name),world_vertices(changed,name),atol=1e-8)
