"""Calibrated dimensions and coupled extremes on the delivered production mesh."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.deformer.basis_deformer import BasisDeformer
from backend.models import Measurements
from backend.template_library.compatibility import validate_combination, MeasurementCompatibilityError

ROOT = Path(__file__).resolve().parents[1]/'assets/templates/GT_001'


@pytest.fixture(scope='module')
def engine():
    return BasisDeformer(ROOT)


def measured(scene):
    f,l,r = (scene.geometry[n].vertices for n in ('Frame','LeftLens','RightLens'))
    return np.array([np.ptp(f[:,0]),np.ptp(l[:,0]),np.ptp(l[:,2]),l[:,0].min()-r[:,0].max()])


def test_jacobian_unit_response_and_region_isolation(engine):
    m = Measurements(frame_width=145,lens_width=55,lens_height=37,bridge_width=18,temple_length=155)
    base,_ = engine.deform(m)
    np.testing.assert_allclose(engine.regions.jacobian @ engine.calibration_matrix,np.eye(4),atol=1e-10)
    for i,name in enumerate(engine.regions.names):
        changed = m.model_copy(update={name:getattr(m,name)+1})
        result,_ = engine.deform(changed)
        np.testing.assert_allclose(measured(result)-measured(base),np.eye(4)[i],atol=1e-5)
        if name=='lens_width':
            # Changing lens width cannot shrink or displace the bridge region.
            frame=engine.scene.geometry['Frame'].vertices
            bridge=np.abs(frame[:,0])<engine.regions.x_source[3]
            np.testing.assert_allclose(result.geometry['Frame'].vertices[bridge],
                                       base.geometry['Frame'].vertices[bridge],atol=1e-7)
        if name=='frame_width':
            for part in ('LeftLens','RightLens'):
                np.testing.assert_allclose(result.geometry[part].vertices,base.geometry[part].vertices,atol=1e-7)


def test_feasible_corner_grid_and_source_assets_unchanged(engine):
    paths=[ROOT/'geometry/template.glb',ROOT/'deformation/basis.npz']
    hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    valid=invalid=0
    for fw,lw,bw,lh,tl in itertools.product((110,140,170),(40,55,70),(12,20,28),(25,50),(120,180)):
        m=Measurements(frame_width=fw,lens_width=lw,bridge_width=bw,lens_height=lh,temple_length=tl)
        if fw < 2*lw+bw+2*m.rim_thickness:
            with pytest.raises(MeasurementCompatibilityError):
                engine.deform(m)
            invalid+=1
            continue
        scene,quality=engine.deform(m)
        valid+=1
        np.testing.assert_allclose(measured(scene),[fw,lw,lh,bw],atol=0.5)
        assert quality.passed
        assert engine.regions.min_jacobian>0
        for mesh in scene.geometry.values():
            assert np.isfinite(mesh.vertices).all() and mesh.area_faces.min()>1e-10
            assert np.prod(mesh.extents)>0
        landmarks=json.loads((ROOT/'metadata/landmarks.json').read_text())['landmarks']
        for side in ('Left','Right'):
            endpoints=engine.regions.warp(np.array([landmarks[f'LM_{side}TempleRoot'],landmarks[f'LM_{side}TempleTip']]))
            assert abs(np.linalg.norm(endpoints[1]-endpoints[0])-tl)<0.5
    assert valid>20 and invalid>20
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]


@pytest.mark.parametrize('values',[(110,40,25,12,120),(170,69.8,50,28,180)])
def test_extreme_meshes_have_no_self_intersecting_optical_surfaces(engine,values):
    # Exact triangle intersection checks, not just bounding-box positivity.
    import open3d as o3d
    fw,lw,lh,bw,tl=values
    scene,_=engine.deform(Measurements(frame_width=fw,lens_width=lw,lens_height=lh,
                                      bridge_width=bw,temple_length=tl))
    for name in ('Frame','LeftLens','RightLens','LeftTemple','RightTemple'):
        mesh=scene.geometry[name].copy()
        # Only the diagnostic copy is welded, so duplicated artist seam
        # vertices do not masquerade as disconnected intersecting triangles.
        mesh.merge_vertices(merge_norm=True,digits_vertex=8)
        check=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                       o3d.utility.Vector3iVector(mesh.faces))
        assert len(check.get_self_intersecting_triangles())==0, name


@pytest.mark.parametrize('route', ['/api/deform','/api/deform/multi-view','/api/deform/measurements'])
def test_all_generation_routes_reject_coupled_input_before_processing(route):
    from backend.api.main import app
    values=dict(frame_width=135,lens_width=60,lens_height=37,bridge_width=20,temple_length=155,rim_thickness=1.2)
    client=TestClient(app)
    if route.endswith('measurements'):
        response=client.post(route,data=values)
    else:
        field='images' if route.endswith('multi-view') else 'front'
        response=client.post(route,data={'measurements':json.dumps(values)},files={field:('unused.jpg',b'invalid')})
    assert response.status_code==422
    assert '142.4' in response.json()['detail']
    bounds=response.json()['ranges']
    assert bounds['frame_width']['min']==142.4
    assert bounds['lens_width']['max']==56.3
    assert bounds['bridge_width']['max']==12.6


def test_dependent_ranges_and_ignored_binary_delivery(engine,tmp_path):
    from backend.api.main import app
    values=dict(frame_width=135,lens_width=56,lens_height=37,bridge_width=18,temple_length=155,rim_thickness=1.2)
    response=TestClient(app).post('/api/measurements/ranges',json=values)
    assert response.status_code==200
    assert response.json()['ranges']['lens_width']['max']==57.3
    with pytest.raises(FileNotFoundError,match='real GT_001 delivery asset'):
        BasisDeformer(tmp_path)
    (tmp_path/'geometry').mkdir()
    (tmp_path/'geometry/template.glb').write_text('version https://git-lfs.github.com/spec/v1')
    with pytest.raises(ValueError,match='Git LFS'):
        BasisDeformer(tmp_path)
