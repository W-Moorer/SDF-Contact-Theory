from sdf_contact.experiments import build_cases


def test_quick_meshes_have_at_least_2000_faces():
    for case in build_cases(quick=True):
        assert case.active_mesh.face_count >= 2000
        assert case.passive_mesh.face_count >= 2000
