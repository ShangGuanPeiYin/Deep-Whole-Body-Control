def test_legacy_terrain_contact_properties_match_the_triangle_mesh_contract():
    """Catch a regression where the terrain authoring receives Isaac Sim defaults."""
    from dwbc_isaaclab.tasks.widow_go1.physics_contracts import legacy_terrain_collision_offsets

    offsets = legacy_terrain_collision_offsets()
    assert offsets.contact_offset == 0.01
    assert offsets.rest_offset == 0.0
