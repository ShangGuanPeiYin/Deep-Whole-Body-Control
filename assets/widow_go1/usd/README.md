# Generated USD

Run the importer under the pinned environment:

```bash
conda run --no-capture-output -n isaac-lab python assets/widow_go1/import_config/widow_go1_urdf_import.py --headless
```

The importer preserves the floating base, merges fixed joints to match the legacy `collapse_fixed_joints=True`, and disables imported joint drives because task actuators define control gains explicitly.
After conversion it authors the versioned `legacy_mass_properties.json` values into the physics layer. These are the properties observed from Isaac Gym Preview 4 after the legacy task requested `recomputeInertia=True`; keeping them explicit avoids converter-version-dependent inertia drift.

Isaac Sim may warn that `wx250s_ee_gripper_link/visuals` has an unresolved visual reference. This is expected from the frozen URDF: that link is an inertial end-effector frame with no `<visual>` or `<collision>` element. The audited rigid-body and collider counts remain 27 and 25 respectively.
