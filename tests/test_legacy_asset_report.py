from tools.export_legacy_asset_report import body_property_row


def test_body_property_row_uses_canonical_inertia_diagonal():
    row = body_property_row("trunk", 5.0, (1.0, 2.0, 3.0))
    assert row == {"name": "trunk", "mass": 5.0, "inertia": [1.0, 2.0, 3.0]}
