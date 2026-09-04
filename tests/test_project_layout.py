from importlib import import_module


def test_project_exposes_both_versioned_packages():
    task_package = import_module("dwbc_isaaclab")
    rl_package = import_module("dwbc_rsl_rl")

    assert task_package.__version__ == "0.1.0"
    assert rl_package.__version__ == "0.1.0"
