import pytest


def test_imports():
    try:
        import agents
        import agents.navigation.basic_agent
        import agents.navigation.global_route_planner
        import agents.tools.misc
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
