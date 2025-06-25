"""Conftest for ascent trajectory optimization."""

import pytest


def pytest_addoption(parser):
    """Add command line options for pytest."""
    parser.addoption(
        "--show-plots",
        action="store_true",
        default=False,
        help="Show plots during tests (default: False)",
    )


@pytest.fixture(scope="session")
def show_plots(request):
    """Fixture to determine if plots should be shown during tests."""
    return request.config.getoption("--show-plots")
