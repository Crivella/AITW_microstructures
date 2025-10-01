"""Command line interface for wood_microstructure."""

from trogon import tui

try:
    import click as original_click
    import rich_click as click
except ImportError:
    import click
    import click as original_click

try:
    import rich
except ImportError:
    pass
else:
    import multiprocessing

    from rich.traceback import install
    install(
        show_locals=True,
        suppress=[rich, click, original_click, multiprocessing],
    )

@tui()
@click.group()
def wood_microstructure():
    pass

@wood_microstructure.group()
def postproc():
    """Utility functions"""
    pass
