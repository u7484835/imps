"""impsy.impsy: provides entry point main() to impsy.""" 


import click
from .dataset import dataset
from .train import train
from .interaction import run
from .tests import test_mdrnn
from .convert import convert
from .convert2 import convert2
from .convert0 import convert0


@click.group()
def cli():
    pass


def main():
    """The entry point function for IMPSY, this just passes through the interfaces for each command"""
    cli.add_command(dataset)
    cli.add_command(train)
    cli.add_command(convert)
    cli.add_command(convert0)
    cli.add_command(convert2)
    cli.add_command(run)
    cli.add_command(test_mdrnn)
    # runs the command line interface
    cli()