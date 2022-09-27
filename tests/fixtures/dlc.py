"""Fixtures for dlc testing."""
import pytest
from sleap_io import Skeleton, Node, Edge
import ruamel.yaml as yaml
import os


def read_yaml(yaml_file: str) -> dict:
    """Read a yaml file into dict object

    Parameters:
    yaml_file (str): path to yaml file

    Returns:
    return_dict (dict): dict of yaml contents
    """
    with open(yaml_file, "r") as yfile:
        yml = yaml.YAML(typ="safe")
        return yml.load(yfile)


@pytest.fixture
def dlc_project_config():
    """Typical label studio file from a multi-animal DLC project (mixes mutli-animal bodyparts and unique bodyparts"""
    proj_rel_path = "tests/data/dlc/dlc_test_project"
    config = read_yaml(os.path.join(proj_rel_path, "config.yaml"))
    # fix the project path so it points to the correct location!
    config["project_path"] = os.path.realpath(proj_rel_path)

    return config
