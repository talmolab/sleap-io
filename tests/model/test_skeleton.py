from sleap_io.model.skeleton import Node, Edge, Skeleton


def test_node():
    assert Node("head").name == "head"
    assert Node("left eye") != Node("right eye")
