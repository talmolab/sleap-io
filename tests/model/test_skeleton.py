from sleap_io.model.skeleton import Node, Edge, Skeleton


def test_skeleton_node_edge():
    skeleton = Skeleton.from_names(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
    )
    assert skeleton.name is not None
    assert skeleton.nodes == [Node("head"), Node("thorax"), Node("abdomen")]
    assert skeleton.edges == [
        Edge(source=Node("head"), destination=Node("thorax")),
        Edge(source=Node("thorax"), destination=Node("abdomen")),
    ]
