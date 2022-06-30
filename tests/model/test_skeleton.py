from sleap_io.model.skeleton import Node, Edge, Skeleton


def test_skeleton_node_edge():
    # Creates a test skeleton with nodes & edges.
    skeleton = Skeleton.from_names(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
    )
    assert skeleton.nodes == [Node("head"), Node("thorax"), Node("abdomen")]
    assert skeleton.edges == [
        Edge(source=Node("head"), destination=Node("thorax")),
        Edge(source=Node("thorax"), destination=Node("abdomen")),
    ]
