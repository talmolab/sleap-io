from sleap_io import Node, Edge, Skeleton


def test_skeleton_node_edge():
    # Creates a test skeleton with nodes & edges.
    skeleton = Skeleton(
        nodes=[Node("head"), Node("thorax"), Node("abdomen")],
        edges=[
            Edge(source=Node("head"), destination=Node("thorax")),
            Edge(source=Node("thorax"), destination=Node("abdomen")),
        ],
    )
    assert skeleton.nodes == [Node("head"), Node("thorax"), Node("abdomen")]
    assert skeleton.edges == [
        Edge(source=Node("head"), destination=Node("thorax")),
        Edge(source=Node("thorax"), destination=Node("abdomen")),
    ]
