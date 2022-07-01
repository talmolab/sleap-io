from sleap_io import Node, Edge, Skeleton


def test_skeleton_node_edge():
<<<<<<< HEAD
    # creates a test skeleton with nodes & edges
    skeleton = Skeleton.from_names(
        nodes=["head", "thorax", "abdomen"],
        edges=[("head", "thorax"), ("thorax", "abdomen")],
    )
=======
    # Creates a test skeleton with nodes and edges.
    skeleton = Skeleton(
        nodes=[Node("head"), Node("thorax"), Node("abdomen")],
        edges=[
            Edge(source=Node("head"), destination=Node("thorax")),
            Edge(source=Node("thorax"), destination=Node("abdomen")),
        ],
    )
    # Asserts that the skeleton has the matching Node and Edge objects.
>>>>>>> 1adc7affdfb67d755252e1d54de1ecab3ac4f472
    assert skeleton.nodes == [Node("head"), Node("thorax"), Node("abdomen")]
    assert skeleton.edges == [
        Edge(source=Node("head"), destination=Node("thorax")),
        Edge(source=Node("thorax"), destination=Node("abdomen")),
    ]
