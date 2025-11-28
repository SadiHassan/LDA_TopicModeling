from pyvis.network import Network

net = Network()
net.add_node(1, label="A")
net.add_node(2, label="B")
net.add_edge(1, 2)
net.show("test_graph.html")