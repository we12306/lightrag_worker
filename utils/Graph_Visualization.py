import networkx as nx
from pyvis.network import Network

def visualize_graph(input_file, output_file):
    # Load the GraphML file
    G = nx.read_graphml(input_file)

    # Create a Pyvis network
    net = Network(notebook=True)

    # Convert NetworkX graph to Pyvis network
    net.from_nx(G)

    # Save and display the network
    net.show(output_file)

if __name__ == '__main__':
    input_file = '/home/xiongdb/test/lightrag/lightrag_store/graph_chunk_entity_relation.graphml'
    output_file = 'knowledge_graph.html'
    visualize_graph(input_file, output_file)