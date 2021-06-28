import networkx as nx
import matplotlib.pyplot as plt


def generate_graph_from_file(file_name: str):
    graph_file = open(file_name, 'r')
    lines = graph_file.readlines()

    kinship_graph = nx.DiGraph()
    already_added_nodes = set()

    for line in lines:
        if line == '\n':
            continue
        relationship = line[:line.index('(')]
        split_by_comma = line.split(',')
        first_node = split_by_comma[0][line.index('(') + 1:]
        second_node = split_by_comma[1][:-2]

        if first_node not in already_added_nodes:
            kinship_graph.add_node(first_node)
        if second_node not in already_added_nodes:
            kinship_graph.add_node(second_node)
        kinship_graph.add_edges_from([(first_node, second_node, {'label': relationship})])

    # nx.draw(kinship_graph, with_labels=True)
    # plt.show()

    return kinship_graph
