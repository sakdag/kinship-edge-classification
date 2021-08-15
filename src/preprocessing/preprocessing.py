import argparse
import os
import random

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import src.config.config as conf


def generate_train_graph_and_test_edges(file_name: str):
    graph_file = open(file_name, 'r')
    lines = graph_file.readlines()
    graph_file.close()

    for line in lines:
        if line == '\n':
            lines.remove(line)

    # Generate set of all nodes before dividing the lines
    node_set = get_set_of_all_nodes(lines)

    # Shuffle the lines to split into test and train data
    random.shuffle(lines)

    split_point = int(len(lines) / 5) + 1

    test_lines = lines[:split_point]
    train_lines = lines[split_point:]

    # Generate graph from train set
    train_graph = generate_graph_from_lines(train_lines, node_set)

    # Get first node, second node and relationship label of each edge in the teest set
    test_edges = get_test_edges(test_lines)

    # nx.draw(kinship_graph, with_labels=True)
    # plt.show()

    return train_graph, test_edges


def get_set_of_all_nodes(lines: list):
    node_set = set()

    for line in lines:
        split_by_comma = line.split(',')
        first_node = split_by_comma[0][line.index('(') + 1:]
        second_node = split_by_comma[1][1:-2]

        if first_node not in node_set:
            node_set.add(first_node)
        if second_node not in node_set:
            node_set.add(second_node)

    return node_set


def generate_graph_from_lines(lines: list, node_set: set):
    kinship_graph = nx.DiGraph()

    for node in node_set:
        kinship_graph.add_node(node)

    for line in lines:
        relationship = line[:line.index('(')]
        split_by_comma = line.split(',')
        first_node = split_by_comma[0][line.index('(') + 1:]
        second_node = split_by_comma[1][1:-2]

        kinship_graph.add_edges_from([(first_node, second_node, {'label': relationship})])

    return kinship_graph


def get_test_edges(lines: list):
    test_edges = list()

    for line in lines:
        relationship = line[:line.index('(')]
        split_by_comma = line.split(',')
        first_node = split_by_comma[0][line.index('(') + 1:]
        second_node = split_by_comma[1][1:-2]

        test_edges.append((first_node, second_node, relationship))

    return test_edges


def extract_features_from_graph(kinship_graph: nx.DiGraph):
    # Get undirected version of the kinship_graph for later use
    # Note direction of edge is removed, edge in both directions are not needed
    undirected_kinship_graph = kinship_graph.to_undirected()

    # Betweenness centrality values of nodes in undirected_kinship_graph
    bw_centrality = nx.betweenness_centrality(undirected_kinship_graph, normalized=False)

    # Katz centrality values of nodes
    phi = max(nx.adjacency_spectrum(kinship_graph)).real  # largest eigenvalue of adj matrix
    katz_centrality = nx.katz_centrality(kinship_graph, 1 / phi - 0.01)

    # Dictionary to create dataframe from, keys are edge indices
    # For each key, value is a list representing features of that edge
    # Features are the following:
    # 0: label of the edge
    # Given features
    # 1: number of other outlinks of n1 (other than the given relationship)
    # 2: number of inlinks of n1
    # 3: number of outlinks of n2
    # 4: number of other inlinks of n2 (other than the given relationship)
    # 5: number of common neighbors (inlink or outlink) of n1 and n2
    # 6: betweenness centrality value of n1
    # 7: betweenness centrality value of n2
    # 8: length of the longest path between n1 and n2
    # Added features
    # 9: Adamic Adar index of the edge
    # 10: Katz centrality value of n1
    # 11: Katz centrality value of n2
    feature_dict = dict()

    for u, v, a in kinship_graph.edges(data=True):
        feature_dict[(u, v)] = generate_features_for_nodes(kinship_graph,
                                                           undirected_kinship_graph,
                                                           bw_centrality,
                                                           katz_centrality,
                                                           u,
                                                           v,
                                                           a['label'],
                                                           is_test_edges=False)

    feature_names = conf.ORIGINAL_FEATURES + conf.ADDED_FEATURES

    return pd.DataFrame.from_dict(feature_dict, orient='index', columns=feature_names)


def extract_features_for_test_edges(train_kinship_graph: nx.DiGraph, test_edges: list):
    # Get undirected version of the kinship_graph for later use
    # Note direction of edge is removed, edge in both directions are not needed
    undirected_kinship_graph = train_kinship_graph.to_undirected()

    # Betweenness centrality values of nodes in undirected_kinship_graph
    bw_centrality = nx.betweenness_centrality(undirected_kinship_graph, normalized=False)

    # Katz centrality values of nodes
    phi = max(nx.adjacency_spectrum(train_kinship_graph)).real  # largest eigenvalue of adj matrix
    katz_centrality = nx.katz_centrality(train_kinship_graph, 1 / phi - 0.01)

    # Dictionary to create dataframe from, keys are edge indices
    # For each key, value is a list representing features of that edge
    # Features are the following:
    # 0: label of the edge
    # Given features
    # 1: number of other outlinks of n1 (other than the given relationship)
    # 2: number of inlinks of n1
    # 3: number of outlinks of n2
    # 4: number of other inlinks of n2 (other than the given relationship)
    # 5: number of common neighbors (inlink or outlink) of n1 and n2
    # 6: betweenness centrality value of n1
    # 7: betweenness centrality value of n2
    # 8: length of the longest path between n1 and n2
    # Added features
    # 9: Adamic Adar index of the edge
    # 10: Katz centrality value of n1
    # 11: Katz centrality value of n2
    feature_dict = dict()

    # current_tuple structure (u, v, label)
    for current_tuple in test_edges:
        feature_dict[(current_tuple[0], current_tuple[1])] = generate_features_for_nodes(train_kinship_graph,
                                                                                         undirected_kinship_graph,
                                                                                         bw_centrality,
                                                                                         katz_centrality,
                                                                                         current_tuple[0],
                                                                                         current_tuple[1],
                                                                                         current_tuple[2],
                                                                                         is_test_edges=True)

    feature_names = conf.ORIGINAL_FEATURES + conf.ADDED_FEATURES

    return pd.DataFrame.from_dict(feature_dict, orient='index', columns=feature_names)


# is_test_edges boolean is used to control if provided nodes have label in training dataset, if so number of other
# outlinks of n1 and number of other inlinks of n2 should be subtracted 1 to not count edge itself.
def generate_features_for_nodes(kinship_graph, undirected_kinship_graph, bw_centrality, katz_centrality,
                                u, v, relationship, is_test_edges: bool):
    feature_list = list()

    # Store label of the edge
    feature_list.append(relationship)

    # ------------------- Original Features -------------------
    # Store number of other outlinks of n1 (other than the given relationship)
    n1_out_degree = kinship_graph.out_degree(u)
    if not is_test_edges:
        n1_out_degree -= 1
    feature_list.append(n1_out_degree)

    # Store number of inlinks of n1
    feature_list.append(kinship_graph.in_degree(u))

    # Store number of outlinks of n2
    feature_list.append(kinship_graph.out_degree(v))

    # Store number of other inlinks of n2 (other than the given relationship)
    n2_in_degree = kinship_graph.in_degree(v)
    if not is_test_edges:
        n2_in_degree -= 1
    feature_list.append(n2_in_degree)

    # Store number of common neighbors (inlink or outlink) of n1 and n2
    neighbors_of_n1 = undirected_kinship_graph.neighbors(u)
    neighbors_of_n2 = undirected_kinship_graph.neighbors(v)

    feature_list.append(len(list(set(neighbors_of_n1).intersection(neighbors_of_n2))))

    # Store betweenness centrality value of n1
    feature_list.append(bw_centrality[u])

    # Store betweenness centrality value of n2
    feature_list.append(bw_centrality[v])

    # Store length of the longest path between n1 and n2
    feature_list.append(len(max(nx.all_simple_paths(kinship_graph, u, v), key=lambda x: len(x))))

    # ------------------- Added Features -------------------
    # Store Adamic Adar index of the edge
    feature_list.append(nx.adamic_adar_index(undirected_kinship_graph, [(u, v)]).__next__()[2])

    # Store Katz centrality value of n1
    feature_list.append(katz_centrality[u])

    # Store Katz centrality value of n2
    feature_list.append(katz_centrality[v])

    return feature_list


def main():
    dirname = os.path.dirname(__file__)
    kinship_data_file_name = os.path.join(dirname, conf.KINSHIP_GRAPH_FILE_PATH)

    train_kinship_features_file_name = os.path.join(dirname, conf.TRAIN_KINSHIP_FEATURES_FILE_PATH)
    test_kinship_features_file_name = os.path.join(dirname, conf.TEST_KINSHIP_FEATURES_FILE_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',
                        default=kinship_data_file_name,
                        help='absolute path of the dataset you want to use, default: '
                             '{path to project}/data/raw/kinship.data')
    parser.add_argument('--train_dataset_output_path',
                        default=train_kinship_features_file_name,
                        help='absolute path of the train kinship features dataset you want to use, default: '
                             '{path to project}/data/processed/train_kinship_features.csv')
    parser.add_argument('--test_dataset_output_path',
                        default=test_kinship_features_file_name,
                        help='absolute path of the test kinship features dataset you want to use, default: '
                             '{path to project}/data/processed/test_kinship_features.csv')
    parser_args = parser.parse_args()

    # Generate graph from train edges (80%) and get test edges as
    # tuple of (first node, second node, relationship) (20%)
    kinship_graph_train, test_edges = generate_train_graph_and_test_edges(kinship_data_file_name)

    df_train = extract_features_from_graph(kinship_graph_train)
    df_test = extract_features_for_test_edges(kinship_graph_train, test_edges)

    df_train.to_csv(parser_args.train_dataset_output_path, index=False)
    df_test.to_csv(parser_args.test_dataset_output_path, index=False)


if __name__ == '__main__':
    main()
