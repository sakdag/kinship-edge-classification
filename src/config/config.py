# Kinship graph file name
KINSHIP_GRAPH_FILE_PATH = '../../data/raw/kinship.data'

# File paths to store dataframe of features extracted from train and test graphs
TRAIN_KINSHIP_FEATURES_FILE_PATH = '../../data/processed/train_kinship_features.csv'
TEST_KINSHIP_FEATURES_FILE_PATH = '../../data/processed/test_kinship_features.csv'

# List of features both provided in problem description and added ones
ORIGINAL_FEATURES = ['label',
                     'n1_outlinks',
                     'n1_inlinks',
                     'n2_outlinks',
                     'n2_inlinks',
                     'common_neighbors',
                     'n1_bw_centrality',
                     'n2_bw_centrality',
                     'longest_path']

ADDED_FEATURES = ['adamic_adar_index',
                  'n1_katz_centrality',
                  'n2_katz_centrality']
