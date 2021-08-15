import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import src.config.config as conf
import src.preprocessing.preprocessing as prep
import src.classification.classification as classification

if __name__ == '__main__':
    mode = str(sys.argv[1])

    dirname = os.path.dirname(__file__)
    kinship_data_file_name = os.path.join(dirname, conf.KINSHIP_GRAPH_FILE_PATH)

    train_kinship_features_file_name = os.path.join(dirname, conf.TRAIN_KINSHIP_FEATURES_FILE_PATH)
    test_kinship_features_file_name = os.path.join(dirname, conf.TEST_KINSHIP_FEATURES_FILE_PATH)

    if mode == 'generate_datasets':
        # Generate graph from train edges (80%) and get test edges as
        # tuple of (first node, second node, relationship) (20%)
        kinship_graph_train, test_edges = prep.generate_train_graph_and_test_edges(kinship_data_file_name)

        df_train = prep.extract_features_from_graph(kinship_graph_train)
        df_test = prep.extract_features_for_test_edges(kinship_graph_train, test_edges)

        df_train.to_csv(train_kinship_features_file_name, index=False)
        df_test.to_csv(test_kinship_features_file_name, index=False)

    elif mode == 'classify':
        train_df_v2 = pd.read_csv(train_kinship_features_file_name)
        test_df_v2 = pd.read_csv(test_kinship_features_file_name)

        train_df_v1 = train_df_v2.copy().drop(columns=conf.ADDED_FEATURES)
        test_df_v1 = test_df_v2.copy().drop(columns=conf.ADDED_FEATURES)

        classifiers_to_use = dict()
        classifiers_to_use['K Nearest Neighbors'] = KNeighborsClassifier(n_neighbors=4)
        classifiers_to_use['Naive Bayes'] = GaussianNB()
        classifiers_to_use['Random Forest'] = RandomForestClassifier(max_depth=2, random_state=0)
        classifiers_to_use['Decision Tree'] = DecisionTreeClassifier(random_state=0)

        # Run classifiers on first version of the dataset
        print('Running classifiers on first version of the dataset:')
        classification.run_classifiers(train_df_v1, test_df_v1, classifiers_to_use, label_column_name='label')

        # Run classifiers on second version of the dataset with added features
        print('Running classifiers on second version of the dataset:')
        classification.run_classifiers(train_df_v2, test_df_v2, classifiers_to_use, label_column_name='label')
