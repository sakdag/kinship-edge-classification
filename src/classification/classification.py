import argparse
import os

import pandas as pd

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import src.config.config as conf


def run_classifiers(train_df: pd.DataFrame, test_df: pd.DataFrame, classifiers: dict, label_column_name: str):
    for classifier_name in classifiers.keys():
        print('Running ', classifier_name, ' classifier:')

        classifier = classifiers[classifier_name]

        y_train = train_df[label_column_name]
        x_train = train_df.drop(columns=[label_column_name])

        classifier.fit(x_train, y_train)

        y_test_actual = test_df[label_column_name]
        x_test = test_df.drop(columns=[label_column_name])

        y_test_predicted = classifier.predict(x_test)

        print('Classification results:')
        print(classification_report(y_test_actual, y_test_predicted))


def main():
    dirname = os.path.dirname(__file__)

    train_kinship_features_file_name = os.path.join(dirname, conf.TRAIN_KINSHIP_FEATURES_FILE_PATH)
    test_kinship_features_file_name = os.path.join(dirname, conf.TEST_KINSHIP_FEATURES_FILE_PATH)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path',
                        default=train_kinship_features_file_name,
                        help='absolute path of the train kinship features dataset you want to use, default: '
                             '{path to project}/data/processed/train_kinship_features.csv')
    parser.add_argument('--test_dataset_path',
                        default=test_kinship_features_file_name,
                        help='absolute path of the test kinship features dataset you want to use, default: '
                             '{path to project}/data/processed/test_kinship_features.csv')
    parser_args = parser.parse_args()

    train_df_v2 = pd.read_csv(parser_args.train_dataset_path)
    test_df_v2 = pd.read_csv(parser_args.test_dataset_path)

    train_df_v1 = train_df_v2.copy().drop(columns=conf.ADDED_FEATURES)
    test_df_v1 = test_df_v2.copy().drop(columns=conf.ADDED_FEATURES)

    classifiers_to_use = dict()
    classifiers_to_use['K Nearest Neighbors'] = KNeighborsClassifier(n_neighbors=4)
    classifiers_to_use['Naive Bayes'] = GaussianNB()
    classifiers_to_use['Random Forest'] = RandomForestClassifier(max_depth=2, random_state=0)
    classifiers_to_use['Decision Tree'] = DecisionTreeClassifier(random_state=0)

    # Run classifiers on first version of the dataset
    print('Running classifiers on first version of the dataset:')
    run_classifiers(train_df_v1, test_df_v1, classifiers_to_use, label_column_name='label')

    # Run classifiers on second version of the dataset with added features
    print('Running classifiers on second version of the dataset:')
    run_classifiers(train_df_v2, test_df_v2, classifiers_to_use, label_column_name='label')


if __name__ == '__main__':
    main()
