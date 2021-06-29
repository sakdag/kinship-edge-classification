import pandas as pd
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


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
