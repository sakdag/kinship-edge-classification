# Kinship Edge Classification
This project aims to classify family relationships using features extracted from family tree. Details of the project
can be seen in Details part below.

## Usage

If you are going to use default settings, you can run the main.py without any parameters which expects dataset to be
data/raw/kinship.data, then applies both preprocessing, feature extraction and classification steps. Also, you can run
preprocessing and classification parts in order as shown below, while customizing the parameters. For more information, 
use -h option in command line.

```
python3 src/preprocessing/preprocessing.py
python3 src/classification/classification.py
```

## Dataset

Dataset used in this project (data/raw/kinship.data) consists of 104 instances of kinship information among 24
family members. Dataset contain 12 types of relations (edge labels) between pairs of family members.

## Environment

Pyhton 3.9 and Conda environment with dependencies as given in requirements.txt is used.

## Details

1. Family relationships in kinship.data is read and corresponding directed family graph is generated. Note than instead
of k-fold cross validation, traditional train test split is used in this project.
2. Following features are extracted from the graph, for every edge between n1 and n2:
   1. number of other outlinks of n1 (other than the given relationship)
   2. number of inlinks of n1
   3. number of outlinks of n2
   4. number of other inlinks of n2 (other than the given relationship)
   5. number of common neighbors (inlink or outlink) of n1 and n2
   6. betweenness centrality value of n1
   7. betweenness centrality value of n2
   8. length of the longest path between n1 and n2
   9. Adamic Adar index of the edge
   10. Katz-centrality value of n1
   11. Katz-centrality value of n2
3. Using the generated features above, family relationsips in test dataset is tried to be classified using following 4
classifiers:
   1. K-Nearest Neighbors Classifier with number of neighbors: 4
   2. Gaussian Naive Bayes Classifier
   3. Random Forest Classifier with max_depth: 2

## License
[MIT](https://choosealicense.com/licenses/mit/)
