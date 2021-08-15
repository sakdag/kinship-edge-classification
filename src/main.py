import src.preprocessing.preprocessing as prep
import src.classification.classification as classification

if __name__ == '__main__':

    # Apply preprocessing steps and generate features then save
    prep.main()

    # Run classification algorithms to predict kinship relations
    classification.main()
