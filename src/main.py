import os
import src.config.config as conf
import src.preprocessing.preprocessing as prep

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    kinship_data_file_name = os.path.join(dirname, conf.KINSHIP_GRAPH_FILE_PATH)

    kinship_graph = prep.generate_graph_from_file(kinship_data_file_name)
