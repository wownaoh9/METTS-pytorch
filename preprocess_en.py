import argparse

import yaml

from preprocessor.preprocessor_esden_lj import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config", type=str,required=False, default="./config/en/preprocess.yaml",help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
