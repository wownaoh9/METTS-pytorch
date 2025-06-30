import argparse
import yaml
from preprocessor import DOE, biaobei

def main(config):
    if "DOE" in config["dataset"]:
        DOE.prepare_align(config)
    if "biaobei" in config["dataset"]:
        biaobei.prepare_align(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config", type=str,required=False, default="./config/biaobei/preprocess.yaml",help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)