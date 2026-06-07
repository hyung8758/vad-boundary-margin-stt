import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.common import get_dataset_config, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Read pipeline values from conf/base.yaml.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "base.yaml"))
    parser.add_argument("--dataset", default=None, help="Dataset name under datasets.")
    parser.add_argument("--field", required=True, choices=["splits", "asr_tag", "vad_policy_name"])
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.field == "splits":
        _, dataset = get_dataset_config(config, args.dataset)
        for split in dataset["splits"]:
            print(split)
        return

    if args.field == "asr_tag":
        print(config["asr"]["model_name"].replace("-", "_"))
        return

    if args.field == "vad_policy_name":
        print(config["vad_policy"]["policy_name"])
        return


if __name__ == "__main__":
    main()
