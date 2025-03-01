import argparse
from lift.gated_memory.utils import preprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input', required=True)
    parser.add_argument('-O', '--output', required=True)
    args = parser.parse_args()
    preprocess(args.input, args.output)
