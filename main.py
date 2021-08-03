import argparse
import json

from data_loader import get_dataloader
from solver import Solver


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', action='store', type=str, required=True)
    parser.add_argument('--num_epoch', action='store', type=int, default=3000)
    args = parser.parse_args()

    config = json.load(open(args.config_file))

    train_loader = get_dataloader(training=True, batch_size=16, num_workers=2)
    test_loader = get_dataloader(training=False, batch_size=16, num_workers=2)

    solver = Solver(train_loader, test_loader, config)
    solver.train(num_epoch=args.num_epoch)
