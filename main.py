import params
from params import ENCODERS
from defend import trainer
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default=ENCODERS[0], choices=ENCODERS)

    args = parser.parse_args()
    config = params.defend_config(args=args)
    print(f"\nconfig:\n{config}\n")
    trainer.run(config)

"""
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/luigi/Desktop/defend-pytorch'])
"""