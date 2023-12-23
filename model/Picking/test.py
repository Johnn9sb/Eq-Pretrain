import argparse

parser = argparse.ArgumentParser(description='Some training hyperparameter')

parser.add_argument(
    '--batch_size', 
    default=32,
    help='Training batch size',
)

parser.add_argument(
    '--epochs',
    default='200',
    help='Training epochs',
)

args = parser.parse_args()

print('batch_size = ', args.batch_size)
print('epochs = ', args.epochs)

print('batch_size type = ', type(int(args.batch_size)))
print('epochs type = ', type(args.epochs))