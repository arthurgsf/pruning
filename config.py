import os
import argparse

parser = argparse.ArgumentParser()

# dataset related
parser.add_argument('--train_dir', type=str,
                    default=f"{os.path.expanduser('~')}/Datasets/SEGTHOR_EXTRACTED/train/",
                    help='The directory containing the train image data.')
parser.add_argument('--test_dir', type=str,
                    default=f"{os.path.expanduser('~')}/Datasets/SEGTHOR_EXTRACTED/val/",
                    help='The directory containing the test image data.')
parser.add_argument('--records_dir', type=str,
                    default="./records",
                    help='The directory to store model records (weights, graphics, etc.)')

# input related
parser.add_argument('--n_channels', type=int, default=1, 
                    help='number of channels in the image.')
parser.add_argument('--image_shape', type=tuple, default=(256, 256),
                    help='The original image shape.')
parser.add_argument('--input_shape', type=tuple, default=(256, 256),
                    help='The shape of model input.')

# training related
parser.add_argument('--n_classes', type=int, default=1,
                    help='Number of classes.')
parser.add_argument('--batch_size', type=int, default=20,
                    help='Number of images per train batch.')
parser.add_argument('--epochs', type=int, default=100,
                    help='The model training epochs.')

# optimization related
parser.add_argument('--opt_epochs', type=int, default=500,
                    help='The optimization epochs.')

args = parser.parse_args()