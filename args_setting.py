import argparse

parser = argparse.ArgumentParser(description='args_setting')
# Train args
parser.add_argument('--DEVICE', type=str, default='cuda:0')
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=256)

parser.add_argument('--task', type=str, default='SPECT-MRI')  # CT-MRI, PET-MRI, SPECT-MRI
parser.add_argument('--model', type=str, default='W-HSSF')  # CT-MRI, PET-MRI, SPECT-MRI

args = parser.parse_args()
