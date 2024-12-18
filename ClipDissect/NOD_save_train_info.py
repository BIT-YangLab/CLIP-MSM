import torch
import argparse
import os

root_path = os.path.abspath("../")

parser = argparse.ArgumentParser(description='save_train_info')
parser.add_argument("--md", type=str, choices=[str(i) for i in range(1)], default="0")
parser.add_argument("--cs", type=str, choices=[str(i) for i in range(1, 10)], default='1')
parser.add_argument("--cr", type=str, choices=[str(i) for i in range(5)], default='0')

if __name__ == "__main__":
    args = parser.parse_args()
    current_subject = args.cs
    current_roi = args.cr
    current_model = args.md
    torch.save((current_model, current_subject, current_roi), root_path + "/result/NOD_train_info.pt")
