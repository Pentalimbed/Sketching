import os
import sys
import subprocess

interpreter = sys.executable
for dir, command in [
    ("outputs/def", "data_record.py -o outputs/def --colour"),
    ("outputs/t05", "data_record.py -o outputs/t05 --colour -t 0.5"),
    ("outputs/t02", "data_record.py -o outputs/t02 --colour -t 0.2"),
    ("outputs/t01", "data_record.py -o outputs/t01 --colour -t 0.1"),
    ("outputs/t005", "data_record.py -o outputs/t005 --colour -t 0.05"),
    ("outputs/t001", "data_record.py -o outputs/t001 --colour -t 0.01"),
    ("outputs/l2", "data_record.py -o outputs/l2 --colour -l 2"),
    ("outputs/l4", "data_record.py -o outputs/l4 --colour -l 4"),
    ("outputs/l05", "data_record.py -o outputs/l05 --colour -l 0.5"),
    ("outputs/lp", "data_record.py -o outputs/lp --colour --perceptual"),
]:
    if not os.path.exists(dir):
        os.makedirs(dir)
        subprocess.check_call([interpreter, *command.split(' ')], shell=True, stdout=sys.stdout,
                              stderr=subprocess.STDOUT)
