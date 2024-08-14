import os
import sys
import subprocess

interpreter = sys.executable
for dir, command in [
    ("outputs/def", "data_record.py -o outputs/def --colour"),
    ("outputs/def", "data_record.py -o outputs/def --colour -t 0.5"),
    ("outputs/def", "data_record.py -o outputs/def --colour -t 0.2"),
    ("outputs/def", "data_record.py -o outputs/def --colour -t 0.1"),
    ("outputs/def", "data_record.py -o outputs/def --colour -t 0.05"),
    ("outputs/def", "data_record.py -o outputs/def --colour -t 0.01")
]:
    if not os.path.exists(dir):
        os.makedirs(dir)
        subprocess.check_call([interpreter, *command.split(' ')], shell=True, stdout=sys.stdout,
                              stderr=subprocess.STDOUT)
