import os
import sys
import subprocess

interpreter = sys.executable
for dir, command in [
    # ("outputs/def", ""),
    # ("outputs/t05", "-t 0.5"),
    # ("outputs/t1", "-t 1"),
    # ("outputs/t01", "-t 0.1"),
    # ("outputs/t005", "-t 0.05"),
    # ("outputs/t001", "-t 0.01"),
    # ("outputs/l2", "-l 2"),
    # ("outputs/l4", "-l 4"),
    # ("outputs/l05", "-l 0.5"),
    # ("outputs/st", "--straight_through"),
    # ("outputs/st07", "--straight_through"),
    # ("outputs/init1", "--init_length 1.0"),
    # ("outputs/init05", "--init_length 0.5"),
    # ("outputs/init01", "--init_length 0.1"),
    ("outputs/iw02", "--init_width 0.1 -t 1.0"),
]:
    if not os.path.exists(dir):
        os.makedirs(dir)
    true_command = f"data_record.py --colour -o {dir} {command}"
    subprocess.check_call([interpreter, *true_command.split(' ')], shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
