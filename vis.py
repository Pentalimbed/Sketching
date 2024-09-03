import copy
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import ImageGrid
import torch

plt.style.use("ggplot")
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def convert_name(name, mode=0):
    if mode == 0:
        if name == "def":
            name = "t02"
        name = name[1:]
    elif mode == 1:
        if name == "def":
            name = "init02"
        name = name[4:]
    elif mode == 2:
        if name == "iw01":
            return "$w^0_{\\max} = 0.1$"
        name = name[1:]

    name = name[:1] + "." + name[1:]
    if name.endswith("."):
        name = name[:-1]

    if mode == 0:
        name = f"$w_{{\\max}} = {name}$"
    elif mode == 1:
        name = f"$l^0_{{\\max}} = {name}$"
    elif mode == 2:
        name = f"$w_{{\\max}} = {name}$"

    return name


dirs = [
    ["t1", "t05", "def", "t01", "t005", "t001"],
    ["init1", "init05", "def", "init01"],
    ["t1", "t01", "iw01"]
]


def draw_curves(mode=0):
    out_names = ["wmax", "lmax", "w0max"]

    curves = dict()
    for path in dirs[mode]:
        name = convert_name(path, mode)

        curve_path = os.path.join("outputs", path, "err_curve.pt")
        err_curve = torch.load(curve_path)['error_curve']
        curves[name] = err_curve[0].numpy()

    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 2, figure=fig)

    ax = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    for name, curve in iter(sorted(curves.items())):
        ax.plot(curve[:, 0], label=name)
        ax.set(yscale='log')
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.legend()

        ax1.plot(curve[:100, 0])
        ax1.set(yscale='log', xlabel="first 100 strokes")

        ax2.plot(np.arange(900, 1000), curve[-100:, 0])
        ax2.set(yscale='log', xlabel="last 100 strokes")
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()

    fig.supxlabel("number of strokes")
    fig.supylabel("$L^2$ loss")
    fig.tight_layout()

    fig.savefig(f'viz/{out_names[mode]}.eps', format='eps')
    plt.show()


def imgarray(mode=0):
    steps = ["100", "200", "400", "800", "999"]
    out_names = ["wmax-sample",
                 "lmax-sample",
                 "w0max-sample"]

    fig = plt.figure(figsize=(9.5, len(dirs[mode]) * 2 + 0.1))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(len(dirs[mode]), len(steps)),  # creates 2x2 grid of Axes
                     axes_pad=0,  # pad between Axes in inch.
                     share_all=True
                     )
    for cax in grid.cbar_axes:
        cax.axis[cax.orientation].set_label('Foo')

    i = 0

    for dir in dirs[mode]:
        name = convert_name(dir, mode)

        grid[i].set(ylabel=name)
        for step in ["100", "200", "400", "800", "999"]:
            img_path = os.path.join("outputs", dir, f"{step}-move99.png")
            im = plt.imread(img_path)

            ax = grid[i]
            ax.imshow(im)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])

            ax.set(xlabel=step if step != "999" else "1000")

            i += 1

    fig.tight_layout()
    fig.supxlabel("number of strokes")
    fig.savefig(f'viz/{out_names[mode]}.eps')
    plt.show()


def print_final():
    err_curve = torch.load("outputs/final/err_curve.pt")['error_curve'][0]
    for u in [199,499,999]:
        print(f"{u}: {err_curve[u]}")


# draw_curves(0)
# draw_curves(1)
# draw_curves(2)
#
# imgarray(0)
# imgarray(1)
# imgarray(2)

print_final()
