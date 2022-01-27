from rrtpp.world_gen import get_rand_start_end, make_terrain
from rrtpp import surface, plots, rrt

from scipy.interpolate import RectBivariateSpline
from math import sqrt
import numba as nb
import numpy as np
import matplotlib.pyplot as plt


w, h = 120, 80
X, Y, Hterrain = make_terrain((w, h), (6, 4), cutoff=True, thresh=0.34, height=4)
# open space on the borders
Hterrain[:10, :] = 0
Hterrain[-10:, :] = 0
# get start goal
xstart, xgoal = np.array([int(5), int(h / 2)]), np.array([int(w - 5), int(h / 2)])

# scale aspect to terrain, but exaggerated
z_exaggerate = 1.0
terrain_aspect = [
    X.max() - X.min(),
    Y.max() - Y.min(),
    (Hterrain.max() - Hterrain.min()) * z_exaggerate,
]

Hsurf = surface.get_surface(
    X, Y, Hterrain, 2.5, 0.5, 1.0, np.tan(1.0), 0.04, verbose=True, solver="ECOS"
)


cost_hscale = 1e2


@nb.njit(fastmath=True)
def r2norm(x):
    return sqrt(x[0] * x[0] + x[1] * x[1])


@nb.njit(fastmath=True)
def costfn(vcosts, points, v, x) -> float:
    if vcosts[v] < 1e10:
        # get integer representation of points
        ax, ay = points[v]
        bx, by = x
        # get discrete points from surface
        line = surface.line_idx(ax, bx, ay, by)
        # get h-value of surface at each idx point on line
        # we need to unpack these if they are in compiled func
        linex, liney = line[:, 0], line[:, 1]

        # get cost of each line point
        cost = np.empty((linex.shape[0],), dtype=np.float64)
        for i in np.arange(linex.shape[0]):
            lx, ly = linex[i], liney[i]
            cost[i] = Hsurf[lx, ly]

        # average h-val in cells
        cost = np.sum(cost) / cost.shape[0]
        # integrate average cost over line distance
        # then multiply by scaling factor.
        cost *= r2norm(points[v] - x) * cost_hscale
        # finally, we add the r2 distace, so that in very
        # low-cost (areas where cost ~= 0) we still get min
        # distance rewirings.
        cost += r2norm(points[v] - x)
        # add r2 distance
        return vcosts[v] + cost
    else:
        return np.inf


rrts = rrt.RRTStar(Hterrain, n=4000, r_rewire=w, costfn=costfn)
T, vgoal = rrts.make(xstart, xgoal)

rrts_c = rrt.RRTStar(Hterrain, n=4000, r_rewire=w)
T_c, vgoal_c = rrts_c.make(xstart, xgoal)


fig2, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), tight_layout=True)
for ij in np.ndindex(ax.shape):
    plots.plot_start_goal(ax[ij], xstart, xgoal)
    ax[ij].imshow(Hterrain.T, cmap="binary", origin="lower")
    i, j = ij
    # first row plots
    if i == 0:
        G, vg = T_c, vgoal_c
        rrts_ = rrts
    # second row plots
    elif i == 1:
        G, vg = T, vgoal
        rrts_ = rrts_c
        if j == 1 or j == 2:
            ax[ij].contour(
                X,
                Y,
                Hsurf,
                levels=9,
                cmap="inferno",
                linewidths=0.5,
                antialiased=True,
            )
    if j == 1 or j == 2:
        path = rrts_.path_points(G, rrts_.route2gv(G, vg))
        plots.plot_path(ax[ij], path)
    if j == 2:
        plots.plot_rrt_lines(ax[ij], G, cmap="BuRd")

    title = (
        r"RRT* ($\mathbb{R}^2$ Distance Heuristic)"
        if i == 0
        else r"RRT* (Optimal Cost Heuristic)"
    )
    title += " "
    if j == 0:
        title += "Cspace Only"
    elif j == 1:
        title += "Path"
    elif j == 2:
        title += "Tree"
    ax[i, j].set_title(title)
    ax[i, j].get_xaxis().set_visible(False)
    ax[i, j].get_yaxis().set_visible(False)
fig2.savefig("rrt_surface.pdf", dpi=300)
plt.show()
