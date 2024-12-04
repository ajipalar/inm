import graph_tool.all as gt
import pickle as pkl
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import click

logger = logging.getLogger(__name__)

def model_output_graph_from_average_edge_list(
        modeling_output_dirpath: Path,
        fname="average_predicted_edge_scores.tsv",
        threshold = 0.5):

    df = pd.read_csv(modeling_output_dirpath / fname, sep = "\t") 
    sel = df['w'] >= threshold 
    df = df[sel]
    edge_list = [(r['a_gene'], r['b_gene']) for i, r in df.iterrows()]
    g = gt.Graph(edge_list, directed = False, hashed = True)
    return g 

def get_graph(modeling_output_dirpath, fname="average_predicted_edge_scores.tsv"):
    df = pd.read_csv(modeling_output_dirpath / fname, sep = "\t") 
    edge_list = [(r['a_gene'], r['b_gene'], r['w']) for i, r in df.iterrows()]
    g = gt.Graph(edge_list, directed = False, hashed = True, eprops=[('weight', 'double')])
    return g

def radial_layout(r, n):
    theta_frac = 2 * np.pi / n
    array = np.zeros((n, 2))
    theta = 0
    for i in range(n):
        array[i, :] = [r * np.cos(theta), r * np.sin(theta)]
        theta += theta_frac
    return array

def add_edge_labels_and_property_map_to_graph(g):
    n_vertices = g.num_vertices()

def radial_vp(g, r):
    n = g.num_vertices()
    xy = radial_layout(r, n)
    pos = g.new_vp("vector<double>")
    for vidx in range(g.num_vertices()):
        pos[vidx] = xy[vidx]
    return pos

def get_neighbor_graphview(g, vidx):
    neighbors = g.get_out_neighrbors(vidx)
    return gt.GraphView(g, vfilt = lambda v: v in neighbors)  


def get_local_network(g, r, name):
    mask = g.new_vp("bool")
    for vidx in g.vertices():
        if g.vp['ids'][vidx] == name:
            val = True 
        else:
            val = False 
        mask[vidx] = val

def calc_theta(x, y, r, threshold=0.5):
    """
     Numerically stable
    """
    if abs(x) < threshold:
        assert abs(y) >= threshold, (x, y, r) 
        theta = np.arcsin(y / r)
    else:
        theta = np.arccos(x / r)
    return theta

def get_radial_offset(g, pos):
    """
    Add a radial offset to place text in the hierarchy
    g : graph
    """
    # 1. Get the positions of every node
    n = g.num_vertices()
    radial_offset = g.new_vp("float")
    compute_r = True
    for vidx in g.vertices(): 
        xy = pos[vidx]
        x, y = xy
        if compute_r:
            r = np.sqrt(np.sum(np.square(xy))) 
            compute_r = False
        theta = calc_theta(x, y, r) 
        if theta < 0:
            theta = 2 *np.pi - theta
        radial_offset[vidx] = theta 
    return radial_offset

def get_neighborhood_view(g, idx):
    neighbors = g.get_all_neighbors(idx)
    arr = np.array(neighbors)
    neighborhood = np.concatenate([np.array(idx), arr])
    gn = gt.GraphView(g, vfilt = neighborhood)
    return gn

def get_text_offset(g, pos):
    text_offset = g.new_vp()
    for vidx in g.vertices():
        xy = pos[vidx]

def get_thresholded_edge_view(g, threshold):
    return gt.GraphView(g, efilt = lambda e: g.edge_properties['weight'][e] >= threshold)

def plot_nested_edgeview(g, modeling_output_dirpath, view_threshold = 0.99, r=1.):
    ug = get_thresholded_edge_view(g, view_threshold) 
    state = gt.minimize_nested_blockmodel_dl(ug)
    pos = radial_vp(ug, r)
    hpos, ht, htpos = gt.draw_hierarchy(state, 
                      output=str(modeling_output_dirpath / "nested.pdf")) 
    radial_offset = get_radial_offset(ug, hpos)
    hpos, ht, htpos = gt.draw_hierarchy(state, 
                      pos = hpos,
                      vertex_text = g.vp['ids'],
                      vertex_text_position = radial_offset,
#                      vertex_text_rotation = radial_offset,
                      nodes_first = True,
                      output=str(modeling_output_dirpath / "nested_with_text.pdf")) 
    return dict(ug = ug,
                state = state,
                pos = pos,
                hpos = hpos,
                ht = ht,
                htpos = htpos)


def add_radial_offest(xy, rel_r=0.1):
    """
    add a radial offset to 
    """
    r = np.sqrt(np.sum(np.square(xy)))
    x, y = xy
    theta = np.arccos(x / r)
    r2 = r + r * rel_r
    x2 = r2 * np.cos(theta)
    y2 = r2 * np.sin(theta)
    return np.array([x2, y2])


def plot_graph(g, modeling_output_dirpath, fname="outgraph.pdf", layout="sfdp", C=0.2):
    layout_dict = {"sfdp" : gt.sfdp_layout, "arg_layout": gt.arf_layout} 
    layout_func = layout_dict[layout]
    pos = layout_func(g)
    gt.graph_draw(g,  pos = pos, vertex_text = g.vertex_properties['ids'], output = str(modeling_output_dirpath / fname)) 

def nested_plot_from_output_dir(o, view_threshold, edge_threshold):
    if isinstance(o, str):
        o = Path(o)
    elif isinstance(o, Path):
        ...
    else:
        raise TypeError("Expected string or Path")

    g = model_output_graph_from_average_edge_list(o, threshold = edge_threshold)
    plot_nested_edgeview(g = g, modeling_output_dirpath = o, view_threshold = view_threshold, r=1)


_2024_04_01_dummy = Path("../results/2024_04_01_dummy/")

@click.command()
@click.option("--o")
@click.option("--view-threshold", default=0.99)
@click.option("--edge-threshold", default=0.5)
def main(
    o,
    view_threshold,
    edge_threshold):

    _main(
        o = o,
        view_threshold = view_threshold,
        edge_threshold=edge_threshold)


def _main(o, view_threshold, edge_threshold):

    assert isinstance(o, str)
    view_threshold = float(view_threshold)
    edge_threshold = float(edge_threshold)
    o = Path(o)

    if not o.is_dir():
        raise ValueError(f"input for o is not a directory: {str(o)}")

    assert view_threshold >= 0
    assert view_threshold <= 1
    assert edge_threshold >= 0
    assert edge_threshold <= 1
    assert edge_threshold < view_threshold 

    nested_plot_from_output_dir(
            o = o, 
            view_threshold = view_threshold,
            edge_threshold = edge_threshold)

if __name__ == "__main__":
    main()
