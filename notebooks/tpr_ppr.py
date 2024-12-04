"""
Given two UndirectedEdgeLists
- pred and ref
- predictions are real values
- reference may be real valued edges or simply binary (presence / absence)

Results are calculated based on the intersection of pred and ref
Calculate 

Initializes the reference based on the predictions.
For all edges in the prediction, the reference is updated such that
a lack of an edge is set to 0
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from undirected_edge_list import UndirectedEdgeList
from typing import NamedTuple, Any
import sklearn
import sklearn.metrics
Array = Any

_tpr_ppr_style = ""

class PprTprCalculator:
    def __init__(self, pred, ref):
        _calculator_init(self, pred, ref)
    def crunch(self, thresholds=None):
        return _calculator_crunch(self, thresholds = thresholds)
    def __repr__(self):
        return _calculator_repr(self)


class PprTprResults(NamedTuple):
    tpr_points : Array
    ppr_points : Array
    shuff_ppr_points : Array
    shuff_tpr_points : Array
    n_total_positives : int
    n_predicted_positives : int
    n_thresholds : Array
    auc : float
    shuff_auc : float
    delta_auc : float

class PprTprScatterStyle(NamedTuple):
    true_color : str
    shuff_color : str

class PprTprPlotter:
    """
    Takes one or more PprTprResults objects and makes a plot with the appropriate curves
    """
    def __init__(self):
        _plotter_init(self)
    def plot(self, save_path, title, results: PprTprResults):
        _plotter_plot(self, save_path, title, results) 

def add_curve_to_roc_plot(ax, results):
    ax.plot(results.ppr_points, results.tpr_points, label=f"predicted AUC {round(results.auc, 3)}")

def add_shuffled_curve_to_roc_plot(ax, results):
    ax.plot(results.shuff_ppr_points, results.shuff_tpr_points, label=f"shuffled AUC {round(results.shuff_auc, 3)}")

def set_roc_limits(ax):
    """Set the limits of the plot to be between 0 and 1.05
    Default is 1.05 to allow for a little bit of space around the plot"""
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    return ax

def set_roc_labels_with_N(ax, results):
    ax.set_xlabel(f"PPR N={results.n_predicted_positives}")
    ax.set_ylabel(f"TPR N={results.n_total_positives}")

def dpi_save(save_path, fig, dpi: int):
    fig.savefig(save_path + f"_{dpi}.png", dpi=dpi)


def _plotter_plot(o, save_path, title, results):
    fig, ax = plt.subplots(1, 1)
    ax = set_roc_limits(ax)
    add_curve_to_roc_plot(ax, results)
    add_shuffled_curve_to_roc_plot(ax, results)
    ax.set_title(title)
    ax.legend()
    dpi_save(save_path, fig, 300)
    dpi_save(save_path, fig, 1200)
    plt.close(fig=fig)

def _plotter_init(x):
    ...


def _calculator_init(self, pred, ref, rebuild_edge_dicts = True):
    assert isinstance(pred, UndirectedEdgeList)
    assert isinstance(ref, UndirectedEdgeList)
    assert len(pred.node_intersection(ref)) > 0, "No intersecting nodes"
    # Select the set of shared edges
    shared_edges = pred.edge_identity_intersection(ref, rebuild_dicts = rebuild_edge_dicts) 
    pred_edges = {}
    ref_edges = {}
    node_dict = {}
    for edge in shared_edges:
        pred_edges[edge] = pred._edge_dict[edge]
        ref_edges[edge] = ref._edge_dict[edge]
        a, b = edge
        assert a != b
        node_dict[a] = 0
        node_dict[b] = 0
    not_shared_edges = set(pred._edge_dict.keys()) - shared_edges
    for edge in not_shared_edges:
        pred_edges[edge] = pred._edge_dict[edge]
        ref_edges[edge] = 0
        a, b = edge
        assert a != b
        node_dict[a] = 0
        node_dict[b] = 0
    self.pred_edge_dict = pred_edges.copy()
    self.ref_edge_dict = ref_edges.copy()
    self.nodes = set(node_dict.keys())
    self.n_nodes = len(node_dict.keys())
    self.n_edges = len(pred_edges.keys())

def _calculator_repr(x):
    n_shared = x.n_edges
    n_nodes = x.n_nodes
    return f"""N shared edges {n_shared}
N nodes {n_nodes}
"""

def _calculator_crunch(x, thresholds):
    if thresholds is None:
        thresholds = np.linspace(0, 1, 1000)
    # Generate the edge lists using numpy
    pred = np.zeros(x.n_edges)
    ref = np.zeros(x.n_edges)
    # populate the arrays
    for i, (edge, ref_value) in enumerate(x.ref_edge_dict.items()):
        pred_value = x.pred_edge_dict[edge]
        ref[i] = ref_value
        pred[i] = pred_value
    shuff_pred = pred.copy()
    np.random.shuffle(shuff_pred)  
    ref = ref.astype(bool)
    n_edges = len(pred)
    n_total_positives = np.sum(ref)
    ppr_points = []
    tpr_points = []
    shuff_ppr_points = []
    shuff_tpr_points = []
    # Caculate TPR and PPR over a range of thhresholds
    assert np.alltrue(~np.isnan(pred))
    assert np.alltrue(~np.isnan(ref))
    assert np.alltrue(~np.isnan(shuff_pred))
    for threshold in thresholds:
        #print(pred, threshold)
        positive_predictions = pred >= threshold
        shuff_positive_predictions = shuff_pred >= threshold
        ppr = np.sum(positive_predictions) / n_edges
        shuff_ppr = np.sum(shuff_positive_predictions) / n_edges
        true_positives = ref & positive_predictions 
        shuff_true_positives = ref & shuff_positive_predictions
        tpr = np.sum(true_positives) / n_total_positives
        shuff_tpr = np.sum(shuff_true_positives) / n_total_positives
        ppr_points.append(ppr)
        tpr_points.append(tpr)
        shuff_ppr_points.append(shuff_ppr)
        shuff_tpr_points.append(shuff_tpr)
    ppr_points = np.array(ppr_points)
    tpr_points = np.array(tpr_points)
    shuff_ppr_points = np.array(shuff_ppr_points)
    shuff_tpr_points = np.array(shuff_ppr_points)
    auc = sklearn.metrics.auc(ppr_points, tpr_points)
    shuff_auc  = sklearn.metrics.auc(shuff_ppr_points, shuff_tpr_points)
    n_thresholds = len(thresholds)
    delta_auc = auc - shuff_auc
    return PprTprResults(tpr_points = tpr_points,
                         ppr_points = ppr_points,
                         shuff_ppr_points = shuff_ppr_points,
                         shuff_tpr_points = shuff_tpr_points,
                         n_total_positives = n_total_positives,
                         n_predicted_positives = n_edges,
                         n_thresholds = n_thresholds,
                         auc = auc,
                         shuff_auc = shuff_auc,
                         delta_auc = delta_auc)
