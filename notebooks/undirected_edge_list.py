"""
Implementation for Binary undirected edges read from tables

1. Edges are unique.
2. Edge weights are 1, 0 edges are not represented. 
3. Edges cannot be self edges.
4. If any pair is in the input table the edge is represented

Edge ordering
- a > b
- 

u = EdgeList

"""
import pandas as pd
import numpy as np

class UndirectedEdge:
    def __init__(self, a, b):
        assert a != b
        self.a = a
        self.b = b
    def sort(self):
        if a > b:
            ...
        else:
            t = self.a
            self.a = self.b 
            self.b = t
        
class UndirectedEdgeList:
    def __init__(self):
        _init(self)

    def get_dense_all_pairs(self):
        nodes = list(self.nodes)
        N = len(nodes)
        al = []
        bl = []
        for i in range(N):
            for j in range(0, i):
                a = nodes[i]
                b = nodes[j]
                al.append(a)
                bl.append(b)
        df = pd.DataFrame({"auid" : al, "buid" : bl})
        u = UndirectedEdgeList()
        u.update_from_df(df)
        return u
 
    def read_csv(self, path,  a_colname="auid", b_colname="buid", edge_value_colname = None, sep="\t"):
        _read_csv(u=self, path=path, a_colname=a_colname, b_colname=b_colname, sep=sep)
    def read_example(self, name="biogrid"):
        _read_csv(self, _example_path[name])
    def reindex(self, reindexer, enforce_coverage = True, all_vs_all = False):
        """
        Return a new UndirectedEdgeList instance with all nodes
        reindexed based on reindexer
        """
        if not all_vs_all:
            _reindex(self, reindexer, enforce_coverage = enforce_coverage)
        else:
            _all_vs_all_reindex(self, reindexer, enforce_coverage = enforce_coverage)
    def to_csv(self, path, a_colname, b_colname, index, sep, header, edge_colname="value"):
        if self.edge_values:
            df = pd.DataFrame({a_colname : self.a_nodes, b_colname : self.b_nodes, edge_colname : self.edge_values})
        else:
            df = pd.DataFrame({a_colname : self.a_nodes, b_colname : self.b_nodes})
        df.to_csv(path, sep=sep, index = index, header = header)
    def update_properties(self):
        _update_properties(self)
    def update_from_df(self, df,a_colname="auid", b_colname="buid", edge_value_colname = None, multi_edge_value_merge_strategy = None):
        """
        df : a_colname b_colname edge_value_colname
        multi_edge_value_merge_strategy : "max", 
          how to handle repeated observation of the same edge.
        """
        _update_from_df(self, df, a_colname=a_colname, b_colname=b_colname, edge_value_colname = edge_value_colname, multi_edge_value_merge_strategy = multi_edge_value_merge_strategy) 
    def node_intersection(self, other):
        assert isinstance(other, UndirectedEdgeList)
        return self.nodes.intersection(other.nodes)
    def edge_identity_intersection(self, other, rebuild_dicts = True): # fix this
        assert isinstance(other, UndirectedEdgeList)
        intersecting_node_set = self.node_intersection(other)
        if len(intersecting_node_set) == 0:
            return set()
        if rebuild_dicts:
            self._build_edge_dict()
            other._build_edge_dict()
        if self._edge_dict is None:
            self._build_edge_dict()
        if other._edge_dict is None:
            other._build_edge_dict()
        return set(list(self._edge_dict.keys())).intersection(set(list(other._edge_dict.keys())))
    def _build_edge_dict(self):
        if self.n_nodes == 0:
            self.edge_dict = {}
        else:
            edge_dict = {}
            if self.edge_values is not None:
                def edge_getter_f(u, i):
                    return u.edge_values[i]
            else:
                def edge_getter_f(u, i):
                    return 1.
            for i, a in enumerate(self.a_nodes):
                b = self.b_nodes[i]
                edge = frozenset((a, b))
                val = edge_getter_f(self, i)
                assert edge not in edge_dict
                edge_dict[edge] = val
        self._edge_dict = edge_dict

    def __repr__(self):
        return _repr(self)
    def __getitem__(self, key):
        if self.edge_values is not None:
            return self.a_nodes[key], self.b_nodes[key], self.edge_values[key]
        else:
            return self.a_nodes[key], self.b_nodes[key]

def _update_from_df(u, df,a_colname="auid", b_colname="buid", edge_value_colname = None,
                    multi_edge_value_merge_strategy = None): 
    u.a_nodes = set(df[a_colname].values)
    u.b_nodes = set(df[b_colname].values)
    u.nodes = u.a_nodes.union(u.b_nodes)
    u.n_nodes = len(u.nodes)
    u.n_a_nodes = len(u.a_nodes)
    u.n_b_nodes = len(u.b_nodes)
    seen_edges = {}
    anodes = []
    bnodes = []
    for i, r in df.iterrows():
        a = r[a_colname]
        b = r[b_colname]
        if a != b: # No self edges
            assert isinstance(a, str)
            assert isinstance(b, str)
            edge = frozenset((a, b)) 
            if edge not in seen_edges:
                if edge_value_colname:
                   edge_value = r[edge_value_colname]
                else:
                    edge_value = 1
                anodes.append(a)
                bnodes.append(b)
            else:
                if edge_value_colname:
                    if multi_edge_value_merge_strategy == "max":
                        edge_value = max(r[edge_value_colname], seen_edges[edge])
                    else:
                        raise NotImplementedError
            seen_edges[edge] = edge_value
    # Update edges based on the strategy
    anodes = np.array(anodes)
    bnodes = np.array(bnodes)
    if edge_value_colname:
        vals = []
        for i, a in enumerate(anodes):
            b = bnodes[i]
            assert a != b
            edge = frozenset((a, b))
            vals.append(seen_edges[edge])
        assert len(vals) == len(anodes)
        u.edge_values = vals 
    assert len(anodes) == len(bnodes)
    u.a_nodes = anodes 
    u.b_nodes = bnodes 
    u.update_properties()

def _init(u):
    u.nedges = 0 
    u.a_nodes = set()
    u.b_nodes = set()
    u.nodes = set()
    u.update_properties()
    u.edge_values = None 
    u._edge_dict = None

def _update_properties(u):
    a_node_set = set(u.a_nodes)
    b_node_set = set(u.b_nodes)
    a_b = a_node_set.union(b_node_set)
    u.n_nodes = len(a_b)
    u.nodes = a_b
    u.n_a_nodes = len(a_node_set)
    u.n_b_nodes = len(b_node_set)
    assert len(u.a_nodes) == len(u.b_nodes)
    u.nedges = len(u.a_nodes)

def _reindex(u, reindexer, enforce_coverage):
    """
    Return a new UndirectedEdgeList instance with all nodes
    reindexed based on reindexer
    """
    if isinstance(reindexer, dict):
        if enforce_coverage:
            assert len(set(reindexer.keys()).union(u.nodes)) == u.n_nodes
        anew = []
        bnew = []
        if enforce_coverage:
            for i, a in enumerate(u.a_nodes):
                b = u.b_nodes[i]
                anew.append(reindexer[a])
                bnew.append(reindexer[b])
        else:
            for i, a in enumerate(u.a_nodes):
                b = u.b_nodes[i]
                if (a in reindexer) and (b in reindexer):
                    anew.append(reindexer[a])
                    bnew.append(reindexer[b])
        u.a_nodes = anew
        u.b_nodes = bnew
        u.update_properties()
    else:
        raise ValueError

def _all_vs_all_reindex(u, reindexer, enforce_coverage):
    if isinstance(reindexer, dict):
        if enforce_coverage:
            assert len(set(reindexer.keys()).union(u.nodes)) == u.n_nodes
        for key, l in reindexer.items():
            assert isinstance(l, list)

        if u.edge_values:
            def vals_updater(i, u):
                vals.append(u.edge_values[i])
        else:
            def vals_updater(i, u):
                ...
        def common(i, u, reindexer, a, b):
            for x, ra in enumerate(reindexer[a]):
                for y, rb in enumerate(reindexer[b]):
                    anew.append(ra)
                    bnew.append(rb)
                    vals_updater(i, u)
        anew = []
        bnew = []
        vals = []
        if enforce_coverage:
            for i, a in enumerate(u.a_nodes):
                len_a = len(a)
                b = u.b_nodes[i]
                len_b = len(b)
                common(i, u, reindexer, a, b)
        else:
            for i, a in enumerate(u.a_nodes):
                len_a = len(a)
                b = u.b_nodes[i]
                len_b = len(b)
                if (a in reindexer) and (b in reindexer):
                    common(i, u, reindexer, a, b)
        u.a_nodes = anew
        u.b_nodes = bnew
        if u.edge_values:
            u.edge_values = vals
        u.update_properties()
    else:
        raise ValueError

def _repr(u):
    if u.n_a_nodes > 0: 
        s = f"""N nodes {u.n_nodes}
      N a IDs {u.n_a_nodes}
        {u.a_nodes[0]}, ..., {u.a_nodes[-1]}
      N b Ids {u.n_b_nodes}
        {u.b_nodes[0]}, ..., {u.b_nodes[-1]}
N edges {u.nedges}"""
    else:
        s = f"""N nodes {u.n_nodes}
      N a IDs {u.n_a_nodes}
        {u.a_nodes}
      N b Ids {u.n_b_nodes}
        {u.b_nodes}
N edges {u.nedges}"""
    return s

def _read_csv(u, path, a_colname="auid", b_colname="buid", edge_value_colname = None, sep="\t"):
    df = pd.read_csv(path, sep=sep)
    u.update_from_df(df, a_colname = a_colname, b_colname = b_colname, edge_value_colname = edge_value_colname)

_example_path = {"biogrid_ref" : "../data/processed/references/biogrid_reference.tsv",
                 "huri_ref": "../data/processed/references/HuRI_reference.tsv",
                 "huMAP_medium_ref" : "../data/processed/references/humap2_ppis_medium.tsv",
                 "huMAP_high_ref" : "../data/processed/references/humap2_ppis_high.tsv",
                 "cullin" : "../data/processed/cullin/composite_table.tsv"}