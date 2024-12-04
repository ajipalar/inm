class AnyId:
    """
    An AnyId reppresents a set of identifiers

    If a concrete identifer is in the set it equals the any_id instance.
    If any ident
    """
    def __init__(self, items):
        if isinstance(items, str):
            self.ids = {items : None}
        elif isinstance(items, int):
            self.ids = {items : None}
        else:
            ids = {}
            for i in items:
                ids[i] = None
            self.ids = ids

    def __eq__(self, other):
        if isinstance(other, AnyId):
            for i in other.ids:
                if i in self.ids:
                    return True
            return False
        else: 
            return True if other in self.ids else False
    def __add__(self, other):
        if isinstance(other, AnyId):
            return AnyId(self.ids | other.ids)
        else:
            return AnyId(self.ids | {other : None})

class AnyIdRegistry:
    """
    1. Add a group of identifiers
    2. Every AnyId for equality
    3. If equality update the AnyId and stop
       else add the AnyId to the list
    """
    def __init__(self):
        self.any_id_lst = []
    def add_id(self, identifier_group): 
        if not isinstance(identifier_group, AnyId):
            identifier_group = AnyId(identifier_group)
        found = False
        for i, any_id in enumerate(self.any_id_lst):
            if identifier_group == any_id:
                self.any_id_lst[i] = any_id + identifier_group
                found = True
                break
        if not found:
            self.any_id_lst.append(identifier_group)
    def create_registry(self):
        registry = {}
        for idx, any_id in enumerate(self.any_id_lst):
            for identifier in any_id.ids:
                registry[identifier] = idx
        self.registry = registry
    def __getitem__(self, node_id):
        return self.any_id_lst[self.registry[node_id]] 

def _create_example():
    reg = AnyIdRegistry()
    for key, val in _examples.items():
        reg.add_id(val)
    reg.create_registry()
    return reg

_examples = {0: (0, 1), 1 : (1, 2, 3), 3: (9, 10), 4: (11, 13), 5: ("d", "asdf", "ls")}

