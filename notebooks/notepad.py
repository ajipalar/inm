"""
Write information on a NotePad

Usage

```
import notepad
pad = notepad.NotePad()
pad.write("Title")
pad.write(dict(a = 1, b = 2))
pad

data = pad.data

# Format output in a new way
pad.fmt_func = my_new_fmt_func # Define this
```
"""

def h(x):
    return "{:,}".format(x)

def is_all_elements_int(x):
    answer = True 
    for i in x:
        if not isinstance(i, int):
            answer = False
    return answer

def _default_fmt_func(i, tup):
    key, val = tup
    assert "\n" not in key
    if val == None:
        return key + "\n"
    else:
        if isinstance(val, tuple):
            if not is_all_elements_int(val):
                val = tuple(h(i) for i in val)
                s = f"{key}  :  ("
                for i in val:
                    s+= i +"; "
                s = s.removesuffix("; ")
                s+= ")\n"
                return s
            return "%-30s : %30s" % tup + "\n"
        elif isinstance(val, int):
            return f"{key}  :  {h(val)}\n"
        else:
            return f"{key}  :  {val}\n"

def default_fmt_func(i, tup):
    key, val = tup
    assert isinstance(key, str)
    if val == None:
        return key + "\n"
    elif type(val) == int: 
        val = h(val)
    elif (type(val) ==  tuple) and is_all_elements_int(val):
        val_str = "("
        for i in val:
            val_str += h(i) + "; "
        val_str = val_str.removesuffix("; ")
        val_str += ")"
        val = val_str
    width = 80
    key_len = len(key)
    val_len = len(str(val))
    assert key_len + val_len <= width
    gap = width - key_len - val_len
    return key + " " * gap + str(val) + "\n"

class NotePad:
    def __init__(self, fmt_func = default_fmt_func): 
        self.data = []
        self.fmt_func = fmt_func 
        self.indent = []
    def write(self, other):
        if isinstance(other, tuple):
            assert len(other) == 2
            assert isinstance(other[0], str)
            self.data.append(other)
            self.indent.append(0)
        elif isinstance(other, str):
            self.data.append((other, None))
            self.indent.append(0)
        elif isinstance(other, dict):
            for key, value in other.items():
                assert isinstance(key, str)
                self.data.append((key, value))
                self.indent.append(0)
        elif isinstance(other, list):
            for tup in other:
                assert isinstance(tup, tuple)
                assert len(tup) == 2
                assert isinstance(tup[0], str)
                self.indent.append(0)
            self.data = self.data + other    
        else:
            msg = f"type {type(other)} not supported"
            raise TypeError(msg)
    def clear(self):
        self.data = []
        self.indent = []
    def __repr__(self):
        if len(self.data) == 0:
            return "Empty NotePad at " + super().__repr__() 
        else:
            s = ""
            for i, tup in enumerate(self.data):
                s += " " * self.indent[i] + self.fmt_func(i, tup)
            return s.removesuffix("\n")
