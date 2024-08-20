class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = data
        self.grad = 0
        self.backward = lambda: None
        self.parents = parents
        self.op = op
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(other.data + self.data, parents=(self, other), op="+")
        def backward():
            self.grad += out.grad
            other.grad += out.grad
            self.backward()
            other.backward()
        out.backward = backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, parents=(self, other), op="*")
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            self.backward()
            other.backward()
        out.backward = backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __pow__(self, pow):
        assert isinstance(pow, (float, int)), "power must be a float/int"
        out = Value(self.data ** pow, parents=(self, pow), op="^")
        def backward():
            self.grad += (pow * (self.data ** (pow - 1))) * out.grad
            self.backward()
        out.backward = backward
        return out

    def __sub__(self, other):
        return self + -other
    
    def __neg__(self):
        return self * -1
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, op={self.op})"
        
        
def print_map(value):
    print(value.data)
    ops = [value.op]
    parents = [value.parents]
    while len(parents) > 0:
        _ops = []
        _parents = []
        printables = []
        for pair_idx, pair in enumerate(parents):
            if not pair: continue
            a = pair[0].data if isinstance(pair[0], Value)  else pair[0]
            b = pair[1].data if isinstance(pair[1], Value)  else pair[1]
            printables.append(f"{a} {ops[pair_idx]} {b}")
            _ops.extend([p.op for p in pair if isinstance(p, Value)])
            for p in pair:
                if not isinstance(p, Value): continue
                _parents.append(p.parents)
        print(" | ".join(printables))
        ops = _ops
        parents = _parents
