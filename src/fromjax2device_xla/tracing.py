# When executing a jax program jax first traces the operations to create a computation graph. represented in jaxpre intermediate representation 
# This graph is then optimized and compiled for the target device (CPU/GPU/TPU).

import utils 
def gemm(a, b):
    

#implement the TraceNode logic
class TraceContext:
    var_count = 0
    operations = []
    inputs = []


    def new_var(self):
        pass
    def add_inputs(self, inputs):
        pass
    def add_operation(self, name, inputs, output):
        pass
    def dump_jaxpr(self):
        pass


class TraceNode:
    def __init__(self,ctx, shape, dtype=_'f32'):
        pass
    def __matmul__(self, other):
        pass
    def __add__(self, other):
        pass
    def __repr__(self):
        pass
        