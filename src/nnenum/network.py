'''
Stanley Bak

Network container classes for nnenum
'''

import numpy as np
import onnx
from scipy.signal import convolve2d, fftconvolve

from nnenum.util import Freezable
from nnenum.timerutil import Timers

def _hwc_to_chw_permutation(h, w, c):
    '''Compute the permutation that reorders HWC-flat indices to CHW-flat indices.

    After Conv layers, nnenum stores star/zono data in HWC-flat order.
    ONNX FC weight matrices (MatMul/Gemm) index inputs in CHW-flat order.
    This permutation maps CHW-flat position k to the HWC-flat index of the same element,
    so that `data[perm]` converts from HWC-flat to CHW-flat row ordering.
    '''
    perm = np.empty(h * w * c, dtype=int)
    for ch in range(c):
        for r in range(h):
            for col in range(w):
                chw_idx = ch * h * w + r * w + col
                hwc_idx = r * w * c + col * c + ch
                perm[chw_idx] = hwc_idx
    return perm

class NeuralNetwork(Freezable):
    'neural network container'

    def __init__(self, layers, dag_predecessors=None):
        '''
        layers: list of layer objects
        dag_predecessors: optional dict mapping layer_num -> [input_layer_nums]
            Only needed for layers with non-sequential inputs (e.g. SkipAddLayer).
            Sequential layers (each layer takes the output of the previous one) do not
            need entries here.  Example: {5: [2, 4]} means layer 5 takes inputs from
            layers 2 and 4.
        '''
    def __init__(self, layers, dag_predecessors=None):
        '''
        layers: list of layer objects
        dag_predecessors: optional dict mapping layer_num -> [input_layer_nums]
            Only needed for layers with non-sequential inputs (e.g. SkipAddLayer).
            Sequential layers (each layer takes the output of the previous one) do not
            need entries here.  Example: {5: [2, 4]} means layer 5 takes inputs from
            layers 2 and 4.
        '''

        assert layers, "layers should be a non-empty list"

        for i, layer in enumerate(layers):
            assert layer.layer_num == i, f"Layer {i} has incorrect layer num: {layer.layer_num}: {layer}"

        # dag_predecessors maps layer_num -> list of input layer_nums
        # Empty dict means fully sequential network
        self.dag_predecessors = dag_predecessors if dag_predecessors is not None else {}

        # True when the network has multi-branch fan-out (BranchRestoreLayer present).
        # star.lp overapprox is unsupported for such networks because the parameter space
        # (LP columns) differs across paths and cannot be trivially merged at SkipAddLayer.
        self.has_multi_branch = any(isinstance(l, BranchRestoreLayer) for l in layers)

        self.layers = layers
        self.check_io()

        for layer in layers:
            layer.network = self

        self.freeze_attrs()

    def __str__(self):
        return f'[NeuralNetwork with {len(self.layers)} layers with {self.layers[0].get_input_shape()} input and ' + \
          f'{self.get_output_shape()} output]'

    def num_relu_layers(self):
        'count the number of relu layers'

        rv = 0

        for l in self.layers:
            if isinstance(l, ReluLayer):
                rv += 1

        return rv

    def num_relu_neurons(self):
        'count the number of relu neurons'

        rv = 0

        for l in self.layers:
            if isinstance(l, ReluLayer):
                count = 1

                for dim in l.shape:
                    count *= dim

                rv += count

        return rv

    def get_input_shape(self):
        'get the input shape to the first layer'

        return self.layers[0].get_input_shape()

    def get_output_shape(self):
        'get the output shape from the last layer'

        return self.layers[-1].get_output_shape()

    def chw_to_hwc_init_box(self, init_box):
        '''Convert a flat init_box from CHW order (ONNX/vnnlib convention) to HWC order (nnenum internal convention).

        vnnlib specifies input bounds indexed in CHW-flat order.
        nnenum's internal layers expect HWC layout.
        This function reorders the rows of init_box so that index i refers to the i-th element in HWC-flat order.

        Only applied when the first layer has a 3D (H, W, C) input shape.
        For 1D inputs (FC networks), no reordering is needed.

        Returns a new array with the same shape as init_box, reordered if necessary.
        '''

        input_shape = self.get_input_shape()

        if len(input_shape) != 3:
            return init_box  # 1D input: no reordering needed

        h, w, c = input_shape
        # CHW-flat index i corresponds to: channel = i // (h*w), row = (i % (h*w)) // w, col = i % w
        # HWC-flat index j corresponds to: row = j // (w*c), col = (j % (w*c)) // c, channel = j % c
        # Build the permutation: for each HWC-flat index j, find the CHW-flat index of the same element.
        # HWC element (r, col, ch) has HWC-flat index = r*w*c + col*c + ch
        #                           and CHW-flat index = ch*h*w + r*w + col
        chw_indices = np.empty(h * w * c, dtype=int)
        for r in range(h):
            for col in range(w):
                for ch in range(c):
                    hwc_idx = r * w * c + col * c + ch
                    chw_idx = ch * h * w + r * w + col
                    chw_indices[hwc_idx] = chw_idx

        return init_box[chw_indices]

    def get_num_inputs(self):
        'get the scalar number of inputs'

        shape = self.get_input_shape()

        rv = 1

        for x in shape:
            rv *= x

        return rv

    def get_num_outputs(self):
        'get the scalar number of outputs'

        shape = self.get_output_shape()

        rv = 1

        for x in shape:
            rv *= x

        return rv

    def execute(self, input_vec, save_branching=False):
        '''execute the neural network with the given input vector

        if save_branching is True, returns (output, branch_list), where branch_list contains one list for each layer,
            and each layer-list is a list of the branching decisions taken by each neuron. For layers with ReLUs, this
            will be True/False values (True if positive branch is taken), for max pooling layers these will be ints, or
            lists of ints (if multiple max values are equal)

        otherwise, just returns output
        '''

        if save_branching:
            branch_list = []

        state = input_vec.copy() # test with float32 dtype?


        if state.shape != self.get_input_shape():
            state = nn_unflatten(state, self.get_input_shape())

        # For DAG networks with skip connections, cache activations keyed by
        # layer index so SkipAddLayer can fetch the skip-path activation.
        # activation_cache[k] = state immediately before processing layer k.
        activation_cache = {}

        for layer_idx, layer in enumerate(self.layers):
            # Cache current state before processing this layer (matches star_cache semantics)
            if self.dag_predecessors:
                activation_cache[layer_idx] = nn_flatten(state).copy()

            if isinstance(layer, BranchRestoreLayer):
                # Restore state from the cached checkpoint
                src_key = self.dag_predecessors[layer_idx][0]
                state = nn_unflatten(activation_cache[src_key], layer.get_output_shape())
                if save_branching:
                    branch_list.append([])
            elif isinstance(layer, SkipAddLayer):
                # Retrieve the skip-path activation from cache (at branch point)
                skip_cache_key = self.dag_predecessors[layer_idx][0]
                skip_state = activation_cache[skip_cache_key]
                skip_state = nn_unflatten(skip_state, layer.skip_branch_shape)
                state = layer.execute(skip_state, state)
                if save_branching:
                    branch_list.append([])
            elif save_branching and (isinstance(layer, ReluLayer) or isinstance(layer, PoolingLayer)):
                state, layer_branch_list = layer.execute(state, save_branching=True)
                branch_list.append(layer_branch_list)
            else:
                if save_branching:
                    branch_list.append([])

                assert state.shape == layer.get_input_shape(), \
                    f"Layer {layer_idx} ({layer.__class__.__name__}): state shape {state.shape} != input shape {layer.get_input_shape()}"
                assert state.shape == layer.get_input_shape(), \
                    f"Layer {layer_idx} ({layer.__class__.__name__}): state shape {state.shape} != input shape {layer.get_input_shape()}"
                state = layer.execute(state)
                assert state.shape == layer.get_output_shape()

        assert state.shape == self.get_output_shape()

        rv = (state, branch_list) if save_branching else state


        return rv

    def check_io(self):
        'check the neural network for input / output compatibility'

        for i, layer in enumerate(self.layers):
            if i == 0:
                continue

            # SkipAddLayer and BranchRestoreLayer have non-sequential inputs tracked
            # in dag_predecessors; skip the simple sequential shape check for them.
            if isinstance(layer, (SkipAddLayer, BranchRestoreLayer)):
                continue

            prev_output_shape = self.layers[i-1].get_output_shape()
            my_input_shape = layer.get_input_shape()

            assert prev_output_shape == my_input_shape, f"output of layer {i-1} was {prev_output_shape}, " + \
              f"and this doesn't match input of layer {i} which is {my_input_shape}"

class ReluLayer(Freezable):
    'relu layer'

    def __init__(self, layer_num, shape, filter_func=None):
        '''
        filter_func(i) returns True if output i should have a relu branch
        '''

        self.layer_num = layer_num
        self.shape = shape

        self.filter_func = filter_func # returns True if relu should be applied for neuron i

    def __str__(self):
        return f'[ReluLayer with shape {self.shape}]'

    def get_input_shape(self):
        'get the input shape to this layer'

        return self.shape

    def get_output_shape(self):
        'get the output shape from this layer'

        return self.shape

    def execute(self, state, save_branching=False):
        '''execute the layer on a concrete state

        if save_branching is True, returns (output, branch_list), where branch_list is a list of booleans for each
            neuron in the layer that is True if the nonnegative branch of the ReLU was taken, False if negative

        otherwise, just returns output
        '''

        Timers.tic('execute relu')

        if save_branching:
            branch_list = []

        assert state.shape == self.get_input_shape(), f"state shape to fully connected layer was {state.shape}, " + \
            f"expected {self.get_input_shape()}"

        state = nn_flatten(state)

        if save_branching:
            for i, val in enumerate(state):
                if self.filter_func is not None:
                    if not self.filter_func(i):
                        continue

                branch_list.append(val >= 0)

        if self.filter_func is None:
            state = np.clip(state, 0, np.inf)
        else:
            res = []

            for i, val in enumerate(state):
                if not self.filter_func(i):
                    res.append(val)
                else:
                    res.append(max(0, val))

            state = np.array(res, dtype=float)

        rv = nn_unflatten(state, self.shape)

        rv = (rv, branch_list) if save_branching else rv

        Timers.toc('execute relu')

        return rv

class ConstantLayer(Freezable):
    'constant onnx layer - outputs a fixed value'

    def __init__(self, layer_num, value):

        self.layer_num = layer_num
        self.value = value
        self.network = None # populated later when constructing network

        # Constant has no input shape, output shape is the value's shape
        self.input_shape = None
        self.output_shape = value.shape

        self.freeze_attrs()

    def __str__(self):
        return f'[Constant with output shape {self.get_output_shape()}]'

    def get_input_shape(self):
        'get the input shape to this layer'

        return self.input_shape

    def get_output_shape(self):
        'get the output shape from this layer'

        return self.output_shape

    def transform_star(self, star):
        'transform the star for this layer'

        # Constant replaces the current state with a fixed value
        # This means the star becomes a single point (the constant)
        # The generator columns become zero, bias becomes the constant
        star.a_mat = None
        star.bias = nn_flatten(self.value)

    def transform_zono(self, zono):
        'transform the zono for this layer'

        # Constant replaces state with fixed value
        # Center becomes the constant, all generators become zero
        zono.mat_t = None
        zono.center = nn_flatten(self.value)

    def transform_deeppoly(self, deeppoly):
        'transform the deeppoly for this layer'

        # Constant replaces state - bounds become the constant value
        flattened = nn_flatten(self.value)
        deeppoly.lbs = flattened.copy()
        deeppoly.ubs = flattened.copy()

    def execute(self, state):
        '''execute the layer on a concrete state

        returns output (ignores input, always returns constant)
        '''

        return self.value.copy()

class FlattenLayer(Freezable):
    'flatten onnx layer'

    def __init__(self, layer_num, input_shape):

        self.layer_num = layer_num
        self.input_shape = input_shape
        self.network = None # populated later when constructing network

        os = 1

        for i in input_shape:
            os *= i

        self.output_shape = (os, )

        self.freeze_attrs()

    def __str__(self):
        return f'[Flatten with input {self.get_input_shape()}]'

    def get_input_shape(self):
        'get the input shape to this layer'

        return self.input_shape

    def get_output_shape(self):
        'get the output shape from this layer'

        return self.output_shape

    def transform_star(self, star):
        'transform the star for this layer'

        if len(self.input_shape) == 3:
            # Input is 3D HWC. After Conv, star rows are in HWC-flat order.
            # FC weight matrices (next layer) expect CHW-flat ordering.
            # Permute rows to convert HWC-flat to CHW-flat.
            h, w, c = self.input_shape
            perm = _hwc_to_chw_permutation(h, w, c)
            star.bias = star.bias[perm]
            if star.a_mat is not None:
                from scipy.sparse import issparse
                if issparse(star.a_mat):
                    star.a_mat = star.a_mat.tocsr()[perm, :]
                else:
                    star.a_mat = star.a_mat[perm, :]

    def transform_zono(self, zono):
        'transform the zono for this layer'

        if len(self.input_shape) == 3:
            h, w, c = self.input_shape
            perm = _hwc_to_chw_permutation(h, w, c)
            zono.center = zono.center[perm]
            if zono.mat_t is not None:
                zono.mat_t = zono.mat_t[perm, :]

    def transform_deeppoly(self, deeppoly):
        'transform the deeppoly for this layer'

        # do nothing

    def execute(self, state):
        '''execute the layer on a concrete state

        returns output
        '''

        # ONNX Flatten expects CHW format, but nnenum uses HWC
        # Need to convert HWC (H, W, C) → CHW (C, H, W) before flattening
        if len(state.shape) == 3:
            # Convert from HWC to CHW
            state = np.transpose(state, (2, 0, 1))

        rv = nn_flatten(state)
        assert rv.shape == self.output_shape

        return rv

class ReshapeLayer(Freezable):
    'reshape onnx layer'

    def __init__(self, layer_num, new_shape, input_shape):

        self.layer_num = layer_num
        self.input_shape = input_shape
        self.new_shape = new_shape
        self.network = None # populated later when constructing network

        # Validate that total elements match
        input_size = 1
        for i in input_shape:
            input_size *= i

        output_size = 1
        dynamic_dim_index = None
        for i, dim in enumerate(new_shape):
            if dim == -1:
                assert dynamic_dim_index is None, "only one dimension can be -1 in reshape"
                dynamic_dim_index = i
            else:
                output_size *= dim

        # If there's a -1 dimension, infer its size
        if dynamic_dim_index is not None:
            inferred_size = input_size // output_size
            assert input_size == output_size * inferred_size, "reshape dimensions don't match input size"
            new_shape = list(new_shape)
            new_shape[dynamic_dim_index] = inferred_size
            new_shape = tuple(new_shape)
            self.new_shape = new_shape
        else:
            assert input_size == output_size, f"reshape size mismatch: input {input_size} vs output {output_size}"

        self.freeze_attrs()

    def __str__(self):
        return f'[Reshape from {self.get_input_shape()} to {self.get_output_shape()}]'

    def get_input_shape(self):
        'get the input shape to this layer'

        return self.input_shape

    def get_output_shape(self):
        'get the output shape from this layer'

        # For 3D shapes, return HWC format (transpose of stored CHW)
        if len(self.new_shape) == 3:
            # CHW → HWC: (C, H, W) → (H, W, C)
            return (self.new_shape[1], self.new_shape[2], self.new_shape[0])
        else:
            return self.new_shape

    def transform_star(self, star):
        'transform the star for this layer'

        if len(self.input_shape) == 3 and len(self.new_shape) == 1:
            # Flattening from 3D HWC to 1D: permute rows HWC-flat → CHW-flat
            # so subsequent FC weight matrices (CHW-indexed) are applied correctly.
            h, w, c = self.input_shape
            perm = _hwc_to_chw_permutation(h, w, c)
            star.bias = star.bias[perm]
            if star.a_mat is not None:
                from scipy.sparse import issparse
                if issparse(star.a_mat):
                    star.a_mat = star.a_mat.tocsr()[perm, :]
                else:
                    star.a_mat = star.a_mat[perm, :]
        # For other cases (flat→flat, flat→3D), no row permutation needed.

    def transform_zono(self, zono):
        'transform the zono for this layer'

        if len(self.input_shape) == 3 and len(self.new_shape) == 1:
            h, w, c = self.input_shape
            perm = _hwc_to_chw_permutation(h, w, c)
            zono.center = zono.center[perm]
            if zono.mat_t is not None:
                zono.mat_t = zono.mat_t[perm, :]

    def transform_deeppoly(self, deeppoly):
        'transform the deeppoly for this layer'

        # do nothing - reshape doesn't change the abstract representation

    def execute(self, state):
        '''execute the layer on a concrete state

        returns output
        '''

        if len(self.new_shape) == 3:
            # 3D output: ONNX target shape is CHW. Reshape to CHW then transpose to HWC.
            state_chw = state.reshape(self.new_shape)
            rv = np.transpose(state_chw, (1, 2, 0))
        elif len(state.shape) == 3:
            # Flattening from 3D HWC input: must convert HWC→CHW before flattening,
            # so the flat ordering matches what ONNX FC weights expect (CHW-indexed).
            state_chw = np.transpose(state, (2, 0, 1))
            rv = state_chw.reshape(self.new_shape)
        else:
            rv = state.reshape(self.new_shape)
            assert rv.shape == self.new_shape

        return rv

class AddLayer(Freezable):
    'add onnx layer'

    def __init__(self, layer_num, vec):

        self.layer_num = layer_num
        self.vec = vec
        self.network = None # populated later when constructing network

        self.freeze_attrs()

    def __str__(self):
        return f'[AddLayer with shape {self.get_input_shape()}]'

    def get_input_shape(self):
        'get the input shape to this layer'

        return self.vec.shape

    def get_output_shape(self):
        'get the output shape from this layer'

        return self.vec.shape

    def transform_star(self, star):
        'transform the star'

        # well, hope star.bias is flat?

        star.bias += nn_flatten(self.vec)

    def transform_zono(self, zono):
        'transform the zono'

        zono.center += nn_flatten(self.vec)

    def transform_deeppoly(self, deeppoly):
        'apply the linear transformation part of the layer to the passed-in deeppoly weights (not relu)'
        ubconst_nl = nn_flatten(self.vec)  # upper bounds constants of new layer
        lbconst_nl = nn_flatten(self.vec)  # lower bounds constants of new layer

        # back substitution
        updated_ubconst_nl = deeppoly.ubconst + ubconst_nl

        updated_lbconst_nl = deeppoly.ubconst + lbconst_nl

        deeppoly.ubconst = updated_ubconst_nl
        deeppoly.lbconst = updated_lbconst_nl

        deeppoly.ubs = np.where(deeppoly.ubcoef >= 0, deeppoly.ubcoef, 0) @ deeppoly.inputbounds[:, 1]
        deeppoly.ubs += np.where(deeppoly.ubcoef < 0, deeppoly.ubcoef, 0) @ deeppoly.inputbounds[:, 0]
        deeppoly.ubs += deeppoly.ubconst
        deeppoly.lbs = np.where(deeppoly.lbcoef >= 0, deeppoly.lbcoef, 0) @ deeppoly.inputbounds[:, 0]
        deeppoly.lbs += np.where(deeppoly.lbcoef < 0, deeppoly.lbcoef, 0) @ deeppoly.inputbounds[:, 1]
        deeppoly.lbs += deeppoly.lbconst


    def execute(self, state):
        '''execute on a concrete state

        returns output
        '''

        return state + self.vec

class ScaleLayer(Freezable):
    'element-wise scale and shift: y = scale * x + shift (e.g. standalone BatchNorm)'

    def __init__(self, layer_num, scale, shift, input_shape):
        assert scale.shape == shift.shape
        assert scale.ndim == 1

        self.layer_num = layer_num
        self.scale = scale      # 1D, flattened order matching the layer's flat representation
        self.shift = shift      # 1D
        self.input_shape = input_shape
        self.network = None

        self.freeze_attrs()

    def __str__(self):
        return f'[ScaleLayer with shape {self.input_shape}]'

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.input_shape

    def transform_star(self, star):
        star.a_mat = star.a_mat * self.scale[:, np.newaxis]
        star.bias = star.bias * self.scale + self.shift

    def transform_zono(self, zono):
        zono.mat_t = zono.mat_t * self.scale[:, np.newaxis]
        zono.center = zono.center * self.scale + self.shift

    def transform_deeppoly(self, deeppoly):
        # ScaleLayer is a diagonal linear map: y = diag(scale)*x + shift
        pos = np.maximum(self.scale, 0)
        neg = np.minimum(self.scale, 0)
        new_ubcoef = pos[:, np.newaxis] * deeppoly.ubcoef + neg[:, np.newaxis] * deeppoly.lbcoef
        new_lbcoef = pos[:, np.newaxis] * deeppoly.lbcoef + neg[:, np.newaxis] * deeppoly.ubcoef
        new_ubconst = pos * deeppoly.ubconst + neg * deeppoly.lbconst + self.shift
        new_lbconst = pos * deeppoly.lbconst + neg * deeppoly.ubconst + self.shift
        deeppoly.ubcoef = new_ubcoef
        deeppoly.lbcoef = new_lbcoef
        deeppoly.ubconst = new_ubconst
        deeppoly.lbconst = new_lbconst
        deeppoly.ubs = (np.maximum(deeppoly.ubcoef, 0) @ deeppoly.inputbounds[:, 1]
                       + np.minimum(deeppoly.ubcoef, 0) @ deeppoly.inputbounds[:, 0]
                       + deeppoly.ubconst)
        deeppoly.lbs = (np.maximum(deeppoly.lbcoef, 0) @ deeppoly.inputbounds[:, 0]
                       + np.minimum(deeppoly.lbcoef, 0) @ deeppoly.inputbounds[:, 1]
                       + deeppoly.lbconst)

    def execute(self, state):
        scale_shaped = self.scale.reshape(self.input_shape)
        shift_shaped = self.shift.reshape(self.input_shape)
        return state * scale_shaped + shift_shaped

class MatMulLayer(Freezable):
    'onnx matmul layer'

    def __init__(self, layer_num, mat, prev_layer_output_shape=None):

        assert prev_layer_output_shape is None or isinstance(prev_layer_output_shape, tuple)

        self.layer_num = layer_num
        self.mat = mat
        self.prev_layer_output_shape = prev_layer_output_shape
        self.network = None # populated later when constructing network

        assert len(mat.shape) == 2

        if prev_layer_output_shape is not None:
            expected_inputs = 1

            for x in prev_layer_output_shape:
                expected_inputs *= x

            assert expected_inputs == mat.shape[1], f"MatMulLayer matrix shape was {mat.shape}, but " + \
                f"prev_layer_output_shape {prev_layer_output_shape} needs {expected_inputs} columns"

        self.freeze_attrs()

    def __str__(self):
        return f'[MatMulLayer with {self.get_input_shape()} input and {self.get_output_shape()} output]'

    def get_input_shape(self):
        'get the input shape to this layer'

        rv = self.prev_layer_output_shape

        if rv is None:
            rv = (self.mat.shape[1],)

        return rv

    def get_output_shape(self):
        'get the output shape from this layer'

        return (self.mat.shape[0],)

    def transform_star(self, star):
        'apply on star'

        from scipy.sparse import issparse
        # Use @ directly when sparse — dense @ sparse → dense without materializing sparse
        star.a_mat = self.mat @ star.a_mat
        star.bias = np.dot(self.mat, star.bias)

    def transform_zono(self, zono):
        'apply on zono'

        zono.mat_t = np.dot(self.mat, zono.mat_t)
        zono.center = np.dot(self.mat, zono.center)

    def transform_deeppoly(self, deeppoly):
        'apply the linear transformation part of the layer to the passed-in deeppoly weights (not relu)'
        ubcoef_nl = self.mat  # upper bounds coefficients of new layer
        lbcoef_nl = self.mat  # lower bounds coefficients of new layer

        # back substitution
        updated_ubcoef_nl = np.where(ubcoef_nl >= 0, ubcoef_nl, 0) @ deeppoly.ubcoef
        updated_ubcoef_nl += np.where(ubcoef_nl < 0, ubcoef_nl, 0) @ deeppoly.lbcoef
        updated_ubconst_nl = np.where(ubcoef_nl >= 0, ubcoef_nl, 0) @ deeppoly.ubconst
        updated_ubconst_nl += np.where(ubcoef_nl < 0, ubcoef_nl, 0) @ deeppoly.lbconst

        updated_lbcoef_nl = np.where(lbcoef_nl >= 0, lbcoef_nl, 0) @ deeppoly.lbcoef
        updated_lbcoef_nl += np.where(lbcoef_nl < 0, lbcoef_nl, 0) @ deeppoly.ubcoef
        updated_lbconst_nl = np.where(lbcoef_nl >= 0, lbcoef_nl, 0) @ deeppoly.lbconst
        updated_lbconst_nl += np.where(lbcoef_nl < 0, lbcoef_nl, 0) @ deeppoly.ubconst

        deeppoly.ubcoef = updated_ubcoef_nl
        deeppoly.ubconst = updated_ubconst_nl
        deeppoly.lbcoef = updated_lbcoef_nl
        deeppoly.lbconst = updated_lbconst_nl

        deeppoly.ubs = np.where(deeppoly.ubcoef >= 0, deeppoly.ubcoef, 0) @ deeppoly.inputbounds[:, 1]
        deeppoly.ubs += np.where(deeppoly.ubcoef < 0, deeppoly.ubcoef, 0) @ deeppoly.inputbounds[:, 0]
        deeppoly.ubs += deeppoly.ubconst
        deeppoly.lbs = np.where(deeppoly.lbcoef >= 0, deeppoly.lbcoef, 0) @ deeppoly.inputbounds[:, 0]
        deeppoly.lbs += np.where(deeppoly.lbcoef < 0, deeppoly.lbcoef, 0) @ deeppoly.inputbounds[:, 1]
        deeppoly.lbs += deeppoly.lbconst


    def execute(self, state):
        '''execute on a concrete state

        returns output
        '''

        Timers.tic('execute matmul')

        assert state.shape == self.get_input_shape(), f"state shape to matmul was {state.shape}, " + \
            f"expected {self.get_input_shape()}"

        state = nn_flatten(state)

        rv = np.dot(self.mat, state)

        assert rv.shape == self.get_output_shape()

        Timers.toc('execute matmul')

        return rv

class FullyConnectedLayer(Freezable):
    'fully connected layer'

    def __init__(self, layer_num, weights, biases, prev_layer_output_shape=None):

        assert prev_layer_output_shape is None or isinstance(prev_layer_output_shape, tuple)

        if isinstance(weights, list):
            weights = np.array(weights, dtype=float)

        if isinstance(biases, list):
            biases = np.array(biases, dtype=float)

        self.layer_num = layer_num
        self.weights = weights
        self.biases = biases
        self.prev_layer_output_shape = prev_layer_output_shape

        self.network = None

        assert biases.shape[0] == weights.shape[0], "biases vec in layer " + \
            f"{layer_num} has length {biases.shape[0]}, but weights matrix has height " + \
            f"{weights.shape[0]}"

        assert len(biases.shape) == 1, f'expected 1-d bias vector at layer {layer_num}, got {biases.shape}'
        assert len(weights.shape) == 2

        if prev_layer_output_shape is not None:
            expected_inputs = 1

            for x in prev_layer_output_shape:
                expected_inputs *= x

            assert expected_inputs == weights.shape[1], f"FC Layer weight matrix shape was {weights.shape}, but " + \
                f"prev_layer_output_shape {prev_layer_output_shape} needs {expected_inputs} columns"

        self.freeze_attrs()

    def __str__(self):
        return f'[FullyConnectedLayer with {self.get_input_shape()} input and {self.get_output_shape()} output]'

    def get_input_shape(self):
        'get the input shape to this layer'

        rv = self.prev_layer_output_shape

        if rv is None:
            rv = (self.weights.shape[1],)

        return rv

    def get_output_shape(self):
        'get the output shape from this layer'

        return (self.weights.shape[0],)

    def transform_star(self, star):
        'apply the linear transformation part of the layer to the passed-in lp_star (not relu)'

        if star.a_mat is None:
            star.a_mat = self.weights.copy()
        else:
            # Use @ directly — dense @ sparse → dense without materializing sparse input
            star.a_mat = self.weights @ star.a_mat

        if star.bias is None:
            star.bias = self.biases.copy()
        else:
            star.bias = np.dot(self.weights, star.bias) + self.biases

    def transform_zono(self, zono):
        'apply the linear transformation part of the layer to the passed-in zonotope (not relu)'

        zono.mat_t = np.dot(self.weights, zono.mat_t)
        zono.center = np.dot(self.weights, zono.center) + self.biases

    def transform_deeppoly(self, deeppoly):
        'apply the linear transformation part of the layer to the passed-in deeppoly weights (not relu)'

        ubcoef_nl = self.weights  # upper bounds coefficients of new layer
        ubconst_nl = self.biases  # upper bounds constants of new layer
        lbcoef_nl = self.weights  # lower bounds coefficients of new layer
        lbconst_nl = self.biases  # lower bounds constants of new layer

        # back substitution
        updated_ubcoef_nl = np.where(ubcoef_nl >= 0, ubcoef_nl, 0) @ deeppoly.ubcoef
        updated_ubcoef_nl += np.where(ubcoef_nl < 0, ubcoef_nl, 0) @ deeppoly.lbcoef
        updated_ubconst_nl = np.where(ubcoef_nl >= 0, ubcoef_nl, 0) @ deeppoly.ubconst
        updated_ubconst_nl += np.where(ubcoef_nl < 0, ubcoef_nl, 0) @ deeppoly.lbconst
        updated_ubconst_nl += ubconst_nl

        updated_lbcoef_nl = np.where(lbcoef_nl >= 0, lbcoef_nl, 0) @ deeppoly.lbcoef
        updated_lbcoef_nl += np.where(lbcoef_nl < 0, lbcoef_nl, 0) @ deeppoly.ubcoef
        updated_lbconst_nl = np.where(lbcoef_nl >= 0, lbcoef_nl, 0) @ deeppoly.lbconst
        updated_lbconst_nl += np.where(lbcoef_nl < 0, lbcoef_nl, 0) @ deeppoly.ubconst
        updated_lbconst_nl += lbconst_nl

        deeppoly.ubcoef = updated_ubcoef_nl
        deeppoly.ubconst = updated_ubconst_nl
        deeppoly.lbcoef = updated_lbcoef_nl
        deeppoly.lbconst = updated_lbconst_nl

        deeppoly.ubs = np.where(deeppoly.ubcoef >= 0, deeppoly.ubcoef, 0) @ deeppoly.inputbounds[:, 1]
        deeppoly.ubs += np.where(deeppoly.ubcoef < 0, deeppoly.ubcoef, 0) @ deeppoly.inputbounds[:, 0]
        deeppoly.ubs += deeppoly.ubconst
        deeppoly.lbs = np.where(deeppoly.lbcoef >= 0, deeppoly.lbcoef, 0) @ deeppoly.inputbounds[:, 0]
        deeppoly.lbs += np.where(deeppoly.lbcoef < 0, deeppoly.lbcoef, 0) @ deeppoly.inputbounds[:, 1]
        deeppoly.lbs += deeppoly.lbconst

    def execute(self, state):
        '''execute the fully connected layer on a concrete state

        returns output
        '''

        Timers.tic('execute fully connected')

        assert state.shape == self.get_input_shape(), f"state shape to fully connected layer was {state.shape}, " + \
            f"expected {self.get_input_shape()}"

        state = nn_flatten(state)

        rv = np.dot(self.weights, state)

        assert len(self.biases.shape) == 1
        rv = rv + self.biases
        assert len(rv.shape) == 1

        assert rv.shape == self.get_output_shape()

        Timers.toc('execute fully connected')

        return rv

class Convolutional2dLayer(Freezable):
    '''a 2d convolutional layer which takes in multi-channel 2d input data and
    outputs multi-channel 2d data
    '''

    def __init__(self, layer_num, kernels, biases, prev_layer_output_shape, mode='same', boundary='fill', strides=(1, 1), pads=None, is_transpose=False):
        self.layer_num = layer_num
        self.biases = biases
        self.mode = mode
        self.boundary = boundary
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        # Store explicit padding values for ONNX compatibility
        # pads format: [top, left, bottom, right] or None for auto-padding based on mode
        self.pads = pads
        self.is_transpose = is_transpose

        assert isinstance(prev_layer_output_shape, tuple), f"prev_layer_shape was {prev_layer_output_shape}"

        self.prev_layer_output_shape = prev_layer_output_shape

        self.network = None # assigned on network construction

        self.kernels = [] # a list of lists of 2d kernels

        assert len(prev_layer_output_shape) == 3, "previous layer should provide 3 channel output"

        assert len(kernels) >= 1, "need at least one kernel"
        assert isinstance(biases, np.ndarray)
        assert len(kernels.shape) == 4, "expected shape is 4: (# output channels, # input channels, x, y); " + \
                                f"got: {kernels.shape}"

        # ONNX ConvTranspose weight layout is [C_in, C_out, kH, kW]; swap to [C_out, C_in, kH, kW]
        # so the rest of the init (and execute) can treat it identically to regular Conv.
        if is_transpose:
            kernels = kernels.swapaxes(0, 1)

        # for now, all kernels have same width and height so this is a good sanity check for input correctness
        assert kernels[0][0].shape[0] == kernels[0][0].shape[1], \
            f"kernel w and h are not the same: {kernels[0][0].shape}"

        num_output_channels = kernels.shape[0]
        assert biases.shape == (num_output_channels, ), "expected one bias per output channel, shape: " + \
                                                        f"({num_output_channels}, ), got {biases.shape}"

        for k in kernels:
            flipped_channel_kernel = []
            self.kernels.append(flipped_channel_kernel)

            for channel_kernel in k:
                assert len(channel_kernel.shape) == 2, "expected a list of list of 2d kernels"
                if is_transpose:
                    # ConvTranspose uses raw kernel with convolve2d (no pre-flip needed)
                    flipped_channel_kernel.append(channel_kernel.copy())
                else:
                    # flip each kernel since convolution2d works in reverse order
                    flipped_channel_kernel.append(np.flipud(np.fliplr(channel_kernel)))

        # Build (C_out, C_in, kH, kW) array for vectorized execute paths.
        # Always stores the UNFLIPPED (correlation-style) kernels:
        #   - Regular Conv:    self.kernels stores flipud(fliplr(weight)), so we flip back
        #   - ConvTranspose:   self.kernels stores weight as-is (no flip), keep as-is
        # This lets im2col (dot-product = correlation) use kernels_array directly.
        # fftconvolve (mathematical convolution) also uses these unflipped kernels correctly
        # because fftconvolve(x, w) = correlate(x, flip(w)) = convolve(x, w) in the
        # standard mathematical sense, which is what ConvTranspose requires.
        C_out_n = len(self.kernels)
        C_in_n = len(self.kernels[0])
        kH_n = self.kernels[0][0].shape[0]
        kW_n = self.kernels[0][0].shape[1]
        if is_transpose:
            # ConvTranspose: self.kernels already stores unflipped weights
            self.kernels_array = np.array(
                [[self.kernels[co][ci] for ci in range(C_in_n)] for co in range(C_out_n)],
                dtype=np.float32
            ).reshape(C_out_n, C_in_n, kH_n, kW_n)
        else:
            # Regular Conv: self.kernels stores flip(weight), flip back to get original weights
            self.kernels_array = np.array(
                [[np.flipud(np.fliplr(self.kernels[co][ci])) for ci in range(C_in_n)] for co in range(C_out_n)],
                dtype=np.float32
            ).reshape(C_out_n, C_in_n, kH_n, kW_n)

        # Sparse conv matrix (built lazily on first use when CONV_METHOD='sparse')
        self.conv_matrix = None

        self.freeze_attrs()

    def __str__(self):
        return f'[Convolutional2dLayer with {self.get_input_shape()} input and {self.get_output_shape()} output]'

    def get_input_shape(self):
        'get the input shape to this layer'

        return self.prev_layer_output_shape

    def get_output_shape(self):
        'get the output shape from this layer'

        # prev_layer_output_shape: <height, width, depth>

        depth = len(self.kernels)
        h_in = self.prev_layer_output_shape[0]
        w_in = self.prev_layer_output_shape[1]
        kernel_h = self.kernels[0][0].shape[0]
        kernel_w = self.kernels[0][0].shape[1]

        if self.is_transpose:
            # ConvTranspose output formula: (H_in - 1)*s - 2*p + k
            ph = pw = 0
            if self.pads is not None:
                top, left, bottom, right = self.pads
                ph, pw = top + bottom, left + right
            height = (h_in - 1) * self.strides[0] - ph + kernel_h
            width  = (w_in - 1) * self.strides[1] - pw + kernel_w
        elif self.pads is not None:
            # ONNX explicit padding: pads = [top, left, bottom, right]
            top, left, bottom, right = self.pads
            # Add padding to input size
            padded_height = h_in + top + bottom
            padded_width = w_in + left + right
            # Calculate output size with 'valid' convolution on padded input
            height = (padded_height - kernel_h) // self.strides[0] + 1
            width = (padded_width - kernel_w) // self.strides[1] + 1
        elif self.mode == 'valid':
            height = (h_in - kernel_h) // self.strides[0] + 1
            width = (w_in - kernel_w) // self.strides[1] + 1
        else:  # mode == 'same'
            height = (h_in + self.strides[0] - 1) // self.strides[0]
            width = (w_in + self.strides[1] - 1) // self.strides[1]

        return (height, width, depth)

    def _compute_output_region(self, input_region, kernel_size):
        """
        Compute the output region affected by an input region after convolution.

        For 'same' padding, the output region is expanded by approximately kernel_size//2 in each direction.
        For 'valid' padding, the output region is shrunk.
        For explicit ONNX padding, account for the actual pad values.

        Returns: (out_min_y, out_max_y, out_min_x, out_max_x) or None
        """
        if input_region is None:
            return None

        min_y, max_y, min_x, max_x = input_region
        output_shape = self.get_output_shape()
        stride_y, stride_x = self.strides

        if self.pads is not None:
            # ONNX explicit padding: pads = [top, left, bottom, right]
            top, left, bottom, right = self.pads

            # Input region is in original (unpadded) coordinates
            # Convert to padded coordinates
            padded_min_y = min_y + top
            padded_max_y = max_y + top
            padded_min_x = min_x + left
            padded_max_x = max_x + left

            # Now compute output region using 'valid' mode logic on padded input
            # Output pixel (oy, ox) depends on padded input [oy*stride : oy*stride + kernel_size - 1]
            # We want output pixels whose receptive field overlaps [padded_min_y, padded_max_y]

            # First output pixel affected
            out_min_y = max(0, (padded_min_y - kernel_size + stride_y) // stride_y)
            out_max_y = min(output_shape[0] - 1, padded_max_y // stride_y)
            out_min_x = max(0, (padded_min_x - kernel_size + stride_x) // stride_x)
            out_max_x = min(output_shape[1] - 1, padded_max_x // stride_x)

        elif self.mode == 'same':
            # For 'same' mode, each input pixel affects a region of size ~kernel_size in output
            padding = kernel_size // 2

            # Convert input coordinates to output coordinates (accounting for stride)
            out_min_y = max(0, (min_y - padding) // stride_y)
            out_max_y = min(output_shape[0] - 1, (max_y + padding) // stride_y)
            out_min_x = max(0, (min_x - padding) // stride_x)
            out_max_x = min(output_shape[1] - 1, (max_x + padding) // stride_x)
        else:  # 'valid' mode
            # For valid mode, output is smaller
            # Output pixel (i, j) depends on input region [i*stride : i*stride + kernel_size, j*stride : j*stride + kernel_size]

            # Find output pixels affected by input region [min_y:max_y+1, min_x:max_x+1]
            # Output pixel (oy, ox) is affected if its receptive field overlaps [min_y, max_y]
            #   Receptive field: [oy * stride_y, oy * stride_y + kernel_size - 1]

            # First output pixel affected: ceil((min_y - kernel_size + 1) / stride_y)
            out_min_y = max(0, (min_y - kernel_size + stride_y) // stride_y)
            out_max_y = min(output_shape[0] - 1, max_y // stride_y)
            out_min_x = max(0, (min_x - kernel_size + stride_x) // stride_x)
            out_max_x = min(output_shape[1] - 1, max_x // stride_x)

        return (out_min_y, out_max_y, out_min_x, out_max_x)

    def _apply_conv_to_mat(self, mat, shape):
        """Apply the conv/ConvTranspose transformation to all generator columns at once.

        mat: (in_size, G) float array — G generator columns stacked horizontally.
        shape: (H, W, C_in) input shape — must equal self.prev_layer_output_shape.
        Returns: (out_size, G) float array.

        For regular Conv: batched im2col + single BLAS matmul for all G generators.
        For ConvTranspose: batched upsample + im2col on upsampled grid + matmul.
          ConvTranspose(x) = FullConv(upsample(x), W) which, after zero-insertion, is
          equivalent to a regular (no-stride) convolution on the upsampled grid.
          We upsample all G generators at once, then apply batched im2col + matmul.
        """
        K = self.kernels_array  # (C_out, C_in, kH, kW)
        C_out, C_in, kH, kW = K.shape
        sh, sw = self.strides
        H, W, _ = shape
        G = mat.shape[1]

        # Reshape all G generator columns to (G, H, W, C_in)
        X = mat.T.reshape(G, H, W, C_in).astype(np.float32, copy=False)

        if self.is_transpose:
            # Upsample all G generators at once: (G, H2, W2, C_in)
            H2 = (H - 1) * sh + 1
            W2 = (W - 1) * sw + 1
            X_up = np.zeros((G, H2, W2, C_in), dtype=np.float32)
            X_up[:, ::sh, ::sw, :] = X

            # Apply im2col in 'full' mode (no stride on output, pad with kH-1, kW-1)
            ph, pw = kH - 1, kW - 1
            X_p = np.pad(X_up, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
            Gp, Hp, Wp, _ = X_p.shape
            out_H = Hp - kH + 1
            out_W = Wp - kW + 1

            if self.pads is not None:
                top, left, bottom, right = self.pads
                # Apply explicit padding crop (output_padding for ConvTranspose)
                out_H -= (top + bottom)
                out_W -= (left + right)
        else:
            # Regular Conv: apply explicit or mode-based padding
            if self.pads is not None:
                top, left, bottom, right = self.pads
                X_p = np.pad(X, ((0, 0), (top, bottom), (left, right), (0, 0)), mode='constant')
            elif self.mode == 'same':
                pad_h = max(kH - 1, 0)
                pad_w = max(kW - 1, 0)
                X_p = np.pad(X, ((0, 0), (pad_h//2, pad_h - pad_h//2),
                                 (pad_w//2, pad_w - pad_w//2), (0, 0)), mode='constant')
            else:
                X_p = X  # valid, no padding

            Gp, Hp, Wp, _ = X_p.shape
            out_H = (Hp - kH) // sh + 1
            out_W = (Wp - kW) // sw + 1

        # im2col computes correlation (dot product).  For regular Conv, kernels_array stores
        # the unflipped (original) weights and correlation = what we want.
        # For ConvTranspose, execute() uses fftconvolve (mathematical convolution), which
        # requires flipping the kernel.  Match that by flipping kernels_array here too.
        if self.is_transpose:
            K_use = K[:, :, ::-1, ::-1]  # flip spatial dims: correlation of flipped = convolution
        else:
            K_use = K
        K_r = K_use.transpose(0, 2, 3, 1).reshape(C_out, kH * kW * C_in)

        # Chunk generators to keep im2col buffer under ~128MB.
        # Process all generators at once (no chunking); memory is ~G * out_H * out_W * kH * kW * C_in * 4 bytes
        chunk = G

        out_stride_h = 1 if self.is_transpose else sh
        out_stride_w = 1 if self.is_transpose else sw

        result_parts = []
        for g_start in range(0, G, chunk):
            g_end = min(g_start + chunk, G)
            Gc = g_end - g_start
            X_chunk = X_p[g_start:g_end]  # (Gc, Hp, Wp, C_in)

            col_chunk = np.lib.stride_tricks.as_strided(
                X_chunk,
                shape=(Gc, out_H, out_W, kH, kW, C_in),
                strides=(X_chunk.strides[0],
                         X_chunk.strides[1] * out_stride_h,
                         X_chunk.strides[2] * out_stride_w,
                         X_chunk.strides[1], X_chunk.strides[2], X_chunk.strides[3])
            ).reshape(Gc * out_H * out_W, kH * kW * C_in)

            out_chunk = (col_chunk @ K_r.T).reshape(Gc, out_H, out_W, C_out)
            result_parts.append(out_chunk.reshape(Gc, -1))

        result = np.concatenate(result_parts, axis=0) if len(result_parts) > 1 else result_parts[0]

        return result.T  # (out_size, G)

    def _build_conv_matrix(self):
        """Build and return the sparse Toeplitz matrix W of shape (out_size, in_size).

        W encodes the convolution as a linear map: out_vec = W @ in_vec, where
        in_vec and out_vec are HWC-flat vectors.  Each row of W corresponds to one
        output element (h_out, w_out, c_out) and has nonzeros at the kH*kW*C_in
        input positions in its receptive field, with values equal to the kernel weights.

        The matrix is built once and cached in self.conv_matrix for reuse across
        all generator columns and all verification calls on this network.

        Only supported for regular Conv (not ConvTranspose).
        """
        from scipy.sparse import csr_matrix

        K = self.kernels_array          # (C_out, C_in, kH, kW) unflipped weights
        C_out, C_in, kH, kW = K.shape
        sh, sw = self.strides
        H_in, W_in, _ = self.prev_layer_output_shape

        # Compute padded input dimensions and output dimensions (same logic as _apply_conv_to_mat)
        if self.pads is not None:
            top, left, bottom, right = self.pads
            H_p = H_in + top + bottom
            W_p = W_in + left + right
            pad_top, pad_left = top, left
        elif self.mode == 'same':
            pad_h = max(kH - 1, 0)
            pad_w = max(kW - 1, 0)
            pad_top  = pad_h // 2
            pad_left = pad_w // 2
            H_p = H_in + pad_h
            W_p = W_in + pad_w
        else:  # 'valid'
            pad_top, pad_left = 0, 0
            H_p, W_p = H_in, W_in

        out_H = (H_p - kH) // sh + 1
        out_W = (W_p - kW) // sw + 1

        in_size  = H_in * W_in * C_in
        out_size = out_H * out_W * C_out

        # Build COO arrays
        nnz = out_H * out_W * C_out * kH * kW * C_in
        row_idx = np.empty(nnz, dtype=np.int32)
        col_idx = np.empty(nnz, dtype=np.int32)
        data    = np.empty(nnz, dtype=np.float32)

        idx = 0
        for h_out in range(out_H):
            for w_out in range(out_W):
                h_base = h_out * sh - pad_top   # top-left corner in padded input (unpadded coords)
                w_base = w_out * sw - pad_left
                for c_out in range(C_out):
                    row = (h_out * out_W + w_out) * C_out + c_out  # HWC-flat output index
                    for kh in range(kH):
                        h_in = h_base + kh
                        if h_in < 0 or h_in >= H_in:
                            continue
                        for kw in range(kW):
                            w_in = w_base + kw
                            if w_in < 0 or w_in >= W_in:
                                continue
                            for c_in in range(C_in):
                                col = (h_in * W_in + w_in) * C_in + c_in  # HWC-flat input index
                                row_idx[idx] = row
                                col_idx[idx] = col
                                data[idx]    = K[c_out, c_in, kh, kw]
                                idx += 1

        # Trim (padding zeros skipped above reduce actual nnz)
        row_idx = row_idx[:idx]
        col_idx = col_idx[:idx]
        data    = data[:idx]

        return csr_matrix((data, (row_idx, col_idx)), shape=(out_size, in_size), dtype=np.float32)

    def _apply_conv_to_mat_sparse(self, mat, shape):
        """Sparse matrix implementation of conv transformation.

        Converts the (in_size, G) generator matrix to CSC sparse format, multiplies
        by the prebuilt Toeplitz W_sparse (CSR).  Returns a CSC sparse matrix so the
        caller can propagate sparsity through subsequent layers.

        mat: (in_size, G) dense or CSC sparse float array
        shape: (H, W, C_in) — must equal self.prev_layer_output_shape
        Returns: (out_size, G) CSC sparse float32 matrix
        """
        from scipy.sparse import csc_matrix, issparse

        if self.conv_matrix is None:
            self.conv_matrix = self._build_conv_matrix()

        if not issparse(mat):
            gen_sparse = csc_matrix(mat.astype(np.float32, copy=False))
        else:
            gen_sparse = mat.tocsc()

        result = self.conv_matrix @ gen_sparse   # CSR @ CSC → sparse result
        result = result.tocsc()
        result.eliminate_zeros()
        return result  # (out_size, G) CSC sparse

    def _would_exceed_memory(self, star):
        '''True if propagating star.a_mat through this conv layer would exceed MEMORY_BUDGET_GB.

        For sparse stars, estimates output nnz = G × rf_h × rf_w × C_out where rf is the
        current receptive field size of each generator. rf is estimated from the avg nonzeros
        per generator column in the input a_mat.

        Falls back to dense estimate (out_neurons × G × 4) if generators are already dense.
        '''
        from nnenum.settings import Settings
        from scipy.sparse import issparse
        if not issparse(star.a_mat):
            return False

        G = star.a_mat.shape[1]
        if G == 0:
            return False

        out_h, out_w, c_out = self.get_output_shape()
        out_neurons = out_h * out_w * c_out

        # Estimate avg nnz per generator in the output.
        # Each input nonzero fans out to at most kH × kW output rows per output channel,
        # but since the output is spatially local, the output nnz per gen ≈ rf_h × rf_w × C_out
        # where rf = sqrt(avg input nnz per gen / C_in).
        input_nnz = star.a_mat.nnz
        avg_nnz_per_gen = input_nnz / G  # avg nonzeros per generator column

        kh, kw = self.kernels[0][0].shape[:2]
        # Output nonzeros per generator: each input nonzero contributes to at most kH×kW output
        # positions, times C_out output channels.
        estimated_output_nnz_per_gen = avg_nnz_per_gen * kh * kw
        # Cap at dense output size per generator
        estimated_output_nnz_per_gen = min(estimated_output_nnz_per_gen, out_neurons)

        estimated_total_nnz = estimated_output_nnz_per_gen * G
        estimated_bytes = estimated_total_nnz * 4  # float32

        return estimated_bytes > Settings.MEMORY_BUDGET_GB * 1e9

    def _batch_generators_for_conv(self, mat, shape):
        """
        Group generator columns that won't interact during convolution into batches.

        For sparse generators, we can combine multiple generators whose nonzero regions
        are separated by at least the kernel size, perform convolution once on the batch,
        then separate them back out.

        Returns: list of batches, where each batch is a dict:
            {
                'indices': list of column indices in this batch,
                'input_regions': list of (min_y, max_y, min_x, max_x) tuples for input nonzero regions,
                'output_regions': list of (min_y, max_y, min_x, max_x) tuples for output regions
            }
        """
        Timers.tic('batch_generators_for_conv')

        from nnenum.settings import Settings

        height, width, channels = shape
        kernel_size = self.kernels[0][0].shape[0]  # Assume square kernels

        # For each generator, find its nonzero bounding box
        generator_info = []
        for cindex in range(mat.shape[1]):
            column = mat[:, cindex]

            # Find nonzero region in 2D space (considering all channels)
            multichannel = nn_unflatten(column, shape)

            # Find bounding box across all channels
            nonzero_indices = np.nonzero(multichannel)
            if len(nonzero_indices[0]) > 0:
                min_y, max_y = nonzero_indices[0].min(), nonzero_indices[0].max()
                min_x, max_x = nonzero_indices[1].min(), nonzero_indices[1].max()
                input_region = (min_y, max_y, min_x, max_x)
            else:
                input_region = None

            # Compute output region
            output_region = self._compute_output_region(input_region, kernel_size)

            generator_info.append({
                'index': cindex,
                'input_region': input_region,
                'output_region': output_region,
                'column': column
            })

        def output_regions_overlap(region1, region2):
            if region1 is None or region2 is None:
                return False
            min_y1, max_y1, min_x1, max_x1 = region1
            min_y2, max_y2, min_x2, max_x2 = region2
            return not (max_y1 < min_y2 or max_y2 < min_y1 or max_x1 < min_x2 or max_x2 < min_x1)

        def make_batch(gen_info):
            return {
                'indices': [gen_info['index']],
                'input_regions': [gen_info['input_region']],
                'output_regions': [gen_info['output_region']]
            }

        def add_to_batch(batch, gen_info):
            batch['indices'].append(gen_info['index'])
            batch['input_regions'].append(gen_info['input_region'])
            batch['output_regions'].append(gen_info['output_region'])

        strategy = Settings.CONV_BATCHING_STRATEGY

        if strategy == 'greedy':
            # O(n*B): scan all existing batches in order, place into first compatible one
            batches = []
            for gen_info in generator_info:
                placed = False
                for batch in batches:
                    if not any(output_regions_overlap(gen_info['output_region'], r)
                               for r in batch['output_regions']):
                        add_to_batch(batch, gen_info)
                        placed = True
                        break
                if not placed:
                    batches.append(make_batch(gen_info))

        elif strategy == 'random':
            # O(n*k): try up to MAX_FAILURES random batches before creating a new one.
            # Randomizing which batch we try avoids the greedy bias of always filling the
            # first batch, leading to better average fill rates.
            import random
            max_failures = Settings.CONV_BATCHING_MAX_FAILURES
            batches = []
            for gen_info in generator_info:
                placed = False
                if batches:
                    # Sample a random permutation of existing batch indices
                    order = list(range(len(batches)))
                    random.shuffle(order)
                    failures = 0
                    for bi in order:
                        if failures >= max_failures:
                            break
                        batch = batches[bi]
                        if not any(output_regions_overlap(gen_info['output_region'], r)
                                   for r in batch['output_regions']):
                            add_to_batch(batch, gen_info)
                            placed = True
                            break
                        failures += 1
                if not placed:
                    batches.append(make_batch(gen_info))

        else:
            assert strategy == 'period', f"Unknown batching strategy: {strategy!r}"
            # O(n): assign generators to batches by (y % period, x % period).
            # Generators in the same period-cell have non-overlapping output regions
            # by construction (period = output region size in each spatial dimension).
            # This requires generators to be 1-hot in spatial position (the typical
            # initial-star case). Falls back to greedy for non-1-hot generators.
            #
            # Output region size per generator = kernel_size (for same/pad convolutions).
            # To guarantee non-overlap we need period >= output_region_width.
            # For a kxk kernel with same padding, each 1-hot input at (y,x) produces
            # output nonzeros in a region of width k (centered on (y,x)).
            # Period = k ensures generators (y1,x1) and (y2,x2) with the same
            # (y%k, x%k) are separated by at least k in y or x, so their output
            # regions don't overlap.
            kernel_size = self.kernels[0][0].shape[0]
            period = kernel_size  # minimum safe period for same-padded conv
            batch_map = {}  # (y%k, x%k, slot) -> batch index
            pixel_slot_counts = {}  # (y, x) -> count of gens assigned so far
            batches = []
            for gen_info in generator_info:
                region = gen_info['input_region']
                if region is None:
                    batches.append(make_batch(gen_info))
                    continue
                min_y, max_y, min_x, max_x = region
                # Only use period assignment for single-pixel (1-hot) generators.
                # Multi-pixel generators could span multiple period cells.
                if max_y == min_y and max_x == min_x:
                    # Generators with the same (y%period, x%period) are separated by >=period
                    # pixels in y or x, so their output regions don't overlap.
                    # However, multiple generators at the exact same (y, x) pixel (different
                    # channels) share an identical output region and cannot be batched together.
                    # Use a per-pixel counter to assign them to successive period-cell slots.
                    pixel_key = (min_y, min_x)
                    slot = pixel_slot_counts.get(pixel_key, 0)
                    pixel_slot_counts[pixel_key] = slot + 1
                    # The batch key is (y%period, x%period, slot) so same-pixel generators
                    # land in different slots while same-slot different-pixel generators batch.
                    key = (min_y % period, min_x % period, slot)
                    if key in batch_map:
                        add_to_batch(batches[batch_map[key]], gen_info)
                    else:
                        batch_map[key] = len(batches)
                        batches.append(make_batch(gen_info))
                else:
                    # Non-1-hot: fall back to greedy placement
                    placed = False
                    for batch in batches:
                        if not any(output_regions_overlap(gen_info['output_region'], r)
                                   for r in batch['output_regions']):
                            add_to_batch(batch, gen_info)
                            placed = True
                            break
                    if not placed:
                        batches.append(make_batch(gen_info))

        # Optional logging
        if Settings.LOG_CONV_BATCHING:
            num_gens = mat.shape[1]
            num_batches = len(batches)
            batch_sizes = [len(b['indices']) for b in batches]
            compression_ratio = num_gens / num_batches if num_batches > 0 else 0
            print(f"[Conv Batching L{self.layer_num}] {num_gens} gens → {num_batches} batches " +
                  f"(ratio: {compression_ratio:.2f}x, mean batch size: {np.mean(batch_sizes):.1f})")

        Timers.toc('batch_generators_for_conv')
        return batches, generator_info

    def transform_star(self, star):
        'apply the linear transformation part of the layer to the passed-in lp_star (not relu)'

        shape = self.get_input_shape()

        from nnenum.settings import Settings

        # Determine effective method for this layer
        method = Settings.CONV_METHOD
        if self.is_transpose:
            method = 'dense'  # sparse/_build_conv_matrix not implemented for ConvTranspose
        elif method == 'batching' and not Settings.CONV_BATCHING_ENABLED:
            method = 'dense'
        elif method == 'batching' and Settings.CONV_BATCHING_FIRST_LAYER_ONLY and self.layer_num > 0:
            method = 'dense'

        # Sparsity fallback: both 'batching' and 'sparse' degrade to 'dense' when generators are too dense
        if method in ('batching', 'sparse') and star.a_mat.shape[1] > 0:
            from scipy.sparse import issparse as _issparse
            if _issparse(star.a_mat):
                first_col = star.a_mat.getcol(0)
                first_gen_sparsity = first_col.nnz / star.a_mat.shape[0]
            else:
                first_gen_sparsity = np.count_nonzero(star.a_mat[:, 0]) / star.a_mat.shape[0]
            if first_gen_sparsity > Settings.CONV_BATCHING_MIN_SPARSITY:
                method = 'dense'

        # Generator count fallback: sparse build cost only amortizes for large G
        if method == 'sparse':
            G = star.a_mat.shape[1]
            if G < Settings.CONV_SPARSE_MIN_GENERATORS:
                method = 'dense'

        if method == 'sparse':
            Timers.tic('transform_star_sparse_conv')
            if star.a_mat.shape[1] > 0:
                result = self._apply_conv_to_mat_sparse(star.a_mat, shape)
                if Settings.SPARSE_STAR:
                    # Keep as sparse CSR — densification deferred to FC/MatMul layers
                    star.a_mat = result
                else:
                    star.a_mat = result.toarray().astype(star.a_mat.dtype) if hasattr(result, 'toarray') else result
            multichannel_state = nn_unflatten(star.bias, shape)
            multichannel_state = self.execute(multichannel_state)
            star.bias = nn_flatten(multichannel_state)
            Timers.toc('transform_star_sparse_conv')
            assert star.bias.size == star.a_mat.shape[0]
            return

        if method == 'dense':
            # Vectorized path: apply conv to all generator columns simultaneously
            Timers.tic('transform_star_unbatched')
            if star.a_mat.shape[1] > 0:
                star.densify()  # _apply_conv_to_mat requires dense input (reshape/im2col)
                star.a_mat = self._apply_conv_to_mat(star.a_mat, shape)

            # bias transformation
            multichannel_state = nn_unflatten(star.bias, shape)
            multichannel_state = self.execute(multichannel_state)
            flat = nn_flatten(multichannel_state)
            star.bias = flat
            Timers.toc('transform_star_unbatched')

            assert star.bias.size == star.a_mat.shape[0]
            return

        # method == 'batching': batch generators to reduce convolution operations
        batches, generator_info = self._batch_generators_for_conv(star.a_mat, shape)

        Timers.tic('transform_star_batched_conv')

        output_shape = self.get_output_shape()
        n_out = int(np.prod(output_shape))
        G = star.a_mat.shape[1]
        result = np.zeros((n_out, G), dtype=star.a_mat.dtype)

        for batch in batches:
            if len(batch['indices']) == 1:
                # Single generator - process normally
                idx = batch['indices'][0]
                column = generator_info[idx]['column']
                multichannel_state = nn_unflatten(column, shape)
                multichannel_state = self.execute(multichannel_state, zero_bias=True)
                result[:, idx] = nn_flatten(multichannel_state)
            else:
                # Multiple non-conflicting generators - combine, convolve once, separate
                # Key insight: convolution is linear, so conv(sum(g_i)) = sum(conv(g_i))
                # Since output regions don't overlap, we can extract each contribution by masking

                # Combine generators in this batch
                combined = np.zeros(shape, dtype=star.a_mat.dtype)

                for idx in batch['indices']:
                    column = generator_info[idx]['column']
                    multichannel_state = nn_unflatten(column, shape)
                    combined += multichannel_state

                # Perform ONE convolution on combined batch
                combined_result = self.execute(combined, zero_bias=True)

                # Extract each generator's contribution using output regions
                H_out, W_out, C_out = output_shape
                for i, idx in enumerate(batch['indices']):
                    output_region = batch['output_regions'][i]

                    if output_region is not None:
                        out_min_y, out_max_y, out_min_x, out_max_x = output_region
                        # Write patch directly into pre-allocated result column
                        patch = combined_result[out_min_y:out_max_y+1, out_min_x:out_max_x+1, :]
                        ph = out_max_y - out_min_y + 1
                        pw = out_max_x - out_min_x + 1
                        # Flat indices of the patch in a (H_out, W_out, C_out) layout
                        row_starts = (np.arange(ph) + out_min_y) * (W_out * C_out) + out_min_x * C_out
                        flat_idx = (row_starts[:, None, None]
                                    + np.arange(pw)[None, :, None] * C_out
                                    + np.arange(C_out)[None, None, :]).ravel()
                        result[flat_idx, idx] = patch.ravel()
                    # else: output_region is None → zero generator, result[:, idx] already zero

        star.a_mat = result

        # bias (anchor) transformation includes layer bias
        multichannel_state = nn_unflatten(star.bias, shape)
        multichannel_state = self.execute(multichannel_state)
        flat = nn_flatten(multichannel_state)
        star.bias = flat

        Timers.toc('transform_star_batched_conv')

        assert star.bias.size == star.a_mat.shape[0]

    def transform_zono(self, zono):
        'apply the linear transformation part of the layer to the passed-in zonotope (not relu)'

        # mat_t has one generator PER COLUMN
        shape = self.get_input_shape()

        from nnenum.settings import Settings

        # Determine effective method for this layer
        method = Settings.CONV_METHOD
        if self.is_transpose:
            method = 'dense'  # sparse/_build_conv_matrix not implemented for ConvTranspose
        elif method == 'batching' and not Settings.CONV_BATCHING_ENABLED:
            method = 'dense'
        elif method == 'batching' and Settings.CONV_BATCHING_FIRST_LAYER_ONLY and self.layer_num > 0:
            method = 'dense'

        from scipy.sparse import issparse

        # If mat_t is already sparse (from a previous sparse conv layer), force sparse method
        # unless is_transpose (sparse not implemented for ConvTranspose)
        if issparse(zono.mat_t) and not self.is_transpose:
            method = 'sparse'

        # Sparsity fallback: both 'batching' and 'sparse' degrade to 'dense' when generators are too dense
        if method in ('batching', 'sparse') and zono.mat_t.shape[1] > 0:
            if issparse(zono.mat_t):
                nnz = zono.mat_t.nnz
                total = zono.mat_t.shape[0] * zono.mat_t.shape[1]
                first_gen_sparsity = nnz / max(total, 1)
            else:
                first_gen_sparsity = np.count_nonzero(zono.mat_t[:, 0]) / zono.mat_t.shape[0]
            if first_gen_sparsity > Settings.CONV_BATCHING_MIN_SPARSITY:
                method = 'dense'
                if issparse(zono.mat_t):
                    dense_bytes = zono.mat_t.shape[0] * zono.mat_t.shape[1] * 4
                    if dense_bytes <= Settings.MEMORY_BUDGET_GB * 1e9:
                        zono.mat_t = zono.mat_t.toarray().astype(np.float32)
                    # else: leave sparse — dense would OOM, sparse path continues

        # Generator count fallback: sparse conv matrix build has a large one-time cost that
        # only amortizes when G is large enough. With few generators, dense im2col is faster.
        if method == 'sparse' and not issparse(zono.mat_t):
            G = zono.mat_t.shape[1]
            if G < Settings.CONV_SPARSE_MIN_GENERATORS:
                method = 'dense'

        if method == 'sparse':
            Timers.tic('transform_zono_sparse_conv')
            if zono.mat_t.shape[1] > 0:
                zono.mat_t = self._apply_conv_to_mat_sparse(zono.mat_t, shape)
                # Density gate: convert to dense if sparsity benefit is gone,
                # but only if the dense form fits within the memory budget.
                nnz = zono.mat_t.nnz
                total = zono.mat_t.shape[0] * zono.mat_t.shape[1]
                dense_bytes = total * 4  # float32
                if (total > 0 and nnz / total > Settings.CONV_BATCHING_MIN_SPARSITY
                        and dense_bytes <= Settings.MEMORY_BUDGET_GB * 1e9):
                    zono.mat_t = zono.mat_t.toarray().astype(np.float32)
                if Settings.SPARSE_DEBUG:
                    _nnz = zono.mat_t.nnz if issparse(zono.mat_t) else int(np.count_nonzero(zono.mat_t))
                    _tot = zono.mat_t.shape[0] * zono.mat_t.shape[1]
                    print(f"[transform_zono sparse] layer={self.layer_num} "
                          f"shape={zono.mat_t.shape} nnz={_nnz} density={_nnz/max(_tot,1):.4%} "
                          f"sparse={issparse(zono.mat_t)}", flush=True)
            multichannel_state = nn_unflatten(zono.center, shape)
            multichannel_state = self.execute(multichannel_state)
            zono.center = nn_flatten(multichannel_state)
            Timers.toc('transform_zono_sparse_conv')
            assert zono.center.size == zono.mat_t.shape[0]
            return

        if method == 'dense':
            # Ensure dense (in case mat_t arrived sparse but below threshold wasn't caught above)
            if issparse(zono.mat_t):
                dense_bytes = zono.mat_t.shape[0] * zono.mat_t.shape[1] * 4
                if dense_bytes <= Settings.MEMORY_BUDGET_GB * 1e9:
                    zono.mat_t = zono.mat_t.toarray().astype(np.float32)
                # else: leave sparse — dense would OOM

            # For large G with dense generators, use prebuilt sparse W matrix (W @ mat_t) instead
            # of im2col — avoids allocating a large intermediate buffer (e.g., 1.2 GB for 8×8 layers).
            G = zono.mat_t.shape[1] if not issparse(zono.mat_t) else 0
            if (not issparse(zono.mat_t) and not self.is_transpose
                    and G >= Settings.CONV_MATRIX_DENSE_GEN_THRESHOLD):
                Timers.tic('transform_zono_matrix_dense')
                if G > 0:
                    if self.conv_matrix is None:
                        self.conv_matrix = self._build_conv_matrix()
                    result = self.conv_matrix @ zono.mat_t.astype(np.float32, copy=False)
                    # scipy CSR @ dense ndarray returns a dense ndarray; sparse result needs .toarray()
                    zono.mat_t = result.toarray() if issparse(result) else np.asarray(result)
                multichannel_state = nn_unflatten(zono.center, shape)
                multichannel_state = self.execute(multichannel_state)
                zono.center = nn_flatten(multichannel_state)
                Timers.toc('transform_zono_matrix_dense')
                assert zono.center.size == zono.mat_t.shape[0]
                return

            # Vectorized path: apply conv to all generator columns simultaneously
            Timers.tic('transform_zono_unbatched')
            if zono.mat_t.shape[1] > 0:
                zono.mat_t = self._apply_conv_to_mat(zono.mat_t, shape)

            # center transformation
            multichannel_state = nn_unflatten(zono.center, shape)
            multichannel_state = self.execute(multichannel_state)
            flat = nn_flatten(multichannel_state)
            zono.center = flat
            Timers.toc('transform_zono_unbatched')

            assert zono.center.size == zono.mat_t.shape[0]
            return

        # method == 'batching': batch generators to reduce convolution operations
        batches, generator_info = self._batch_generators_for_conv(zono.mat_t, shape)

        Timers.tic('transform_zono_batched_conv')

        # Process each batch
        result_columns = [None] * zono.mat_t.shape[1]

        for batch in batches:
            if len(batch['indices']) == 1:
                # Single generator - process normally
                idx = batch['indices'][0]
                column = generator_info[idx]['column']
                multichannel_state = nn_unflatten(column, shape)
                multichannel_state = self.execute(multichannel_state, zero_bias=True)
                flat = nn_flatten(multichannel_state)
                flat.shape = (flat.size, 1)
                result_columns[idx] = flat
            else:
                # Multiple non-conflicting generators - combine, convolve once, separate
                # Combine generators in this batch
                combined = np.zeros(shape, dtype=zono.mat_t.dtype)

                for idx in batch['indices']:
                    column = generator_info[idx]['column']
                    multichannel_state = nn_unflatten(column, shape)
                    combined += multichannel_state

                # Perform ONE convolution on combined batch
                combined_result = self.execute(combined, zero_bias=True)
                combined_result_2d = combined_result

                # Extract each generator's contribution using output regions
                output_shape = self.get_output_shape()

                for i, idx in enumerate(batch['indices']):
                    output_region = batch['output_regions'][i]

                    if output_region is None:
                        # Zero generator
                        flat = np.zeros((np.prod(output_shape), 1), dtype=zono.mat_t.dtype)
                        result_columns[idx] = flat
                    else:
                        # Extract output region for this generator
                        out_min_y, out_max_y, out_min_x, out_max_x = output_region

                        # Create masked output
                        masked_output = np.zeros(output_shape, dtype=zono.mat_t.dtype)
                        masked_output[out_min_y:out_max_y+1, out_min_x:out_max_x+1, :] = \
                            combined_result_2d[out_min_y:out_max_y+1, out_min_x:out_max_x+1, :]

                        flat = nn_flatten(masked_output)
                        flat.shape = (flat.size, 1)
                        result_columns[idx] = flat

        zono.mat_t = np.hstack(result_columns)

        # center transformation includes layer bias
        multichannel_state = nn_unflatten(zono.center, shape)
        multichannel_state = self.execute(multichannel_state)
        flat = nn_flatten(multichannel_state)
        zono.center = flat

        Timers.toc('transform_zono_batched_conv')

        assert zono.center.size == zono.mat_t.shape[0]

    def execute(self, state, zero_bias=False):
        '''execute the convolutional layer on a concrete state

        if save_branching is True, returns (output, branch_list), where branch_list is a list of booleans for each
            relu neuron that is True if input is nonnegative and False otherwise

        if zero_bias is True, use a zero bias instead of what's in the layer (used in ImageStar computations)

        otherwise, just returns output
        '''

        Timers.tic('execute Convolutional2dLayer')

        assert state.shape == self.prev_layer_output_shape, f"expected shape {self.prev_layer_output_shape}, " + \
                                                            f"got {state.shape}"

        K = self.kernels_array  # (C_out, C_in, kH, kW)
        C_out, C_in, kH, kW = K.shape
        sh, sw = self.strides
        state_f = state.astype(np.float32, copy=False)

        if self.is_transpose:
            # ConvTranspose: upsample input then apply full-mode convolution.
            # Vectorized: batch all (C_out * C_in) spatial convolutions via fftconvolve.
            H, W, _ = state_f.shape
            H2 = (H - 1) * sh + 1
            W2 = (W - 1) * sw + 1
            # Upsample all input channels at once: (C_in, H2, W2)
            x_up = np.zeros((C_in, H2, W2), dtype=np.float32)
            x_up[:, ::sh, ::sw] = state_f.transpose(2, 0, 1)
            out_H = H2 + kH - 1
            out_W = W2 + kW - 1
            # Flatten to (C_out*C_in, H2, W2) and (C_out*C_in, kH, kW) for one fftconvolve call
            x_flat = np.broadcast_to(x_up[np.newaxis], (C_out, C_in, H2, W2)).reshape(C_out * C_in, H2, W2).copy()
            k_flat = K.reshape(C_out * C_in, kH, kW)
            convolved = fftconvolve(x_flat, k_flat, mode='full', axes=(1, 2))  # (C_out*C_in, out_H, out_W)
            # Sum over C_in dimension → (C_out, out_H, out_W), then to HWC
            output = convolved.reshape(C_out, C_in, out_H, out_W).sum(axis=1)  # (C_out, out_H, out_W)
        else:
            # Standard Conv: build padded input then im2col + matmul.
            if self.pads is not None:
                top, left, bottom, right = self.pads
                state_p = np.pad(state_f, ((top, bottom), (left, right), (0, 0)), mode='constant')
            elif self.mode == 'same':
                # Compute symmetric padding to match scipy 'same' behaviour
                pad_h = max(kH - 1, 0)
                pad_w = max(kW - 1, 0)
                state_p = np.pad(state_f, ((pad_h // 2, pad_h - pad_h // 2),
                                           (pad_w // 2, pad_w - pad_w // 2), (0, 0)), mode='constant')
            else:
                state_p = state_f  # mode='valid', no padding

            Hp, Wp, _ = state_p.shape
            out_H = (Hp - kH) // sh + 1
            out_W = (Wp - kW) // sw + 1

            # im2col: extract (out_H, out_W, kH, kW, C_in) patches then reshape to (out_H*out_W, kH*kW*C_in)
            col = np.lib.stride_tricks.as_strided(
                state_p,
                shape=(out_H, out_W, kH, kW, C_in),
                strides=(state_p.strides[0] * sh, state_p.strides[1] * sw,
                         state_p.strides[0], state_p.strides[1], state_p.strides[2])
            ).reshape(out_H * out_W, kH * kW * C_in)
            # K: (C_out, C_in, kH, kW) → (C_out, kH, kW, C_in) → (C_out, kH*kW*C_in)
            K_r = K.transpose(0, 2, 3, 1).reshape(C_out, kH * kW * C_in)
            # matmul → (out_H*out_W, C_out) → (C_out, out_H, out_W)
            output = (col @ K_r.T).reshape(out_H, out_W, C_out).transpose(2, 0, 1)

        # Add bias (or zero for zero_bias=True) and convert to HWC
        bias_vec = np.zeros(C_out, dtype=np.float32) if zero_bias else self.biases.astype(np.float32)
        output = output.transpose(1, 2, 0) + bias_vec  # (out_H, out_W, C_out)

        Timers.toc('execute Convolutional2dLayer')

        return output

class BranchRestoreLayer(Freezable):
    '''Restore the current star/state from a cached checkpoint.

    Used when the ONNX graph branches from a single input to multiple parallel
    sub-networks (e.g. cersyve).  After one branch is processed, this layer
    resets the flowing state back to the cached state at `source_cache_key` so
    the next branch can start from the same point.

    In execute(): network.execute() handles this specially, replacing `state`
    with `activation_cache[dag_predecessors[layer_idx][0]]`.

    In lp_star_state: apply_linear_layer() handles this specially, replacing
    self.star (and prefilter.zono) with copies from star_cache / zono_cache.
    '''

    def __init__(self, layer_num, shape):
        self.layer_num = layer_num
        self.input_shape = shape
        self.network = None
        self.freeze_attrs()

    def __str__(self):
        return f'[BranchRestoreLayer -> {self.input_shape}]'

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.input_shape

    def execute(self, state):
        # Handled specially in network.execute(); should not be called directly.
        raise RuntimeError("BranchRestoreLayer.execute() must be handled by network.execute()")

    def transform_star(self, star):
        raise RuntimeError("BranchRestoreLayer.transform_star() must be handled by lp_star_state")

    def transform_zono(self, zono):
        raise RuntimeError("BranchRestoreLayer.transform_zono() must be handled by lp_star_state")

    def transform_deeppoly(self, deeppoly):
        raise RuntimeError("BranchRestoreLayer.transform_deeppoly() must be handled by lp_star_state")


class SkipAddLayer(Freezable):
    '''Skip-connection Add layer with two inputs (element-wise addition).

    In a residual (ResNet) network an Add node merges the "main path" output
    and the "skip path" output from an earlier layer.  Unlike every other
    layer, this one consumes *two* star sets / zonotopes instead of one.

    The API difference from standard layers:
        execute(state1, state2)           -> state1 + state2
        transform_star(star1, star2)      -> combined LpStar
        transform_zono(zono1, zono2)      -> combined Zonotope
        transform_deeppoly(dp1, dp2)      -> combined DeepPoly (not yet used)

    For identity skip connections: state1 is the branch-point activation
    (same shape as input_shape), and skip_layers=None.

    For non-identity skip connections (e.g. ResNet projection shortcuts):
    skip_layers holds the list of linear layers to apply to the branch-point
    activation before adding. skip_branch_shape is the shape at the branch
    point (may differ from input_shape when the skip path changes channels).

    The caller is responsible for supplying both inputs in the correct order.
    '''

    def __init__(self, layer_num, input_shape, skip_layers=None, skip_branch_shape=None):
        self.layer_num = layer_num
        self.input_shape = input_shape
        self.network = None  # assigned when building NeuralNetwork
        # For non-identity skip paths: ordered list of layer objects to apply
        # to the branch-point star/state before adding to the main path.
        self.skip_layers = skip_layers  # None = identity
        # Shape of the activation at the branch point (where skip diverges).
        self.skip_branch_shape = skip_branch_shape if skip_branch_shape is not None else input_shape
        self.freeze_attrs()

    def __str__(self):
        if self.skip_layers:
            return f'[SkipAddLayer {self.input_shape} with {len(self.skip_layers)}-layer skip transform]'
        return f'[SkipAddLayer {self.input_shape}]'

    def get_input_shape(self):
        return self.input_shape

    def get_output_shape(self):
        return self.input_shape

    # ── concrete execution ──────────────────────────────────────────────────

    def execute(self, state1, state2):
        '''Element-wise addition of two concrete states.

        state1 is the skip-path activation (at skip_branch_shape if skip_layers
        are present; at input_shape otherwise).  state2 is the main-path output
        at input_shape.  Returns the summed result at input_shape.
        '''
        if self.skip_layers:
            for layer in self.skip_layers:
                state1 = layer.execute(state1)
        assert state1.shape == self.input_shape, \
            f"state1 shape {state1.shape} != input_shape {self.input_shape}"
        assert state2.shape == self.input_shape, \
            f"state2 shape {state2.shape} != input_shape {self.input_shape}"
        return state1 + state2

    # ── abstract transformations ────────────────────────────────────────────

    def transform_star(self, star_skip, star_main):
        '''Combine two LpStar sets via element-wise addition.

        star_skip is the star at the branch point.  If skip_layers is set,
        those layers are applied to star_skip first (in-place) to produce the
        transformed skip-path star.  Then the result is added to star_main.

        Both stars (after skip transform) share the SAME parameter space u.

        The MAIN PATH star (star_main) carries the LP with all the ReLU
        split constraints that have been applied so far.  We modify it in-
        place and return it; star_skip is treated as consumed.
        '''
        # Apply non-identity skip branch layers if present
        if self.skip_layers:
            for layer in self.skip_layers:
                layer.transform_star(star_skip)

        # Add biases
        star_main.bias = star_main.bias + star_skip.bias

        # Add generator matrices element-wise.
        # In overapprox mode, ReLU splits on the main path add new generator columns
        # that the skip path's cached star doesn't have.  Pad skip with zero columns
        # so shapes match — zero is correct because the skip path contributes nothing
        # to those generators.
        if star_skip.a_mat is not None and star_main.a_mat is not None:
            n_skip = star_skip.a_mat.shape[1]
            n_main = star_main.a_mat.shape[1]
            if n_skip < n_main:
                pad = np.zeros((star_skip.a_mat.shape[0], n_main - n_skip), dtype=star_skip.a_mat.dtype)
                star_skip.a_mat = np.hstack([star_skip.a_mat, pad])
            elif n_skip > n_main:
                pad = np.zeros((star_main.a_mat.shape[0], n_skip - n_main), dtype=star_main.a_mat.dtype)
                star_main.a_mat = np.hstack([star_main.a_mat, pad])
            star_main.a_mat = star_main.a_mat + star_skip.a_mat
        elif star_skip.a_mat is not None:
            star_main.a_mat = star_skip.a_mat.copy()
        # if star_skip.a_mat is None, star_main.a_mat is already correct

        # init_bm / init_bias track the original input-space basis for counterexample recovery.
        # Both skip and main paths originate from the SAME input, so init_bm encodes the same
        # mapping in both cases (u → original_input).  Do NOT add them — that would double-count
        # the mapping and produce out-of-bounds inputs when solving for counterexamples.
        # star_main.init_bm is already correct; nothing to do.

        # LP stays on star_main unchanged (it holds all ReLU split constraints)
        return star_main

    def transform_zono(self, zono_skip, zono_main):
        '''Combine two Zonotopes via element-wise addition.

        If skip_layers is set, apply them to zono_skip first.
        Both zonotopes (after skip transform) share the same generator space.

        Modifies zono_main in-place and returns it.
        '''
        if self.skip_layers:
            for layer in self.skip_layers:
                layer.transform_zono(zono_skip)

        zono_main.center = zono_main.center + zono_skip.center
        if zono_skip.mat_t is not None and zono_main.mat_t is not None:
            n_skip = zono_skip.mat_t.shape[1]
            n_main = zono_main.mat_t.shape[1]
            if n_skip < n_main:
                pad = np.zeros((zono_skip.mat_t.shape[0], n_main - n_skip), dtype=zono_skip.mat_t.dtype)
                zono_skip.mat_t = np.hstack([zono_skip.mat_t, pad])
            elif n_skip > n_main:
                pad = np.zeros((zono_main.mat_t.shape[0], n_skip - n_main), dtype=zono_main.mat_t.dtype)
                zono_main.mat_t = np.hstack([zono_main.mat_t, pad])
                # Extend init_bounds so it matches the new generator count
                zono_main.init_bounds = zono_main.init_bounds + [(-1.0, 1.0)] * (n_skip - n_main)
                zono_main.init_bounds_nparray = None
                zono_main.pos1_gens = None
                zono_main.neg1_gens = None
            zono_main.mat_t = zono_main.mat_t + zono_skip.mat_t
        elif zono_skip.mat_t is not None:
            zono_main.mat_t = zono_skip.mat_t.copy()
        return zono_main

    def transform_deeppoly(self, dp1, dp2):
        'Combine two DeepPoly representations (stub – not used by current verification path)'
        raise NotImplementedError("transform_deeppoly for SkipAddLayer not yet implemented")


class PoolingLayer(Freezable):
    '''a 2d max/mean pooling layer (multi channel)
    '''

    def __init__(self, layer_num, kernel_size, prev_layer_output_shape, method='max'):
        self.layer_num = layer_num
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.prev_layer_output_shape = prev_layer_output_shape

        self.network = None # assigned on network construction

        assert method in ['max', 'mean'], f"unknown method: {method}"

        self.method = method

        self.freeze_attrs()

    def __str__(self):
        s = self.kernel_size

        return f'[PoolingLayer ({self.method}) {s}x{s} with stride {self.stride}, ' + \
               f'input shape {self.get_input_shape()} and output shape {self.get_output_shape()}]'

    def get_input_shape(self):
        'get the input shape to this layer'

        return self.prev_layer_output_shape

    def get_output_shape(self):
        'get the output shape from this layer'

        s = self.kernel_size

        height = self.prev_layer_output_shape[0] // s
        width = self.prev_layer_output_shape[1] // s

        rv = [height, width]

        if len(self.prev_layer_output_shape) > 2:
            rv += self.prev_layer_output_shape[2:]

        return tuple(rv)

    def execute(self, state, save_branching=False):
        '''execute pooling layer, potentially saving branching informaton

        branching info will be an int for each output (if max pool), or possibly a LIST of ints (if two inputs match)
        '''

        Timers.tic('execute PoolingLayer')

        ksize = self.kernel_size

        assert len(state.shape) == 3
        assert state.shape[0] % ksize == 0
        assert state.shape[1] % ksize == 0

        if save_branching:
            rv = self._execute_with_branching(state)
        else:
            rv = self._execute_without_branching(state)

        Timers.toc('execute PoolingLayer')

        return rv

    # based on code from:
    # https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
    def _execute_without_branching(self, state):
        'fast max/mean pooling without storing branching information'

        ksize = self.kernel_size

        ny = state.shape[0] // ksize
        nx = state.shape[1] // ksize

        new_shape = (ny, ksize, nx, ksize) + state.shape[2:]

        if self.method == 'max':
            rv = np.nanmax(state.reshape(new_shape), axis=(1, 3))
        else:
            assert self.method == 'mean'
            rv = np.nanmean(state.reshape(new_shape), axis=(1, 3))

        return rv

    def _execute_with_branching(self, state):
        '''execute pooling layer on a concrete state

        branch_list will be an int for each output (if max pool), or possibly a LIST of ints (if two inputs match)

        note: this is about 50x slower than without branching on a 224x224x3 input
        '''

        Timers.tic('execute_pooling_with_branching')

        ksize = self.kernel_size

        height = state.shape[0] // ksize
        width = state.shape[1] // ksize
        depth = state.shape[2]

        if self.method == 'max':
            output = np.full((height, width, depth), -np.inf, dtype=float)
            branch_list = [None] * (depth * width * height)
        else:
            output = np.zeros((height, width, depth), dtype=float)
            branch_list = []

        for d in range(state.shape[2]):
            depth_offset = d * (width * height)

            for row_index in range(state.shape[0]):
                output_row = row_index // ksize
                height_offset = output_row * width

                for col_index in range(state.shape[1]):
                    block_index = col_index // ksize

                    val = state[row_index, col_index, d]

                    if self.method == 'max':
                        epsilon = 1e-9

                        if val - epsilon > output[output_row, block_index, d]:
                            # new max value
                            output[output_row, block_index, d] = val

                            max_index = col_index % ksize
                            row_in_block = row_index % ksize
                            mindex = row_in_block * ksize + max_index
                            branch_list[depth_offset + height_offset + block_index] = mindex
                        elif val + epsilon > output[output_row, block_index, d]:
                            # two branches are both possible (within epsilon tolerance), both should be in branch string

                            output[output_row, block_index, d] = max(output[output_row, block_index, d], val)

                            max_index = col_index % ksize
                            row_in_block = row_index % ksize
                            mindex = row_in_block * ksize + max_index
                            bindex = depth_offset + height_offset + block_index

                            if isinstance(branch_list[bindex], int):
                                branch_list[bindex] = [branch_list[bindex], mindex]
                            else:
                                branch_list[bindex].append(mindex)

                    else:
                        output[output_row, block_index, d] += val

        if self.method == 'mean':
            divider = self.kernel_size**2
            output = output / divider

        rv = (output, branch_list)

        Timers.toc('execute_pooling_with_branching')

        return rv

def images_to_init_box(min_image, max_image):
    'create an initial box from a min and max image'

    min_vec = nn_flatten(min_image)
    max_vec = nn_flatten(max_image)
    rv = []

    for a, b in zip(min_vec, max_vec):
        rv.append((a, b))

    return rv

def nn_flatten(image, order='C'):
    'flatten a multichannel image to a 1-d array'

    return image.flatten(order)

def nn_unflatten(image, shape, order='C'):
    '''unflatten to a multichannel image from a 1-d array

    this uses reshape, so may not be a copy
    '''

    assert len(image.shape) == 1

    rv = image.reshape(shape, order=order)

    return rv

def convert_weights(weights):
    'convert weights from a list format to an np.array format'

    layers = [] # list of np.array for each layer

    for weight_mat in weights:
        layers.append(np.array(weight_mat, dtype=float))

    # this prevents python from attempting to broadcast the layers together
    rv = np.empty(len(layers), dtype=object)
    rv[:] = layers

    return rv

def convert_biases(biases):
    'convert biases from a list format to an np.array format'

    layers = [] # list of np.array for each layer

    for biases_vec in biases:
        bias_ar = np.array(biases_vec, dtype=float)
        bias_ar.shape = (len(biases_vec),)

        layers.append(bias_ar)

    # this prevents python from attempting to broadcast the layers together
    rv = np.empty(len(layers), dtype=object)
    rv[:] = layers

    return rv

def weights_biases_to_nn(weights, biases, dtype=None):
    '''create a NeuralNetwork from a weights and biases matrix

    this assumes every layer is a fully-connected layer followed by a ReLU, except for the last one
    '''

    if isinstance(weights, list):
        weights = convert_weights(weights)

    if isinstance(biases, list):
        biases = convert_biases(biases)

    num_layers = weights.shape[0]
    assert biases.shape[0] == num_layers, f"nn has {num_layers} layers, but biases shape was {biases.shape}"

    layers = []

    index = 0

    for i, (layer_weights, layer_biases) in enumerate(zip(weights, biases)):
        add_relu = i < num_layers - 1

        if dtype is not None:
            layer_weights = layer_weights.astype(dtype)
            layer_biases = layer_biases.astype(dtype)

        layers.append(FullyConnectedLayer(index, layer_weights, layer_biases))
        index += 1

        if add_relu:
            layers.append(ReluLayer(index, layers[-1].get_output_shape()))
            index += 1

    return NeuralNetwork(layers)

