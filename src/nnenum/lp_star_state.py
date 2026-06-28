'''
LP Star State (includes enumeration state variables)
Stanley Bak
'''

import numpy as np

from nnenum.lp_star import LpStar
from nnenum.prefilter import Prefilter
from nnenum.timerutil import Timers
from nnenum.util import Freezable, compress_init_box
from nnenum.network import FullyConnectedLayer, ReluLayer, FlattenLayer, AddLayer, MatMulLayer, SkipAddLayer, BranchRestoreLayer
from nnenum.specification import DisjunctiveSpec

from nnenum.settings import Settings

class IntervalFallbackSafe(Exception):
    'Raised when IBP proves the current branch safe during interval fallback'

class IntervalFallbackUnknown(Exception):
    'Raised when the star is too large to analyze and IBP is inconclusive — branch is skipped'

class LpStarState(Freezable):
    'variables and methods associated with verification using lp star representation'

    # if not None, split to get to this branch
    TARGET_BRANCH_TUPLE = None
    # 1-9 7, unsafe: '++++-++--+++++++++---++-++-++++++-++-++++--+++++'
    #                '++++++-+++-++---+++----++-+++-++'
    #                '++++++-+++-++---+++----++-+++-++'

    def __init__(self, uncompressed_init_box=None, spec=None, safe_spec_list=None):

        self.star = None
        self.prefilter = None

        self.cur_layer = 0
        self.work_frac = 1.0 # fraction of work represented by this star

        self.should_try_overapprox = True

        if safe_spec_list is not None:
            self.safe_spec_list = safe_spec_list
        elif isinstance(spec, DisjunctiveSpec):
            self.safe_spec_list = [False] * len(spec.spec_list)
        else:
            self.safe_spec_list = None

        # a list of 3-tuples describing the branching taken for this star (layer_index, neuron_index, branch_type)
        # see network.execute for the values of branch_type used by the different layer types
        self.branch_tuples = []

        self.distance_to_unsafe = None

        # star_cache[layer_num] = copy of star AFTER processing that layer.
        # Used to provide the skip-path star when a SkipAddLayer is reached.
        # Only layers whose layer_num appears as a skip source in the network's
        # dag_predecessors need to be cached; we cache all linear layers for
        # simplicity and memory overhead is small for typical ResNet depths.
        self.star_cache = {}

        # star_cache[layer_num] = copy of star AFTER processing that layer.
        # Used to provide the skip-path star when a SkipAddLayer is reached.
        # Only layers whose layer_num appears as a skip source in the network's
        # dag_predecessors need to be cached; we cache all linear layers for
        # simplicity and memory overhead is small for typical ResNet depths.
        self.star_cache = {}

        if uncompressed_init_box is not None:
            assert isinstance(uncompressed_init_box, np.ndarray), "init bounds should be given in a numpy array"
            assert uncompressed_init_box.dtype in [np.float32, np.float64], \
                f"init bounds dtype was not floating-point type: {uncompressed_init_box.dtype}"

            Timers.tic('from_init_box')
            self.from_init_box(uncompressed_init_box)
            Timers.toc('from_init_box')

        self.freeze_attrs()

    def __del__(self):
        # delete the circular reference which would prevent the memory from being freed
        if self.prefilter is not None and self.prefilter.output_bounds:
            self.prefilter.output_bounds.prefilter = None

    def __str__(self):
        split_str = "no splits"
        n = self.remaining_splits()

        if n > 0:
            str_list = [str(s) for s in self.prefilter.output_bounds.branching_neurons]
            split_str = f"{n} splits remaining: " + ", ".join(str_list)

        return f"LpStateState at layer {self.cur_layer} with {split_str}"

    def branch_str(self):
        'get the branch tuples string'

        assert self.branch_tuples is not None

        rv = "".join(["+" if tup[2] else "-" for tup in self.branch_tuples])

        return rv

    def remaining_splits(self):
        'get the number of remaining splits on the current layer'

        rv = 0

        # for the first star (initial set), output_bounds is NOT assigned
        if self.prefilter and self.prefilter.output_bounds:
            rv = self.prefilter.output_bounds.branching_neurons.size

        return rv

    def from_init_star(self, star):
        'initialize from an initial LpStar (makes a copy)'

        self.star = star.copy() # copy the star
        self.prefilter = Prefilter()
        self.prefilter.init_from_star(self.star)

    def from_init_box(self, uncompressed_init_box):
        'initialize from an initial box'

        Timers.tic('make bm')

        if Settings.COMPRESS_INIT_BOX:
            init_bm, init_bias, init_box = compress_init_box(uncompressed_init_box)
        else:
            dims = len(uncompressed_init_box)
            #init_bm = np.identity(dims)
            #init_bias = np.zeros(dims)
            init_bm = None
            init_bias = None
            init_box = uncompressed_init_box

        Timers.toc('make bm')

        # for finding concrete counterexamples
        Timers.tic('star')
        self.star = LpStar(init_bm, init_bias, init_box)
        Timers.toc('star')

        self.prefilter = Prefilter()
        self.prefilter.init_from_uncompressed_box(uncompressed_init_box, self.star, init_box)

    def is_finished(self, network):
        'is the current star finished?'

        return self.cur_layer >= len(network.layers)

    def propagate_up_to_split(self, network, start_time, spec=None):
        'propagate up to the next split or until we finish with the network'

        depth = len(self.branch_tuples)

        while not self.is_finished(network):
            layer = network.layers[self.cur_layer]

            if isinstance(layer, ReluLayer):
                # Cache before each ReLU too: a ReLU output can be a skip source
                # for a downstream SkipAddLayer (e.g. ResNet shortcut bypasses ReLU block).
                self._maybe_cache_star(network)

                # Cache before each ReLU too: a ReLU output can be a skip source
                # for a downstream SkipAddLayer (e.g. ResNet shortcut bypasses ReLU block).
                self._maybe_cache_star(network)

                if self.prefilter.output_bounds is None:
                    # start of a relu layer
                    self.prefilter.init_relu_layer(self.star, layer, start_time, depth)

                if self.prefilter.output_bounds.branching_neurons.size > 0:
                    # Before breaking for a split: check if the collapse is needed soon.
                    # If the next conv layer would OOM, collapse + IBP now instead of splitting.
                    if (Settings.SPARSE_STAR
                            and Settings.SPARSE_INTERVAL_FALLBACK
                            and spec is not None):
                        from scipy.sparse import issparse
                        if issparse(self.star.a_mat):
                            from nnenum.network import Convolutional2dLayer
                            # Find the next conv layer after the current relu
                            next_conv = None
                            for k in range(self.cur_layer + 1, len(network.layers)):
                                if isinstance(network.layers[k], Convolutional2dLayer):
                                    next_conv = network.layers[k]
                                    break
                            if next_conv is not None and next_conv._would_exceed_memory(self.star):
                                # Collapse now using current relu's zono bounds
                                bounds = self.prefilter.zono.box_bounds()
                                lb, ub = bounds[:, 0], bounds[:, 1]
                                if Settings.TRY_IBP:
                                    from nnenum.overapprox import try_ibp_from_bounds
                                    if try_ibp_from_bounds(lb, ub, network, self.cur_layer, spec):
                                        raise IntervalFallbackSafe()
                                # IBP inconclusive — can't verify this branch within memory budget.
                                # Raise IntervalFallbackUnknown so the caller can handle appropriately.
                                raise IntervalFallbackUnknown()
                    break

                self.next_layer()
            else:
                # Cache the star before processing so skip-path stars are
                # available when a SkipAddLayer is reached.
                self._maybe_cache_star(network)

                # Cache the star before processing so skip-path stars are
                # available when a SkipAddLayer is reached.
                self._maybe_cache_star(network)

                # non-relu layer
                # IntervalFallbackSafe is intentionally NOT caught here — let it propagate
                # to the caller (worker.py / enumerate.py) where it's treated as "branch safe"
                self.apply_linear_layer(network, spec=spec)

                self.next_layer()

    def _maybe_cache_star(self, network):
        '''Cache the current star (and simulation) keyed by cur_layer.
        Always caches so skip-source stars are available when SkipAddLayer is reached.'''
        self.star_cache[self.cur_layer] = self.star.copy()
        if self.prefilter and self.prefilter.simulation is not None:
            self.prefilter.simulation_cache[self.cur_layer] = self.prefilter.simulation[1].copy()

    def _maybe_cache_star(self, network):
        '''Cache the current star (and simulation) keyed by cur_layer.
        Always caches so skip-source stars are available when SkipAddLayer is reached.'''
        self.star_cache[self.cur_layer] = self.star.copy()
        if self.prefilter and self.prefilter.simulation is not None:
            self.prefilter.simulation_cache[self.cur_layer] = self.prefilter.simulation[1].copy()

    def next_layer(self):
        'advance to the next layer'

        self.cur_layer += 1

        if self.prefilter:
            self.prefilter.clear_output_bounds()

    def apply_linear_layer(self, network, spec=None):
        'apply linear transformation part of a layer'

        Timers.tic('starstate.apply_linear_layer')

        layer = network.layers[self.cur_layer]
        assert not isinstance(layer, ReluLayer)
        assert self.star
        assert self.prefilter

        # Cache zonotope BEFORE applying the layer (matches star_cache semantics:
        # star_cache[k] and zono_cache[k] both hold the state BEFORE layer k runs).
        self._maybe_cache_zono(network)

        # --- Sparse-to-interval fallback ---
        from nnenum.network import Convolutional2dLayer
        from scipy.sparse import issparse
        if (Settings.SPARSE_STAR
                and Settings.SPARSE_INTERVAL_FALLBACK
                and isinstance(layer, Convolutional2dLayer)
                and issparse(self.star.a_mat)
                and layer._would_exceed_memory(self.star)):
            G = self.star.a_mat.shape[1]
            N = self.star.a_mat.shape[0]
            if Settings.PRINT_OUTPUT:
                print(f"[interval fallback] L{self.cur_layer}: collapsing {G} gens -> {N} interval gens",
                      flush=True)
            bounds = self.prefilter.zono.box_bounds()  # (N, 2), sparse-safe
            lb, ub = bounds[:, 0], bounds[:, 1]

            # Optional IBP fast-path before building the collapsed star
            if Settings.TRY_IBP and spec is not None:
                from nnenum.overapprox import try_ibp_from_bounds
                if try_ibp_from_bounds(lb, ub, network, self.cur_layer, spec):
                    if Settings.PRINT_OUTPUT:
                        print(f"[interval fallback] IBP proved safe at L{self.cur_layer}", flush=True)
                    Timers.toc('starstate.apply_linear_layer')
                    raise IntervalFallbackSafe()

            self.star.collapse_to_interval_star_from_bounds(lb, ub)
            N_new = self.star.a_mat.shape[1]
            self.prefilter.zono.mat_t = self.star.a_mat
            self.prefilter.zono.center = self.star.bias
            self.prefilter.zono.init_bounds = [(-1.0, 1.0)] * N_new
            self.prefilter.zono.init_bounds_nparray = None
            self.prefilter.zono.pos1_gens = None
            self.prefilter.zono.neg1_gens = None

        if isinstance(layer, BranchRestoreLayer):
            # Restore star (and zono) from the cached checkpoint for this branch.
            # We restore a_mat and bias (the output representation) from the cached star,
            # but PRESERVE the current star's LP (lpi) — ReLU split constraints accumulated
            # along the current path must not be discarded.  Discarding them would make the
            # LP too loose, allowing spurious counterexamples (unconfirmed SAT).
            src_key = network.dag_predecessors[self.cur_layer][0]
            assert src_key in self.star_cache, (
                f"BranchRestoreLayer {self.cur_layer}: source {src_key} "
                f"not in star_cache (keys: {list(self.star_cache.keys())})")
            cached_star = self.star_cache[src_key]
            # Restore only the output-representation fields; keep current LP constraints.
            self.star.a_mat = cached_star.a_mat.copy()
            self.star.bias = cached_star.bias.copy()
            # init_bm / init_bias encode the original-input basis and should not change.
            # (Both current and cached stars map the SAME original input.)
            if self.prefilter.zono is not None:
                # Rebuild zono from the restored star so mat_t/center are in sync.
                # zono_cache may not exist for BranchRestore/SkipAdd-adjacent layers,
                # so fall back to creating a fresh zono from self.star.
                from nnenum.zonotope import Zonotope
                src_zono = self.prefilter.zono_cache.get(src_key)
                if src_zono is not None:
                    init_bounds = list(src_zono.init_bounds)
                else:
                    # No cached zono; use [-1,1] per generator (conservative placeholder)
                    n_gens = self.star.a_mat.shape[1] if self.star.a_mat is not None else 0
                    init_bounds = [(-1.0, 1.0)] * n_gens
                self.prefilter.zono = Zonotope(
                    self.star.bias,   # share center with star (same array → is check passes)
                    self.star.a_mat,  # share mat_t with star (same array → is check passes)
                    init_bounds)
            if self.prefilter.simulation is not None:
                src_sim = self.prefilter.simulation_cache.get(src_key)
                if src_sim is not None:
                    self.prefilter.simulation[1] = src_sim.copy()
        elif isinstance(layer, SkipAddLayer):
            # Retrieve the cached skip-path star
            skip_source = network.dag_predecessors[self.cur_layer][0]
            assert skip_source in self.star_cache, (
                f"SkipAddLayer {self.cur_layer}: skip source {skip_source} "
                f"not in star_cache (keys: {list(self.star_cache.keys())})")
            star_skip = self.star_cache[skip_source]
            # transform_star modifies self.star in-place and returns it
            layer.transform_star(star_skip, self.star)
            # update prefilter's zonotope reference
            if self.prefilter.zono is not None:
                from nnenum.zonotope import Zonotope
                # fetch cached zono for skip source and combine
                cached_skip_zono = self.prefilter.zono_cache.get(skip_source)
                # Deep-copy so transform_zono doesn't mutate the cached object
                # (the cache is shared via shallow dict copy between split siblings).
                skip_zono = cached_skip_zono.deep_copy() if cached_skip_zono is not None else None
                if skip_zono is not None:
                    # Build fresh main zono from the current star (after transform_star), then combine.
                    n_gens = self.star.a_mat.shape[1]
                    n_old = len(self.prefilter.zono.init_bounds)
                    if n_gens != n_old:
                        # transform_star grew a_mat (n_skip > n_main); extend init_bounds
                        self.prefilter.zono.init_bounds = self.prefilter.zono.init_bounds + \
                            [(-1.0, 1.0)] * (n_gens - n_old)
                        self.prefilter.zono.init_bounds_nparray = None
                        self.prefilter.zono.pos1_gens = None
                        self.prefilter.zono.neg1_gens = None
                    # Re-bind mat_t and center before running transform_zono
                    self.prefilter.zono.mat_t = self.star.a_mat
                    self.prefilter.zono.center = self.star.bias
                    layer.transform_zono(skip_zono, self.prefilter.zono)
                # re-bind so prefilter still sees the live zono
                self.prefilter.zono.mat_t = self.star.a_mat
                self.prefilter.zono.center = self.star.bias
            # Update simulation: sim_skip_transformed + sim_current
            # (simulation is a concrete witness point tracked for heuristics)
            if self.prefilter.simulation is not None:
                skip_sim = self.prefilter.simulation_cache.get(skip_source)
                if skip_sim is not None:
                    if layer.skip_layers:
                        from nnenum.network import nn_unflatten, nn_flatten
                        skip_sim_state = nn_unflatten(skip_sim, layer.skip_branch_shape)
                        for sl in layer.skip_layers:
                            skip_sim_state = sl.execute(skip_sim_state)
                        skip_sim = nn_flatten(skip_sim_state)
                    self.prefilter.simulation[1] = self.prefilter.simulation[1] + skip_sim
            # zonotope sanity check
            assert self.prefilter.zono.mat_t is self.star.a_mat
            assert self.prefilter.zono.center is self.star.bias
        else:
            layer.transform_star(self.star)

            # update zonotope shallow copy
            self.prefilter.zono.mat_t = self.star.a_mat
            self.prefilter.zono.center = self.star.bias

            self.prefilter.apply_linear_layer(layer, self.star)
            #SDW 2026-06-28: Why are there two of these? Commenting one out...
            #self.prefilter.apply_linear_layer(layer, self.star)

        Timers.toc('starstate.apply_linear_layer')

    def _maybe_cache_zono(self, network):
        '''Cache the current zonotope state keyed by cur_layer (for SkipAdd).'''
        from nnenum.zonotope import Zonotope
        z = self.prefilter.zono
        if z is not None:
            self.prefilter.zono_cache[self.cur_layer] = Zonotope(
                z.center.copy(),
                z.mat_t.copy() if z.mat_t is not None else None,
                z.init_bounds.copy())

    def split_enumerate(self, i, network, spec, start_time):
        '''
        helper for execute_relus

        split using enumerative strategy, returns the child LpStarState object

        ss is the lp star state
        i is the output (neuron) index we're splitting on
        '''

        #print(f".state spliting on neuron {i}")

        Timers.tic('split_enumerate')

        child = LpStarState()
        child.star = self.star.copy()

        # prefilter gets copied later

        if self.safe_spec_list is not None:
            child.safe_spec_list = self.safe_spec_list.copy()

        child.cur_layer = self.cur_layer

        # copy star_cache so child has access to skip-path stars
        child.star_cache = {k: v.copy() for k, v in self.star_cache.items() if v is not None}

        # copy star_cache so child has access to skip-path stars
        child.star_cache = {k: v.copy() for k, v in self.star_cache.items() if v is not None}

        # split work among 2 children
        self.work_frac /= 2.0
        child.work_frac = self.work_frac

        # choose which branch to go down
        if not LpStarState.TARGET_BRANCH_TUPLE:
            #sim_is_positive = self.prefilter.simulation[1][i] >= 0
            #self_gets_positive = sim_is_positive
            self_gets_positive = True
        else:
            self_gets_positive = LpStarState.TARGET_BRANCH_TUPLE[0] == '+'
            LpStarState.TARGET_BRANCH_TUPLE = LpStarState.TARGET_BRANCH_TUPLE[1:]

            took = 'pos' if self_gets_positive else 'neg'
            print(f"Info: Using TARGET_BRANCH_TUPLE for splitting on {i}, took {took}")

        # first do child, as it may be infeasible
        if self_gets_positive:
            pos, neg = self, child
        else:
            neg, pos = self, child

        ### ADD INITIAL STATE INTERSECTION
        row = self.star.get_row(i)
        bias = self.star.bias[i]

        # pos gets output >= 0
        # neg gets output <= 0

        Timers.tic('check child feasible')
        # checking feasibility doesn't add too much time as it's done again layer for witnesses
        if self_gets_positive:
            neg.star.lpi.add_dense_row(row, -bias)
            neg.star.set_row_zero(i)
            neg.star.bias[i] = 0 # reset the current bias as well

            #child_feasible = neg.star.lpi.is_feasible()
            child_feasible = True
        else:
            pos.star.lpi.add_dense_row(-row, bias)

            #child_feasible = pos.star.lpi.is_feasible()
            child_feasible = True

        Timers.toc('check child feasible')

        if not child_feasible:
            rv = None

            # if eager is false this can happen?

            ob = self.prefilter.output_bounds
            ob.branching_neurons = ob.branching_neurons[1:]

        else:
            rv = child

            if self_gets_positive:
                pos.star.lpi.add_dense_row(-row, bias)
            else:
                ### ASSIGN NEURON i OUTPUT
                # neg has 0 output
                neg.star.set_row_zero(i)
                neg.star.bias[i] = 0 # reset the current bias as well

            # update branch_tuples
            child.branch_tuples = self.branch_tuples.copy()
            pos.branch_tuples.append((pos.cur_layer, i, True))
            neg.branch_tuples.append((neg.cur_layer, i, False))

            Timers.tic('prefilter_split_relu')

            depth = len(self.branch_tuples)
            child.prefilter = self.prefilter.split_relu(i, pos.star, neg.star, self_gets_positive, start_time, depth)

            assert child.prefilter.zono.mat_t is child.star.a_mat
            assert child.prefilter.zono.center is child.star.bias

            # copy skip-connection caches to child's prefilter
            child.prefilter.zono_cache = dict(self.prefilter.zono_cache)
            child.prefilter.simulation_cache = {k: v.copy() for k, v in self.prefilter.simulation_cache.items()}

            # copy skip-connection caches to child's prefilter
            child.prefilter.zono_cache = dict(self.prefilter.zono_cache)
            child.prefilter.simulation_cache = {k: v.copy() for k, v in self.prefilter.simulation_cache.items()}

            Timers.toc('prefilter_split_relu')

        Timers.toc('split_enumerate')

        return rv

    def do_first_relu_split(self, network, spec, start_time):
        '''
        do the first relu split for the current layer

        returns a new StarState from the split
        '''

        Timers.tic('do_first_relu_split')

        layer = network.layers[self.cur_layer]
        assert isinstance(layer, ReluLayer)
        assert self.prefilter.output_bounds is not None
        assert self.prefilter.output_bounds.branching_neurons.size > 0

        index = self.prefilter.output_bounds.branching_neurons[0]

        rv = self.split_enumerate(index, network, spec, start_time)

        Timers.toc('do_first_relu_split')

        return rv
