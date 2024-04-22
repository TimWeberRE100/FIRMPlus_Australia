# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 08:28:16 2024

@author: u6942852
"""

import numpy as np 
from numba import jit, njit, prange, float64, int64, boolean
from numba.experimental import jitclass
from copy import deepcopy

# njit compatible timer
import ctypes
import platform

if platform.system() == "Windows":
    from ctypes.util import find_msvcrt

    __LIB = find_msvcrt()
    if __LIB is None:
        __LIB = "msvcrt.dll"
else:
    from ctypes.util import find_library

    __LIB = find_library("c")

clock = ctypes.CDLL(__LIB).clock
clock.argtypes = []

@njit()
def hrects_is_same(h1, h2, tol=1e-10):
    ub_same = (h1.ub == h2.ub).prod() == 1 
    lb_same = (h1.lb == h2.lb).prod() == 1 
    c_same = (h1.centre == h2.centre).prod() == 1
    f_same = h1.f == h2.f
    return ub_same and lb_same and c_same and f_same

@njit()
def hrects_border(h1, h2, tol = 1e-10):
    assert h1.ndim == h2.ndim
    ndim = h1.ndim
    # directions where the domains of each h2 touch
    touch = (((h2.ub - h1.lb) >= -tol) * 
             ((h1.ub - h2.lb) >= -tol))

    # the domains in each direction touch   
    if not (touch.sum() == ndim):
        return False

    # in ndim-1 directions the directions' domains overlap (either perfectly, or one inside another)
    overlap = (signs_of_array(h2.ub - h1.ub, tol) ==
               signs_of_array(h1.lb - h2.lb, tol))
    if not (overlap.sum() == ndim-1):
        return False

    # in exactly one direction domains do not overlap (although they may touch)
    if not ((~overlap).sum() == 1):
        return False
    
    # adjacent (ub=lb or lb=ub) (higher OR lower) in exactly one dimension
    adjacency = ((np.abs(h2.ub - h1.lb) < tol) +
                 (np.abs(h2.lb - h1.ub) < tol)) 
    if not (adjacency.sum() == 1):
        return False

    # Direction of adjacency is the direction not overlapping
    if not ((adjacency == ~overlap).prod() == 1):
        return False
    return True

@njit()
def signs_of_array(arr, tol=1e-10):
    arr[np.abs(arr)<tol] = 0 
    return np.sign(arr)

spec = [
    ('centre', float64[:]),
    ('ndim', int64),
    ('f', float64),
    ('parent_f', float64),
    ('lb', float64[:]),
    ('ub', float64[:]),
    ('rdif', float64),
    ('adif', float64),
    ('n', int64),
    ('length_inds', float64[:]),
    ('volume', float64),
    ('barren', boolean),
    ('has_children', boolean),
    ]

@jitclass(spec)
class hyperrectangle():
    def __init__(self, centre, f, lb, ub, parent_f, parent_n, log_min_l):
        self.centre = centre
        self.ndim = len(centre)

        self.f, self.parent_f = float(f), parent_f
        self.lb, self.ub = lb, ub
        self.rdif, self.adif = self.f/self.parent_f, self.f-self.parent_f
        self.n = parent_n + 1
        self.volume = (self.ub-self.lb).prod()
        self.length_inds = np.log10(self.ub-self.lb)
        self.barren = (self.length_inds < log_min_l).prod()
        self.has_children=False

@njit()
def gen_inds(npop, processes, maxvectorwidth):
    vsize = npop//processes + 1 if npop%processes != 0 else npop//processes
    vsize = min(vsize, maxvectorwidth, npop)
    range_gen = range(npop//vsize + 1) if npop%vsize != 0 else range(npop//vsize)
    indcs = [np.arange(n*vsize, min((n+1)*vsize, npop), dtype=np.int64) for n in range_gen]
    inds = np.empty((len(indcs), vsize), dtype=np.int64)
    for i in range(len(indcs)):
        inds[i] = indcs[i]
    return inds

@njit(parallel=True)
def gen_boolmatrix(ndim):
    z = np.empty((2**ndim, ndim), dtype=np.bool_)
    for i in prange(ndim):
        z[:,i] = _loop(ndim, i)
    return z

@njit(parallel=True)        
def _loop(n, i):
    a = np.zeros(2**n, dtype=np.bool_)
    i2 = 2**i
    for j in range(0, 2**n, i2*2):
        a[j:j+i2] = True
    return a

@jit(nopython=False)
def _func_wrapper(x, func, vectorizable, inds, f_args):
    results = np.empty(len(x.T), dtype=np.float64)
    if vectorizable: 
        # raise NotImplementedErrorNotImplementedError(
# """Requires direct source code changes because of jit static typing. Uncomment
#  lines 216-221, raise error at 227-229, comment out 223-226, and this raise error""")
        if len(f_args) == 0:
            for ind in inds:
                results[ind] = func(x[:, ind])
        else:
            for ind in inds:
                results[ind] = func(x[:, ind], *f_args)
    else: 
        # if len(f_args) == 0:
        #     results[0] = func(x[:])
        # else:
        #     results[0] = func(x[:], *f_args)
        raise NotImplementedError(
"""Requires direct source code changes because of jit static typing. Comment out
 lines 216-221, this raise error, uncomment 223-226, and raise error at 213-215""")
    return results

@njit()
def _bound(lb, ub, indx):
    inv = ~indx
    base = np.empty((indx.shape[0], 2), dtype=np.float64)
    base[indx], base[inv] = lb[indx], ub[inv]
    
    return base[:,0], base[:,1]

@njit(parallel=True)
def _generate_bounds(hrect, indcs, dims):
    lc = np.hstack((np.atleast_2d(hrect.lb).T, np.atleast_2d(hrect.centre).T))[dims,:]
    lu = np.hstack((np.atleast_2d(hrect.centre).T, np.atleast_2d(hrect.ub).T))[dims,:]
    
    lbs = np.repeat(hrect.lb, len(indcs)).reshape((len(hrect.lb), len(indcs))).T
    ubs = np.repeat(hrect.ub, len(indcs)).reshape((len(hrect.ub), len(indcs))).T
    
    for i in prange(len(indcs)):
        lbs[i, dims], ubs[i, dims] = _bound(lc, lu, indcs[i])
        
    return lbs, ubs

@njit()
def _centre(arr1, arr2, indx):
    inv = ~indx
    base = np.empty(indx.shape, dtype=np.float64)
    base[indx], base[inv] = arr1[indx], arr2[inv]
    return base

@njit(parallel=True)
def _generate_centres(hrect, indcs, dims):
    l = (hrect.lb + hrect.centre)[dims]/2
    u = (hrect.centre + hrect.ub)[dims]/2
    
    centres = np.repeat(hrect.centre, len(indcs)).reshape((len(hrect.centre), len(indcs))).T
    for i in prange(len(indcs)):
        centres[i, dims] = _centre(l, u, indcs[i])
    return centres

@jit(nopython=False)
def _divide_hrect(func, hrect, dims, vectorizable, maxvectorwidth, f_args, log_min_l):
    min_length_reached = hrect.length_inds < log_min_l
    # do not split along min_length axes
    dims = np.array([i for i in dims if not min_length_reached[i]])
    dims = dims
    if len(dims) == 0:
        # do not lose hrect - may be splittable along different axis
        return [hrect] 
    
    indcs = gen_boolmatrix(len(dims))
    n_new = 2**len(dims)

    centres = _generate_centres(hrect, indcs, dims)
    lbs, ubs = _generate_bounds(hrect, indcs, dims)
    f_values = _func_wrapper(centres.T, func, vectorizable, gen_inds(n_new, 1, maxvectorwidth), f_args)
    
    pf, pn = hrect.f, hrect.n
    
    hrects = [hyperrectangle(
        centres[k], f_values[k], lbs[k], ubs[k], pf, pn, log_min_l) 
        for k in range(n_new)]
    return hrects

def Direct(
    func, 
    bounds,
    f_args=(),
    maxiter = np.inf,
    maxfev = np.inf,
    callback = None,
    vectorizable = True,
    maxvectorwidth = np.inf,
    rect_dim = -1,
    population = 1,
    min_length = -np.inf,
    disp = False,
    locally_biased=False,
):
    """
    DIRECT (DIviding RECTangles) optimiser 
    This optimiser takes the optimisation decision space and searches by 
    iteratively splitting it into rectangles and sampling the rectangle centre
    (hyperrectangles for ndim>2)
    
    This implementation was built specifically for RE100 | FIRM and has some 
    utilities to reflect this. 
    
    func            - function to be optimised. Should return a numeric values
    bounds          - the problem Must be bounded. Bounds should be a tuple of 
                        lower bounds, upper bounds. Lower and upper bounds should
                        each be an array of the same length as the decision vector
    f_args          - additional arguments to be supplied to objective 
    maxiter         - maximum iterations to perform (default: inf)
    maxfev          - maximum number of function evaluations to perform 
    callback        - callable. The elite hyperrectangle will be passed to it 
                        after each iteration
    vectorizable    - boolean. Whether multiple candidate arrays can be passed to 
                        to the objective function at once. 
                    - It is highly recommended to use a vectorizable version
                        of FIRM.
    maxvectorwidth  - maximum number of vectors to be passed to the objective 
                        function at once. For FIRM this will be determined by your
                        system limits especially RAM. 
                    - It is highly recommended to run a vector-width-tuning script
                        to find your maximum vector size and then choose a max vector
                        width accordingly. 
                    - Also note that the optimiser will only pass a maximum of 
                        2**rect_dim to the objective function.
    rect_dim        - number of directions to split rectangles on at once. This 
                        keeps the number of rectangles in each iteration manageable. 
                        Optimizer will cycle through the directions automatically.
                    - It is recommended to choose a rect_dim such that 2**rect_dim 
                        is below but as close as possible to maxvectorwidth.
    population      - number of local optima to pursue at once. The optimizer will 
                        split the {population} best rectangles, and any adjacent rectangles.
    min_length      - maximum resolution of the solution. Should be number or array 
                        broadcastable to the solution vector. Default: -inf
    disp            - boolean. Prints out at each iteration.
    locally_biased  - boolean. True - converges at first local minimum found.
                        False - keeps looking until alternative termination criteria
                        are reached.
    
    """
    try: 
        lb, ub = bounds
        ndim = len(lb)
        total_vol = (ub-lb).prod()
        
        centre = 0.5*(ub - lb) + lb 
        
        rect_dim = len(centre) if rect_dim == -1 else rect_dim
        assert rect_dim <= ndim
        
        log_min_l = np.log10(min_length)
    
        elite = hyperrectangle(centre, 
                               _func_wrapper(np.atleast_2d(centre).T, func, vectorizable, gen_inds(1,1,1), f_args)[0], 
                               lb, ub, np.inf, -1, log_min_l)
        parents = np.array([elite])
        archive, prev_bests = np.array([], dtype=hyperrectangle), np.array([], dtype=hyperrectangle)
        
        dims = np.arange(rect_dim, dtype=np.int64)
        conv_max = ndim // rect_dim + 1 
        i, fev, conv_count = 0, 1, 0
        interrupted = False
        while i < maxiter and fev < maxfev and conv_count < conv_max:
            it_start = clock()
            # split all hrects to be split from previous iteration
            new_hrects = np.array([hrect for parent in parents 
                                   for hrect in _divide_hrect(
                func, parent, dims, vectorizable, maxvectorwidth, f_args, log_min_l)])
    
            fev += len(new_hrects)
    
            # all hrects which do not have any children  
            childless = np.concatenate((new_hrects, archive))
    
            # generate array of list-index, cost, and volume
            fs = np.array([(j, h.f, h.volume) for j, h in enumerate(childless)], dtype=np.float64)
            # Make sure we haven't lost search hyperspace
            assert abs(1-sum(fs[:, 2])/total_vol) < 1e-6 # tolerance for floating point 
            
            # sort list indices by cost
            fs = fs[fs[:,1].argsort(), 0].astype(np.int64)
            # get list-indicies of the best {population} hrects by cost 
            best = np.array(fs[:min(population, len(fs))], dtype=np.int64)
    
            new_accepted = np.array([j for b in childless[best] for j, hrect in enumerate(childless) 
                                     if hrects_border(b, hrect)], dtype=np.int64)
            
            # combine new and archived hrects to be split next iteration
            new_accepted = np.unique(np.concatenate((best, new_accepted)))
    
            # get list-indices of childless hrects which are not to be split
            to_arch = np.setdiff1d(np.arange(len(childless)), new_accepted, assume_unique=True)
    
            # update archive
            archive = childless[to_arch]
            it_time = (clock() - it_start)/1000 #cpu-seconds
            
            best_ = childless[best]
            if i>0: 
                sames = [hrects_is_same(best_[j], prev_bests[j]) for j in range(len(prev_bests))]
                if sum(sames) == population: 
                    conv_count += 1
                else: 
                    conv_count = 0 
            if locally_biased is True:
                # Triggers termination if the best rectangles all stay the same for the full rotation of splitting axes
                prev_bests = childless[best]
            else: 
                local_minima = np.array([j for j in best if childless[j].barren is True], dtype=np.int64)
                new_accepted = np.setdiff1d(new_accepted, local_minima, assume_unique=True)
                total_vol -= sum([h.volume for h in childless[local_minima]])
            
            if childless[best[0]].f < elite.f: 
                # optimal value may not be found in the youngest generation
                elite = childless[best[0]]
            else: 
                elite.has_children=True
                
            # old parents are forgotten
            parents = childless[new_accepted]
            
            if disp is True:
                print(f'it {i}: #hrects = {len(parents)}. Took: {int(it_time//60)}:{round(it_time%60, 4)}')
            if callback is not None:
                callback(elite)
    
            # rotate splitting axes
            dims += rect_dim 
            dims %= ndim
            i+=1
    except KeyboardInterrupt:
        interrupted = True
        pass
    
    result = DirectResult(elite.x, elite.f, fev, i, interrupted, 
                          elite.lb, elite.ub, elite.volume)
    
    
    return np.asarray(elite.x), elite.f, 

class DirectResult:
    def __init__(self, x, f, nfev, nit, interrupted, volume, vratio):
        self.x = x
        self.f = f 
        self.nfev = nfev
        self.nit = nit
        self.interrupted = interrupted
        self.volume = volume
        self.vratio = vratio
        
