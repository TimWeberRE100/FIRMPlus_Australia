# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 08:28:16 2024

@author: u6942852
"""

import numpy as np 
from numba import jit, njit, prange, float64, int64, boolean
from numba.experimental import jitclass
import datetime as dt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# =============================================================================
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
# =============================================================================
@njit
def hrect_is_semibarren(h1, dims, log_min_l):
    return (h1.length_inds < log_min_l)[dims].prod() == 1

@njit
def hrect_is_parent(h1, h2, tol=1e-10):
    """
    checks if a hyperrectangle is a (grand)parent of another
    h1 is prospective parent, h2 is prospective child
    
    Note: returns True is h1 is a (grand)parent of h2 OR if h1==h2
    """
    ub_bounded = (h1.ub - h2.ub >= -tol).prod() == 1
    lb_bounded = (h1.lb - h2.lb <= tol).prod() == 1
    return ub_bounded and lb_bounded


@njit
def hrects_is_same(h1, h2, tol=1e-10):
    ub_same = (h1.ub == h2.ub).prod() == 1 
    lb_same = (h1.lb == h2.lb).prod() == 1 
    c_same = (h1.centre == h2.centre).prod() == 1
    f_same = h1.f == h2.f
    return ub_same and lb_same and c_same and f_same

@njit
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

@njit
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
        # self.rdif, self.adif = self.f/self.parent_f, self.f-self.parent_f
        self.n = parent_n + 1
        self.volume = (self.ub-self.lb).prod()
        self.length_inds = np.log10(self.ub-self.lb)
        self.barren = (self.length_inds < log_min_l).prod()
        self.has_children=False

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

@njit
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

@njit
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

def _mp_funcwrapper(x, func, f_args):
    return [func(xn, *f_args) for xn in x]


# @jit(nopython=False)
def _divide_hrect(vectorizable, workers, func, hrect, dims, f_args, log_min_l):
    min_length_reached = hrect.length_inds < log_min_l
    # do not split along min_length axes
    dims = np.array([i for i in dims if not min_length_reached[i]])
    if len(dims) == 0:
        # do not lose hrect - may be splittable along different axis
        return [hrect] 
    
    indcs = gen_boolmatrix(len(dims))
    n_new = 2**len(dims)

    centres = _generate_centres(hrect, indcs, dims)
    lbs, ubs = _generate_bounds(hrect, indcs, dims)
    pf, pn = hrect.f, hrect.n
    
    if vectorizable is True: 
        f_values = func(centres.T, *f_args)
        hrects = [hyperrectangle(
            centres[k], f_values[k], lbs[k], ubs[k], pf, pn, log_min_l) 
            for k in range(n_new)]
                
    if workers > 1: 
        with Pool(processes=max(workers, n_new)) as processPool:
            args = [(centres[n], _mp_funcwrapper, *f_args) for n in range(n_new)]
            _result = processPool.starmap(_mp_funcwrapper, args)
        f_values = np.array(_result)
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
    vectorizable = False,
    workers = 1, 
    rect_dim = -1,
    population = 1,
    min_length = -np.inf,
    disp = False,
    locally_biased=False,
    restart='',
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
    rect_dim        - number of directions to split rectangles on at once. This 
                        keeps the number of rectangles in each iteration manageable. 
                        Optimizer will cycle through the directions automatically.
                    - It is recommended to choose a rect_dim such that 2**rect_dim 
                        is below but as close as possible to maximum vector width.
    population      - number of local optima to pursue at once. The optimizer will 
                        split the {population} best rectangles, and any adjacent rectangles.
                    - higher populations will give slower convergence but a more 
                        global search.
    min_length      - maximum resolution of the solution. Should be number or array 
                        broadcastable to the solution vector. Default: -inf
    disp            - boolean. Prints out at each iteration.
    locally_biased  - boolean. True - converges at first local minimum found.
                        False - keeps looking until alternative termination criteria
                        are reached.
    restart         - If you have logged all of the points evaluated (callback mode 2
                        in this FIRM implementation). You can read in that file and restart 
                        the optimisation from where the previous one ended. 
                    - Give the file name containing the points already evaluated. 
    """
        
    try: 
        lb, ub = bounds
        ndim = len(lb)
        total_vol = (ub-lb).prod()
        log_min_l = np.log10(min_length)
        
        centre = 0.5*(ub - lb) + lb 
        
        rect_dim = len(centre) if rect_dim == -1 else rect_dim
        assert rect_dim <= ndim
        workers = cpu_count() if workers == -1 else workers
        assert workers <= cpu_count()
        
        if restart != '': 
            print('Restarting optimisation where',restart,'left off.')
            archive, elite, total_vol = _restart(restart, bounds, log_min_l, disp)
            parents, prev_bests = np.array([], dtype=hyperrectangle), np.array([], dtype=hyperrectangle)

            barren_i = np.array([(j, h.volume) for j, h in enumerate(archive) if h.barren])
            if len(barren_i) > 0:
                total_vol -= barren_i[:,1].sum()
                archive = np.array(archive)[
                    np.setdiff1d(np.arange(len(archive)), barren_i[:,0].astype(int), assume_unique=True)]
            else: 
                archive = np.array(archive)
        else: 
            if vectorizable: 
                elite = hyperrectangle(centre, func(np.atleast_2d(centre).T, *f_args)[0], 
                                       lb, ub, np.inf, -1, log_min_l)
            else: 
                elite = hyperrectangle(centre, func(centre, *f_args), 
                                       lb, ub, np.inf, -1, log_min_l)
            parents = np.array([elite])
            archive, prev_bests = np.array([], dtype=hyperrectangle), np.array([], dtype=hyperrectangle)
        
        dims = np.arange(rect_dim, dtype=np.int64)
        conv_max = ndim // rect_dim + 1 

        i, fev, conv_count, interrupted = 0, 1, 0, False
        while i < maxiter and fev < maxfev and conv_count < conv_max:
            it_start = clock()
            # split all hrects to be split from previous iteration
            new_hrects = np.array([hrect for parent in parents 
                                   for hrect in _divide_hrect(
                vectorizable, workers, func, parent, dims, f_args, log_min_l)])
            
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
    
            best_ = childless[best]
            if i>0: 
                sames = [hrects_is_same(best_[j], prev_bests[j]) for j in range(len(prev_bests))]
                if sum(sames) == population: 
                    conv_count += 1
                else: 
                    conv_count = 0 
            
            if childless[best[0]].f < elite.f: 
                # optimal value may not be found in the youngest generation
                elite = childless[best[0]]
            else: 
                elite.has_children=True

            if locally_biased is True:
                # Triggers termination if the best rectangles all stay the same for the full rotation of splitting axes
                prev_bests = childless[best]
                new_accepted = np.array([j for b in childless[best] for j, hrect in enumerate(childless) 
                                         if hrects_border(b, hrect) and not hrect.barren], dtype=np.int64)
            else: 
                local_minima = np.array([j for j in best if childless[j].barren is True], dtype=np.int64)
                
                
                new_accepted = np.array([j for b in childless[best] for j, hrect in enumerate(childless) 
                                         if hrects_border(b, hrect) and not hrect.barren], dtype=np.int64)
                # rotate splitting axes
                dims += rect_dim 
                dims %= ndim
                
                best_ = []
                for f_i in fs:
                    if not hrect_is_semibarren(childless[f_i], dims, log_min_l):
                        best_.append(f_i)
                    if len(best_) >= min(population, len(fs)):
                        break
                best = np.array(best_, dtype=np.int64)
                new_accepted = np.array([j for b in childless[best] for j, hrect in enumerate(childless) 
                                         if hrects_border(b, hrect) 
                                         and not hrect_is_semibarren(hrect, dims, log_min_l)], dtype=np.int64)

                # forget local minima - (will be remebered by elite if elite)
                new_accepted = np.setdiff1d(new_accepted, local_minima, assume_unique=True)
                total_vol -= sum([h.volume for h in childless[local_minima]])
                
            # combine new and archived hrects to be split next iteration
            new_accepted = np.unique(np.concatenate((best, new_accepted)))
    
            # get list-indices of childless hrects which are not to be split
            to_arch = np.setdiff1d(np.arange(len(childless)), new_accepted, assume_unique=True)
    
            # update archive
            archive = childless[to_arch]
                
                
            # old parents are forgotten
            parents = childless[new_accepted]
            
            it_time = (clock() - it_start)/1000 #cpu-seconds
            if disp is True:
                print(f'it {i}: #hrects = {len(parents)}. Took: {int(it_time//60)}:{round(it_time%60, 4)}.',
                      f'Best value: {elite.f}.')
            if callback is not None:
                callback(elite)
    
            i+=1
    except KeyboardInterrupt:
        interrupted = True
        pass
    
    return DirectResult(elite.centre, elite.f, fev, i, interrupted, 
                          elite.lb, elite.ub, elite.volume, elite.volume/total_vol)

def _restart(restart, bounds, log_min_l, disp):
    history = np.genfromtxt(restart, delimiter=',', dtype=np.float64)
    fs, xs = history[:,0], history[:,1:]
    
    lb, ub = bounds
    total_vol = (ub-lb).prod()
    
    xs = (xs - lb)/(ub-lb)
    
    maxres = 2**31
    incs = (xs * maxres).round().astype(np.uint64)
    
    incs1 = np.empty_like(incs)
    for i in prange(incs1.shape[0]):
        for j in prange(incs1.shape[1]):
            incs1[i,j] = _factor2(incs[i,j])

    lbs, ubs = (incs - incs1)/maxres, (incs + incs1)/maxres    
    xs, lbs, ubs = (arr*(ub-lb)+lb for arr in (xs, lbs, ubs))

    elite = fs.argmin()
    elite = hyperrectangle(xs[elite], fs[elite], lbs[elite], ubs[elite], np.nan, np.nan, log_min_l)
    
# =============================================================================
# Child-wise loop is faster 
# Iterates backwards through list of rectangles
# For each rectangle, directly removes any parents it finds (searching backwards)
    _child_loop(xs[:1,:], fs[:1], lbs[:1, :], ubs[:1, :], log_min_l) # compile jit
    archive = _child_loop(xs, fs, lbs, ubs, log_min_l)
# =============================================================================

# =============================================================================
# Parent-wise loop is slower
# Iterates forwards through list of rectangles
# Adds index of parents to a list which is then ussed to remove parents later

#     _parent_loop(xs[:1,:], fs[:1], lbs[:1, :], ubs[:1, :], log_min_l) #compile jit
#     parents_i, _vol=_parent_loop(xs, fs, lbs, ubs, log_min_l)
#     total_vol -= _vol
#     archive = np.array([hyperrectangle(
#         xs[k], fs[k], lbs[k], ubs[k], np.nan, np.nan, log_min_l)
#         for k in range(len(fs))])
#     
#     parents_i = np.array(parents_i)
#     childless_i = np.setdiff1d(np.arange(len(archive)), parents_i)
#     forget parents
#     archive = archive[childless_i]
#     fs = fs[childless_i]
# =============================================================================
    
    if disp is True:
        print(f'\nRestart: read in {len(xs)} rectangles. Discarded {len(xs)-len(archive)} parents.',
              f'Best value: {elite.f}.')
  
    return archive, elite, total_vol

@njit
def _child_loop(xs, fs, lbs, ubs, log_min_l):
    start = clock()
    archive = [hyperrectangle(xs[i], fs[i], lbs[i], ubs[i], np.nan, np.nan, log_min_l) for i in range(len(fs))]
    for i in range(len(archive)-1, -1, -1):
        if (len(fs)-i)%1000 == 0 and i != 0: 
            itt = (clock()-start)/1000
            h, m, s = int(itt//3600), int(round((itt%3600)//60)), int(round(itt%60))
            itt = itt * i / (len(archive)-i+1)
            rh, rm, rs = int(itt//3600), int(round((itt%3600)//60)), int(round(itt%60))
            print(f'{len(fs)-i}/{len(archive)}. Time: {h}:{m}:{s}. Remaining {rh}:{rm}:{rs}')
        try: 
            h1 = archive[i]
        except: 
            continue
        for j in range(i-1, -1, -1):
            if hrect_is_parent(archive[j], h1):
                del archive[j]
    return archive
         
@njit
def _parent_loop(xs, fs, lbs, ubs, log_min_l):
    start = clock()
    archive = [hyperrectangle(xs[i], fs[i], lbs[i], ubs[i], np.nan, np.nan, log_min_l) for i in range(len(fs))]
    parents_i, _vol = [], 0
    for i in prange(len(archive)):
        if i%10000 == 0 and i != 0: 
            itt = (clock()-start)/1000
            h, m, s = int(itt//3600), int(round(itt%3600)), int(round(itt%60))
            itt = itt * (len(fs)-i) / i
            rh, rm, rs = int(itt//3600), int(round(itt%3600)), int(round(itt%60))
            print(f'{i}/{len(fs)}. Time: {h}:{m}:{s}. Remaining {rh}:{rm}:{rs}')
        h1 = archive[i]
        if h1.barren:
            _vol -= h1.volume
        for j in prange(i+1, len(archive)):
            if hrect_is_parent(h1, archive[j]):
                parents_i.append(i)
                break
    return parents_i, _vol

@njit
def _factor2(n):
    if n==0: 
        return 0
    i=0
    while n%(2**(i+1)) != 2**i:
        i+=1
    return 2**i
    
class DirectResult:
    def __init__(self, x, f, nfev, nit, interrupted, lb, ub, volume, vratio):
        self.x = x
        self.f = f 
        self.nfev = nfev
        self.nit = nit
        self.interrupted = interrupted
        self.lb, self.ub = lb, ub
        self.volume = volume
        self.vratio = vratio
        
