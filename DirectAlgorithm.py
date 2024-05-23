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
import warnings
from csv import writer


# =============================================================================
# njit compatible timer
import ctypes
import platform
if platform.system() == "Windows":
    from ctypes.util import find_msvcrt
    __LIB = find_msvcrt()
    if __LIB is None:
        __LIB = "msvcrt.dll"
    clock = ctypes.CDLL(__LIB).clock
    clock.argtypes = []
    @njit
    def cclock():
        return clock()/1000 #cpu-seconds

else:
    from ctypes.util import find_library
    __LIB = find_library("c")
    clock = ctypes.CDLL(__LIB).clock
    clock.argtypes = []
    @njit
    def cclock():
        return clock()/10_000 #cpu-seconds
# =============================================================================

spec = [
    ('centre', float64[:]),
    ('ndim', int64),
    ('f', float64),
    ('parent_f', float64),
    ('lb', float64[:]),
    ('ub', float64[:]),
    ('rdif', float64),
    ('adif', float64),
    ('generation', int64),
    ('cuts', int64),
    ('volume', float64),
    ('length_inds', float64[:]),
    ]

@jitclass(spec)
class hyperrectangle():
    def __init__(self, centre, f, generation, cuts, lb, ub, parent_f):
        self.centre = centre
        self.ndim = len(centre)

        self.f, self.parent_f = float(f), parent_f
        self.lb, self.ub = lb, ub
        # self.rdif, self.adif = self.f/self.parent_f, self.f-self.parent_f
        self.generation = generation
        self.cuts = cuts
        self.volume = (self.ub-self.lb).prod()
        self.length_inds = np.log10(self.ub-self.lb)

@njit
def hrect_is_semibarren(h, dims, log_min_l):
    return (h.length_inds < log_min_l)[dims].prod() == 1

@njit
def hrect_is_barren(h, log_min_l):
    return (h.length_inds < log_min_l).prod() == 1

@njit
def hrect_is_parent(lb1, ub1, lb2, ub2, tol=1e-10):
    """
    checks if a hyperrectangle is a (grand)parent of another
    h1 is prospective parent, h2 is prospective child
    
    Note: returns True is h1 is a (grand)parent of h2 OR if h1==h2
    """
    ub_bounded = (ub1 - ub2 >= -tol).prod() == 1
    lb_bounded = (lb1 - lb2 <= tol).prod() == 1
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
    # First test is redundant but filters out obivous non-borders fast 
    # the domains in each direction touch   
    if (((h2.ub - h1.lb) >= -tol) * 
        ((h1.ub - h2.lb) >= -tol)# directions where the domains of each h2 touch
        ).sum() != h1.ndim:
        return False

    # in ndim-1 directions the directions' domains overlap (either perfectly, or one inside another)
    overlap = (signs_of_array(h2.ub - h1.ub, tol) ==
                signs_of_array(h1.lb - h2.lb, tol))
    if overlap.sum() != h1.ndim-1:
        return False

    # adjacent (ub=lb or lb=ub) (higher OR lower) in exactly one dimension
    adjacency = ((np.abs(h2.ub - h1.lb) < tol) +
                 (np.abs(h2.lb - h1.ub) < tol)) 
    if adjacency.sum() != 1:
        return False

    # Direction of adjacency is the direction not overlapping
    if (adjacency == ~overlap).prod() != 1:
        return False
    return True

@njit
def signs_of_array(arr, tol=1e-10):
    arr[np.abs(arr)<tol] = 0 
    return np.sign(arr)

@njit #parallel is slower for ndim range
def _generate_boolmatrix(ndim):
    _2ndim = 2**ndim
    z = np.empty((_2ndim, ndim), dtype=np.bool_)
    for i in prange(ndim):
        base = np.zeros(_2ndim, dtype=np.bool_)
        i2 = 2**i
        for j in range(0, _2ndim, i2*2):
            base[j:j+i2] = True
        z[:,i] = base
    return z

@njit #parallel is slower for ndim range
def _generate_bounds(hrect, indcs, dims):
    lc = np.hstack((np.atleast_2d(hrect.lb).T, np.atleast_2d(hrect.centre).T))[dims,:]
    cu = np.hstack((np.atleast_2d(hrect.centre).T, np.atleast_2d(hrect.ub).T))[dims,:]
    
    lbs = np.repeat(hrect.lb, len(indcs)).reshape((len(hrect.lb), len(indcs))).T
    ubs = np.repeat(hrect.ub, len(indcs)).reshape((len(hrect.ub), len(indcs))).T
    
    for i in prange(len(indcs)):
        base = np.empty((len(dims), 2), dtype=np.float64)
        ind = indcs[i,:]
        inv = ~ind
        base[ind,:], base[inv,:] = lc[ind,:], cu[inv,:]
        lbs[i, dims], ubs[i, dims] = base[:,0], base[:,1]
        
    return lbs, ubs

@njit #parallel is slower for ndim range
def _generate_centres(hrect, indcs, dims):
    l = (hrect.lb + hrect.centre)[dims]/2
    u = (hrect.centre + hrect.ub)[dims]/2
    
    centres = np.repeat(hrect.centre, len(indcs)).reshape((len(hrect.centre), len(indcs))).T
    for i in prange(len(indcs)):
        base = np.empty(len(dims), dtype=np.float64)
        ind = indcs[i,:]
        inv = ~ind
        base[ind], base[inv] = l[ind], u[inv]
        centres[i, dims] = base
    return centres


@njit
def _divide_vec(func, hrect, dims, f_args, log_min_l):
    # do not split along resolution axes
    dims = dims[hrect.length_inds >= log_min_l]
    l_dim=len(dims)
    if l_dim == 0:
        # do not lose hrect - may be splittable along different axis
        return [hrect] 
    n_new = 2**l_dim
    indcs = _generate_boolmatrix(l_dim)

    centres = _generate_centres(hrect, indcs, dims)
    lbs, ubs = _generate_bounds(hrect, indcs, dims)
    pf, gen, cuts = hrect.f, hrect.generation + 1, hrect.cuts + l_dim
    
    f_values = func(centres.T, *f_args)
        
    hrects = [hyperrectangle(
        centres[k], f_values[k], gen, cuts, lbs[k], ubs[k], pf) 
        for k in range(n_new)]

    return hrects

@njit(parallel=True)
def _divide_mp(func, hrect, dims, f_args, log_min_l):
    # do not split along resolution axes
    dims = dims[hrect.length_inds >= log_min_l]
    l_dim=len(dims)
    if l_dim == 0:
        # do not lose hrect 
        return [hrect] 
    n_new = 2**l_dim
    indcs = _generate_boolmatrix(l_dim)

    centres = _generate_centres(hrect, indcs, dims)
    lbs, ubs = _generate_bounds(hrect, indcs, dims)
    pf, gen, cuts = hrect.f, hrect.generation + 1, hrect.cuts + l_dim
    
    f_values = np.empty(n_new, dtype=np.float64)
    for i in prange(n_new):
        f_values[i] = func(centres[i,:], *f_args)
    
    hrects = [hyperrectangle(
        centres[k], f_values[k], gen, cuts, lbs[k], ubs[k], pf) 
        for k in range(n_new)]
    return hrects

def Direct(
    func, 
    bounds,
    vectorizable=False,
    callback=None,
    printfile='',
    restart='',
    disp=False,
    program=None,
    
    f_args=(),
    maxiter=np.inf,
    maxfev=np.inf,
    rect_dim=-1,
    population=1,
    resolution=-np.inf,
    locally_biased=False,
    near_optimal=np.inf,
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
    vectorizable    - boolean. Whether multiple candidate arrays can be passed to 
                        to the objective function at once. 
    callback        - callable. The elite hyperrectangle will be passed to it 
                        after each iteration
    restart         - If you have logged all of the points evaluated (callback mode 2
                        in this FIRM implementation). You can read in that file and restart 
                        the optimisation from where the previous one ended. 
                    - Give the file name containing the points already evaluated. 
    program         - A tuple of dictionaries with keyword arguments for DIRECT
                    - Allows changing of optimisation paramters at prespecified
                        points without needing to stop and restart
                    - Note, arguments passed through program will overwrite arguments
                        passed to the optimiser. Arguments will not be reset except 
                        by passing them to the next step of the program
    disp            - boolean. Prints out information at each iteration.
    -------------------------
    The following arguments can be overwritten by program
    -------------------------
    f_args          - additional arguments to be supplied to the objective function
    maxiter         - maximum iterations to perform (default: inf)
    maxfev          - maximum number of function evaluations to perform 
    rect_dim        - number of directions to split rectangles on at once. This 
                        keeps the number of rectangles in each iteration manageable. 
                        Optimizer will cycle through the directions automatically.
                    - It is recommended to choose a rect_dim such that 2**rect_dim 
                        is below but as close as possible to maximum vector width.
    population      - number of local optima to pursue at once. The optimizer will 
                        split the {population} best rectangles, and any adjacent rectangles.
                    - higher populations will give slower convergence but a more 
                        global search.
    resolution      - maximum resolution of the solution. Should be number or array 
                        broadcastable to the solution vector. Default: -inf
    locally_biased  - boolean. 
                        True - converges at first local minimum found.
                        False - keeps looking until alternative termination criteria
                        are reached.
    near_optimal    - float. The optimiser will only split rectangles which are within
                        {near_optimal*(best found value)} or adjacent. There being no more
                        rectangles satisfying this criteria will be a termination criterion.
    """
    
    # if program is not None:
    #     warnings.warn(
    #         'A program was passed. Parameters of the program may override '
    #         +'other arguments passed. ', UserWarning)
        
    lb, ub = bounds
    ndim = len(lb)
    centre = 0.5*(ub - lb) + lb 
    MAXPARENTS = 700_000 #used to prevent memory errors 
    
    if restart != '': 
        archive, elite = _restart(restart, bounds, disp)
        parents, prev_bests = np.array([], dtype=hyperrectangle), np.array([], dtype=hyperrectangle)
        archive = np.array(archive)
        _divide_hrect = _divide_vec if vectorizable is True else _divide_mp
    else: 
        if vectorizable: 
            f = func(np.atleast_2d(centre).T, *f_args)[0]
            _divide_hrect = _divide_vec
        else: 
            f = func(centre, *f_args)
            _divide_hrect = _divide_mp
            
        if printfile!='':
            with open(printfile+'-parents.csv', 'w', newline='') as csvfile:
                writer(csvfile)
            with open(printfile+'-children.csv', 'w') as csvfile:
                writer(csvfile)
            with open(printfile+'-minima.csv', 'w') as csvfile:
                writer(csvfile)
            
        elite = hyperrectangle(centre, f, -1, 0, lb, ub, np.inf)
        parents = np.array([elite])
        archive, prev_bests = np.array([], dtype=hyperrectangle), np.array([], dtype=hyperrectangle)
    
    i, conv_count, miter_adj, mfev_adj = 0, 0, 0, 0
    fev = 1
    
    program = ({},) if program is None else program
    for step in program:
        keys = step.keys()
        f_args = step['f_args'] if 'f_args' in keys else f_args
        maxiter = step['maxiter'] if 'maxiter' in keys else maxiter
        maxfev = step['maxfev'] if 'maxfev' in keys else maxfev
        rect_dim = step['rect_dim'] if 'rect_dim' in keys else rect_dim
        population = step['population'] if 'population' in keys else population
        resolution = step['resolution'] if 'resolution' in keys else resolution
        locally_biased = step['locally_biased'] if 'locally_biased' in keys else locally_biased
        near_optimal = step['near_optimal'] if 'near_optimal' in keys else near_optimal

        rect_dim = len(lb) if rect_dim==-1 else rect_dim
        assert rect_dim <= ndim
        dims = np.arange(rect_dim, dtype=np.int64)
        conv_max = ndim // rect_dim + min(1, ndim%rect_dim)

        log_min_l = np.log10(resolution)
        total_vol = (ub-lb).prod()
        local_minima = np.array([], dtype=hyperrectangle)
        # near_optimal_res = True
        conv_count = 0
        
        while (i < maxiter+miter_adj # stop at max iterations
               and fev < maxfev+mfev_adj  # stop at max function evaluations
               and conv_count < conv_max # stop when no improvement to best (locally biased)
               and total_vol > 0 # stop when resolution fully resolved
               # and near_optimal_res # no more near_optimal space within resolution
            ): 

            it_start = dt.datetime.now()
            
            # split all hrects to be split from previous iteration
            new_hrects = np.array([hrect for parent in 
                                   tqdm(parents, desc=f'it {i} - #hrects: {len(parents)}. Evaluating Rectangles', leave=False)
                                   for hrect in _divide_hrect(func, parent, dims, f_args, log_min_l)])
            fev += len(new_hrects)
    
            # all hrects which do not have any children  
            childless = np.concatenate((new_hrects, archive))
    
            # generate array of list-index, cost, and volume
            fvs = np.array([(j, h.f, h.volume) for j, h in enumerate(childless)], dtype=np.float64)
            # Make sure we haven't lost search space
            assert abs(1-sum(fvs[:, 2])/total_vol) < 1e-6, f'{sum(fvs[:, 2])} / {total_vol}' # tolerance for floating point 
            
            # sort list indices by cost
            fvs = fvs[fvs[:,1].argsort(), :]
            fs = fvs[:,0].astype(np.int64)
            
            if childless[fs[0]].f < elite.f: 
                # optimal value may not be found in the youngest generation
                elite = childless[fs[0]]

            # near-optimal rectangles
            nearoptimalmask = fvs[:, 1] <= near_optimal*elite.f
            best = fs[nearoptimalmask] 
            best = best[:min(population, len(best))]
            
            if locally_biased is True:
                # Triggers termination if the best rectangles all stay the same for the full rotation of splitting axes
                prev_bests = childless[best]
                new_accepted = np.array([j for b in childless[best] for j, hrect in enumerate(childless) 
                                         if hrects_border(b, hrect) 
                                         and not hrect_is_barren(hrect, log_min_l)], dtype=np.int64)
                # combine new and archived hrects to be split next iteration
                new_accepted = np.unique(np.concatenate((best, new_accepted)))
                # get list-indices of childless hrects which are not to be split
                to_arch = np.setdiff1d(np.arange(len(childless)), new_accepted, assume_unique=True)
                
                if i>0: 
                    sames = [hrects_is_same(childless[best][j], prev_bests[j]) for j in range(len(prev_bests))]
                    if (sum(sames) == population):
                        conv_count += 1
                    else: 
                        conv_count = 0 
                
                # rotate splitting axes
                dims += rect_dim 
                dims %= ndim
            else: 
                lm_i = np.array([j for j in fs if hrect_is_barren(childless[j], log_min_l)], dtype=np.int64)
                total_vol -= sum([h.volume for h in childless[lm_i]])
                local_minima = np.concatenate((local_minima, np.array(childless[lm_i])))

                # rotate splitting axes
                dims += rect_dim 
                dims %= ndim
                
                best = fs[nearoptimalmask]
                new_accepted = np.array([], dtype=np.int64)
                if len(best)>0:
                    # list indices of best rectangles which are semibarren
                    best_semibarr = _semibarren_speedup(list(childless[best]), dims, log_min_l)
                    
                    best = np.setdiff1d(best, best[best_semibarr], assume_unique=True)
                    best = best[:min(len(best), population)]
                
                    new_accepted = best.copy()
                    
                # get list-indices of childless hrects which are not to be split
                to_arch = np.setdiff1d(np.arange(len(childless)), new_accepted, assume_unique=True)
                
                # remove local minima
                to_arch = np.setdiff1d(to_arch, lm_i, assume_unique=True)

            if len(new_accepted) == 0: 
                conv_count+=1
            else: 
                conv_count=0
            
            if len(new_accepted) == 0 and conv_count >= conv_max:
                new_accepted=np.array([], dtype=np.int64)
                near_optimal_threshold = elite.f*near_optimal
                near_optimal_minima = np.array([h for h in local_minima if h.f < near_optimal_threshold])
                eligible = np.setdiff1d(np.arange(len(childless), dtype=np.int64),
                                        lm_i, 
                                        assume_unique=True)
                eligible = np.setdiff1d(eligible, 
                                        eligible[_borderheuristic(list(childless[eligible]), list(near_optimal_minima))],
                                        assume_unique=True)
                eligible = np.setdiff1d(eligible, _semibarren_speedup(list(childless[eligible]), dims, log_min_l), assume_unique=True)
                
                new_accepted = sortrectangles(list(near_optimal_minima), list(childless), eligible)

                to_arch = np.setdiff1d(np.arange(len(childless), dtype=np.int64), new_accepted, assume_unique=True)
                to_arch = np.setdiff1d(to_arch, lm_i, assume_unique=True)
            if len(new_accepted) != 0:
                conv_count=0

            if printfile != '':
                with open(printfile+'-parents.csv', 'a', newline='') as csvfile:
                    if len(parents) > 0:
                        printout = np.concatenate((np.array([(h.f, h.generation, h.cuts) for h in parents]), 
                                                   np.array([h.centre for h in parents])), 
                                                   axis=1)
                        writer(csvfile).writerows(printout)
                with open(printfile+'-children.csv', 'w', newline='') as csvfile:
                    if len(childless) > 0:
                        printout = np.concatenate((np.array([(h.f, h.generation, h.cuts) for h in childless]), 
                                                   np.array([h.centre for h in childless])), 
                                                   axis=1)
                        writer(csvfile).writerows(printout)
                with open(printfile+'-minima.csv', 'w', newline='') as csvfile:
                    if len(local_minima) > 0:
                        printout = np.concatenate((np.array([(h.f, h.generation, h.cuts) for h in local_minima]), 
                                                   np.array([h.centre for h in local_minima])), 
                                                   axis=1)
                        writer(csvfile).writerows(printout)

            if len(new_accepted) > MAXPARENTS: # prevents memory error
                to_arch = np.concatenate((to_arch, new_accepted[MAXPARENTS:]))
                new_accepted = new_accepted[:MAXPARENTS]

            it_time = dt.datetime.now() - it_start
            if disp is True: 
                print(f'it {i} - #hrects: {len(parents)}. Took: {it_time}. Best value: {elite.f}.')
            if callback is not None:
                callback(elite)

            # update archive
            archive = childless[to_arch]
            # old parents are forgotten
            parents = childless[new_accepted]
            
            i+=1
        archive = np.concatenate((archive, local_minima))
        if printfile != '':
            with open(printfile+'-minima.csv', 'w') as csvfile:
                writer(csvfile)
            with open(printfile+'-children.csv', 'w', newline='') as csvfile:
                if len(archive) > 0:
                    printout = np.concatenate((np.array([(h.f, h.generation, h.cuts) for h in archive]), 
                                               np.array([h.centre for h in archive])), 
                                               axis=1)
                    writer(csvfile).writerows(printout)
        
        miter_adj += i
        mfev_adj += fev
        print('\nprogram step\n')

    print('\n')
    return DirectResult(elite.centre, elite.f, fev, i, 
                          elite.lb, elite.ub, elite.volume, elite.volume/total_vol)                  


@njit(parallel=True)
def _semibarren_speedup(rects, dims, log_min_l):
    accepted = np.empty(len(rects), dtype=np.bool_)
    for i in prange(len(rects)):
        accepted[i] = hrect_is_semibarren(rects[i], dims, log_min_l)
    return np.arange(len(rects), dtype=np.int64)[accepted]

@njit(parallel=True)
def sortrectangles(minima, archive, eligible):
    accepted=np.zeros(len(eligible), dtype=np.bool_)

    if len(eligible) <= 10000:
        time_test_range = 0
    else: 
        time_test_range = len(eligible//100)
        start = cclock()
        for i in prange(time_test_range):
            for m in minima:
                if hrects_border(archive[eligible[i]], m):
                    accepted[i] = True
                    break
        ttime = cclock() - start
        rtime = (len(eligible) - time_test_range)*ttime/time_test_range
        rh, rm, rs = int(rtime//3600), int(rtime%3600//60), int(rtime%3600%60)
        print(f'Identifying near-optimal neighbours. Estimated time: {rh}:{rm}:{rs}')

    for i in prange(time_test_range, len(eligible)):
        for m in minima:
            if hrects_border(archive[eligible[i]], m):
                accepted[i] = True
                break

    return eligible[accepted]
        

@njit(parallel=True)
def _borderheuristic(rects, best):
    minlb =  np.inf*np.ones(best[0].ndim, dtype=np.float64)
    maxub = -np.inf*np.ones(best[0].ndim, dtype=np.float64)
    
    for i in range(len(best)): 
        minlb = np.minimum(minlb, best[i].lb)
        maxub = np.maximum(maxub, best[i].ub)
    
    accepted = np.empty(len(rects), dtype=np.bool_)
    for i in prange(len(rects)):
        accepted[i] = ((rects[i].lb >= maxub).sum() + (rects[i].ub <= minlb).sum() > 1)

    return np.arange(len(rects), dtype=np.int64)[accepted]
    
@njit
def _normalise(arr, lb, ub):
    return (arr-lb)/(ub-lb)

@njit
def _unnormalise(arr, lb, ub):
    return arr*(ub-lb) + lb

@njit(parallel=True)
def _reconstruct_from_centre(centres, bounds, maxres=2**31):
    lb, ub = bounds
    centres = _normalise(centres, lb, ub)
    incs = np.round((centres*maxres)).astype(np.uint64)
    incs1 = np.empty_like(incs)
    for i in prange(len(centres)):
        for j in range(centres.shape[1]):
            incs1[i,j] = _factor2(incs[i,j])
    lbs = (incs-incs1)/maxres
    ubs = (incs+incs1)/maxres
    
    centres, lbs, ubs = [_unnormalise(arr, lb, ub) for arr in (centres, lbs, ubs)]
    return centres, lbs, ubs
    

def _restart(restart, bounds, disp):
    print('Restarting optimisation where',restart,'left off.')
    history = np.genfromtxt(restart+'-children.csv', delimiter=',', dtype=np.float64)
    try: 
        minima = np.genfromtxt(restart+'-minima.csv', delimiter=',', dtype=np.float64)
        if len(minima) != 0:
            history = np.vstack((history, np.atleast_2d(minima)))
    except FileNotFoundError:
        pass
    try: 
        parents = np.genfromtxt(restart+'-parents.csv', delimiter=',', dtype=np.float64)
        pmin, pminidx = parents[:,0].min(), parents[:,0].argmin()
    except FileNotFoundError:
        pmin, pminidx= np.inf, None
        warnings.warn("Warning: No parents file found.", UserWarning)
    fs, xs = history[:,:3], history[:,3:]
    
    xs, lbs, ubs = _reconstruct_from_centre(xs, bounds)

    if fs[:,0].min() < pmin:
        elite = fs[:,0].argmin()
        elite = hyperrectangle(xs[elite], *fs[elite], lbs[elite], ubs[elite], np.nan)
    else: 
        fps, xps = parents[:,:3], parents[:,3:]
        xps, lbps, ubps = _reconstruct_from_centre(np.atleast_2d(xps[pminidx, :]), bounds)
        elite = hyperrectangle(xps[0,:], *fps[pminidx,:], lbps[0,:], ubs[0,:], np.nan)
        del fps, xps, lbps, ubps

    archive = np.array([hyperrectangle(xs[i], *fs[i,:], lbs[i], ubs[i], np.nan) for i in range(len(xs))])
    
# =============================================================================
# Child-wise loop is faster 
# Iterates backwards through list of rectangles
# For each rectangle, directly removes any parents it finds (searching backwards)
    # _child_loop(xs[:1,:], fs[:1,:], lbs[:1, :], ubs[:1, :]) # compile jit
    # archive = _child_loop(xs, fs, lbs, ubs)
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
        print(f'Restart: read in {len(xs)} rectangles. Discarded {len(xs)-len(archive)} parents.',
              f'Best value: {elite.f}.')
  
    return archive, elite


@njit(parallel=True)
def _child_loop(xs, fs, lbs, ubs):
    start = cclock()
    archive = [hyperrectangle(xs[i], fs[i,0], fs[i,1], fs[i,2], lbs[i], ubs[i], np.nan) for i in range(len(xs))]
    expected_evals = (len(xs)-1)*(len(xs))/2 #no parents found ever 
    
    for i in range(len(xs)-1, -1, -1):
        if (len(xs)-i)%1000 == 0 and i != 0: 
            itt = (cclock()-start)
            h, m, s = int(itt//3600), int(round((itt%3600)//60)), int(round(itt%60))
            evals_remaining = (i**2)/2
            evals_done = expected_evals - evals_remaining
            itt = itt * evals_remaining / evals_done
            rh, rm, rs = int(itt//3600), int(round((itt%3600)//60)), int(round(itt%60))
            print(f'{len(xs)-i}/{len(archive)}. Time: {h}:{m}:{s}. Remaining: < {rh}:{rm}:{rs}')
        try: 
            h1 = archive[i]
        except: 
            continue
        to_del = []
        for j in prange(len(xs)-i, len(xs)):
            j = len(xs)-(j+1)
            if hrect_is_parent(archive[j].lb, archive[j].ub, h1.lb, h1.ub):
                to_del.append(j)
        to_del.sort(reverse=True)
        for j in to_del:
            del archive[j]
    return archive
    
@njit
def _parent_loop(xs, fs, lbs, ubs):
    start = cclock()
    archive = [hyperrectangle(xs[i], fs[i,0], fs[i,1], fs[i,2], lbs[i], ubs[i], np.nan) for i in range(len(fs))]
    parents_i = []
    for i in range(len(archive)):
        if i%10000 == 0 and i != 0: 
            itt = (cclock()-start)
            h, m, s = int(itt//3600), int(round(itt%3600)), int(round(itt%60))
            itt = itt * (len(fs)-i) / i
            rh, rm, rs = int(itt//3600), int(round(itt%3600)), int(round(itt%60))
            print(f'{i}/{len(fs)}. Time: {h}:{m}:{s}. Remaining: < {rh}:{rm}:{rs}')
        h1 = archive[i]
        for j in prange(i+1, len(archive)):
            if hrect_is_parent(h1.lb, h1.ub, archive[j].lb, archive[j].ub):
                parents_i.append(i)
                break
    return parents_i

@njit
def _factor2(n):
    if n==0: 
        return 0
    i=0
    while n%(2**(i+1)) != 2**i:
        i+=1
    return 2**i
    
class DirectResult:
    def __init__(self, x, f, nfev, nit, lb, ub, volume, vratio):
        self.x = x
        self.f = f 
        self.nfev = nfev
        self.nit = nit
        self.lb, self.ub = lb, ub
        self.volume = volume
        self.vratio = vratio
        
