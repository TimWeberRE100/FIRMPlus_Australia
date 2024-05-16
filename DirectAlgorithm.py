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
        return clock()/1000/1000 #cpu-seconds
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
def hrect_is_barren(length_inds, log_min_l):
    return (length_inds < log_min_l).prod() == 1

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

# @njit
def hrects_is_same(h1, h2, tol=1e-10):
    ub_same = (h1.ub == h2.ub).prod() == 1 
    lb_same = (h1.lb == h2.lb).prod() == 1 
    c_same = (h1.centre == h2.centre).prod() == 1
    f_same = h1.f == h2.f
    return ub_same and lb_same and c_same and f_same

@njit
def hrects_border(h1, h2, tol = 1e-10):
    
    assert h1.ndim == h2.ndim 
    # directions where the domains of each h2 touch
    touch = (((h2.ub - h1.lb) >= -tol) * 
             ((h1.ub - h2.lb) >= -tol))

    # the domains in each direction touch   
    if not (touch.sum() == h1.ndim):
        return False

    # in ndim-1 directions the directions' domains overlap (either perfectly, or one inside another)
    overlap = (signs_of_array(h2.ub - h1.ub, tol) ==
               signs_of_array(h1.lb - h2.lb, tol))
    if not (overlap.sum() == h1.ndim-1):
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

def _divide_hrect(hyperrectangle, vectorizable, func, hrect, dims, f_args, log_min_l):
    resolution_reached = hrect.length_inds < log_min_l
    # do not split along resolution axes
    dims = np.array([i for i in dims if not resolution_reached[i]])
    if len(dims) == 0:
        # do not lose hrect - may be splittable along different axis
        return [hrect] 
    
    indcs = gen_boolmatrix(len(dims))
    n_new = 2**len(dims)

    centres = _generate_centres(hrect, indcs, dims)
    lbs, ubs = _generate_bounds(hrect, indcs, dims)
    pf, gen, cuts = hrect.f, hrect.generation + 1, hrect.cuts + len(dims)
    
    if vectorizable is True: 
        f_values = func(centres.T, *f_args, gen, cuts)

    else:
        f_values=_mp_evaluateobjective(func, centres, f_args, gen, cuts)
        
    hrects = [hyperrectangle(
        centres[k], f_values[k], gen, cuts, lbs[k], ubs[k], pf) 
        for k in range(n_new)]

    return hrects

@njit(parallel=True)
def _mp_evaluateobjective(func, centres, f_args, gen, cuts):
    fs = np.empty(len(centres), dtype=np.float64)
    for i in prange(len(centres)):
        fs[i] = func(centres[i,:], *f_args, gen, cuts)
    return fs


def Direct(
    func, 
    bounds,
    vectorizable=False,
    callback=None,
    restart='',
    program=None,
    disp=False,
    
    f_args=(),
    maxiter=np.inf,
    maxfev=np.inf,
    rect_dim=-1,
    population=1,
    resolution=-np.inf,
    locally_biased=False,
    near_optimal=-1,
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
    resolution      - maximum resolution of the solution. Should be number or array 
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
    
    if program is not None:
        warnings.warn(
            'A program was passed. Parameters of the program may override '
            +'other arguments passed. ', UserWarning)
    try: 
        lb, ub = bounds
        ndim = len(lb)
        
        centre = 0.5*(ub - lb) + lb 
        
        if restart != '': 
            archive, elite, total_vol = _restart(restart, bounds, disp)
            parents, prev_bests = np.array([], dtype=hyperrectangle), np.array([], dtype=hyperrectangle)
            archive = np.array(archive)
        else: 
            if vectorizable: 
                f = func(np.atleast_2d(centre).T, *f_args, -1, 0)[0]
            else: 
                f = func(centre, *f_args, -1, 0)
                
            elite = hyperrectangle(centre, f, -1, 0, lb, ub, np.inf)
            parents = np.array([elite])
            archive, prev_bests = np.array([], dtype=hyperrectangle), np.array([], dtype=hyperrectangle)
        
        i, conv_count, miter_adj, mfev_adj, it_best = 0, 0, 0, 0, 0
        interrupted = False
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
            conv_max = ndim // rect_dim + 1 

            log_min_l = np.log10(resolution)
            total_vol = (ub-lb).prod()
            local_minima = np.array([], dtype=hyperrectangle)
            near_optimal_res, conv_count = True, 0
            
            while (i < maxiter+miter_adj # stop at max iterations
                   and fev < maxfev+mfev_adj  # stop at max function evaluations
                   and conv_count < conv_max # stop when no improvement to best (locally biased)
                   and total_vol > 0 # stop when resolution fully resolved
                #    and it_best < elite.f * alt_threshold # stop when no longer finding near-optimal space
                   and near_optimal_res # trigger to move down a resolution step when maximum resolution reached for near_optimal mode
                   ): 
                
                it_start = dt.datetime.now()
                # split all hrects to be split from previous iteration

                new_hrects = np.array([hrect for parent in 
                                       tqdm(parents, desc=f'it {i}: #hrects = {len(parents)}. Evaluating Rectangles: ', leave=False)
                                       for hrect in _divide_hrect(
                    hyperrectangle, vectorizable, func, parent, dims, f_args, log_min_l)])
                
                fev += len(new_hrects)
        
                # all hrects which do not have any children  
                childless = np.concatenate((new_hrects, archive))
        
                # generate array of list-index, cost, and volume
                fvs = np.array([(j, h.f, h.volume) for j, h in enumerate(childless)], dtype=np.float64)
                # Make sure we haven't lost search hyperspace
                assert abs(1-sum(fvs[:, 2])/total_vol) < 1e-6, f'{sum(fvs[:, 2])} / {total_vol}' # tolerance for floating point 
                
                # sort list indices by cost
                fvs = fvs[fvs[:,1].argsort(), :]
                fs = fvs[:,0].astype(np.int64)
                
                if childless[fs[0]].f < elite.f: 
                    # optimal value may not be found in the youngest generation
                    elite = childless[fs[0]]
                
                if near_optimal == -1:
                    # get list-indicies of the best {population} hrects by cost 
                    best = np.array(fs[:min(population, len(fvs))], dtype=np.int64)
                else:
                    # every near-optimal rectangle
                    nearoptimalmask = fvs[:, 1] <= near_optimal*elite.f
                    best = fs[nearoptimalmask] 
                
                if locally_biased is True:
                    # Triggers termination if the best rectangles all stay the same for the full rotation of splitting axes
                    prev_bests = childless[best]
                    new_accepted = np.array([j for b in childless[best] for j, hrect in enumerate(childless) 
                                             if hrects_border(b, hrect) 
                                             and not hrect_is_barren(hrect.length_inds, log_min_l)], dtype=np.int64)
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
                    lm_i = np.array([j for j in fs if hrect_is_barren(childless[j].length_inds, log_min_l)], dtype=np.int64)
                    total_vol -= sum([h.volume for h in childless[lm_i]])
                    local_minima = np.concatenate((local_minima, np.array(childless[lm_i])))
                    
                    # rotate splitting axes
                    dims += rect_dim 
                    dims %= ndim
                    
                    if near_optimal == -1:
                        best_, barrencount = [], 0
                        for f_i in fs:
                            if not hrect_is_semibarren(childless[f_i], dims, log_min_l):
                                best_.append(f_i)
                            else: 
                                barrencount +=1
                            if len(best_) >= min(population, len(fs)) + barrencount: # want to include neighbours 
                                break
                        best = np.array(best_, dtype=np.int64)
                    else: 
                        if len(best)==0:
                            best = np.array([], dtype=np.int64)
                            near_optimal_res = False
                    
                    #This may seem overcomplicated, but it greatly improves speed with 10000+ best in near-optimal exploration
                    new_accepted = np.array([], dtype=np.int64)
                    ineligible = np.array([], dtype=np.int64)
                    if len(best)>0:
                        ineligible = np.concatenate((best, _borderheuristic(list(childless), list(childless[best]))))
                        semi_barr = _semibarren_speedup(list(childless), dims, log_min_l)
                        ineligible = np.union1d(ineligible, semi_barr)
                    for b in tqdm(best, desc=f'it {i}: #hrects = {len(parents)}. Sorting rectangles:', leave=False):
                        eligible = np.setdiff1d(np.arange(len(childless)), ineligible).astype(np.int64)
                        if len(eligible)==0:
                            break
                        new_accepted = _accept_rect(list(childless), childless[b], eligible)
                        ineligible = np.union1d(ineligible, new_accepted)
    
                    # combine new and archived hrects to be split next iteration
                    new_accepted = np.unique(np.concatenate((best, new_accepted)))
                    # remove local minima
                    new_accepted = np.setdiff1d(new_accepted, lm_i, assume_unique=True)
                    # remove non-parents
                    new_accepted = np.setdiff1d(new_accepted, 
                                                _semibarren_speedup(list(childless), dims, log_min_l), 
                                                assume_unique=True)
    
                    # get list-indices of childless hrects which are not to be split
                    to_arch = np.setdiff1d(np.arange(len(childless)), new_accepted, assume_unique=True)
                    
                    # remove local minima
                    to_arch = np.setdiff1d(to_arch, lm_i, assume_unique=True)

                it_time = dt.datetime.now() - it_start
                if disp is True: 
                    print(f'Took: {it_time}. Best value: {elite.f}.')
                if callback is not None:
                    callback(elite)

                
                # update archive
                archive = childless[to_arch]
                # old parents are forgotten
                parents = childless[new_accepted]
                    
                i+=1
            archive = np.concatenate((archive, local_minima))
            miter_adj += i
            mfev_adj += fev
            print('\nprogram step\n')
    except KeyboardInterrupt:
        interrupted = True
        pass
    print('\n')
    return DirectResult(elite.centre, elite.f, fev, i, interrupted, 
                          elite.lb, elite.ub, elite.volume, elite.volume/total_vol)

@njit(parallel=True)
def _semibarren_speedup(rects, dims, log_min_l):
    accepted = np.zeros(len(rects), dtype=np.bool_)
    for i in prange(len(rects)):
        if hrect_is_semibarren(rects[i], dims, log_min_l):
            accepted[i] = True
    
    return np.arange(len(rects), dtype=np.int64)[accepted]

@njit(parallel=True)
def _accept_rect(rects, b, eligible):
    accepted = np.zeros(len(eligible), dtype=np.bool_)
    for i in prange(len(eligible)):
        if hrects_border(b, rects[eligible[i]]):
            accepted[i] = True
    
    return eligible[accepted]

        

@njit
def _borderheuristic(rects, best):
    ndim = best[0].ndim
    minlb =  np.inf*np.ones(ndim, dtype=np.float64)
    maxub = -np.inf*np.ones(ndim, dtype=np.float64)
    
    for i in range(len(best)): 
        minlb = np.minimum(minlb, best[i].lb)
        maxub = np.maximum(maxub, best[i].ub)
    
    _toofaraway=np.array([
        i for i, h in enumerate(rects) if (h.lb >= maxub).sum() + (h.ub <= minlb).sum() > 1
        ], dtype=np.int64)

    return _toofaraway
    

def _restart(restart, bounds, disp):

    history = np.genfromtxt(restart, delimiter=',', dtype=np.float64)
    fs, xs = history[:,:3], history[:,3:]
    
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

    elite = fs[:,0].argmin()
    elite = hyperrectangle(xs[elite], *fs[elite], lbs[elite], ubs[elite], np.nan)
    
    
    print('Restarting optimisation where',restart,'left off.')
# =============================================================================
# Child-wise loop is faster 
# Iterates backwards through list of rectangles
# For each rectangle, directly removes any parents it finds (searching backwards)
    _child_loop(xs[:1,:], fs[:1,:], lbs[:1, :], ubs[:1, :]) # compile jit
    archive = _child_loop(xs, fs, lbs, ubs)
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
    def __init__(self, x, f, nfev, nit, interrupted, lb, ub, volume, vratio):
        self.x = x
        self.f = f 
        self.nfev = nfev
        self.nit = nit
        self.interrupted = interrupted
        self.lb, self.ub = lb, ub
        self.volume = volume
        self.vratio = vratio
        
