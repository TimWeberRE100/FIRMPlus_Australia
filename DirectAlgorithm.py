# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 08:28:16 2024

@author: u6942852
"""

import numpy as np 
from numba import njit, prange, float64, int64, objmode
from numba.experimental import jitclass
import datetime as dt
from tqdm import tqdm
import warnings
from csv import writer
from multiprocessing import cpu_count
from time import sleep 

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
    # ('rdif', float64),
    # ('adif', float64),
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
def hrects_is_same(h1, h2):
    ub_same = (h1.ub == h2.ub).prod() == 1 
    lb_same = (h1.lb == h2.lb).prod() == 1 
    c_same = (h1.centre == h2.centre).prod() == 1
    f_same = h1.f == h2.f
    return ub_same and lb_same and c_same and f_same

@njit
def hrects_border(h1, h2, tol = 1e-12):
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
    
    # Rectangles are the same
    if (h1.ub != h2.ub).sum() + (h1.lb != h2.lb).sum() == 0:
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
    dims = dims[(hrect.length_inds >= log_min_l)[dims]]
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
    dims = dims[(hrect.length_inds >= log_min_l)[dims]]
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
    else: 
        if vectorizable: 
            f = func(np.atleast_2d(centre).T, *f_args)[0]
        else: 
            f = func(centre, *f_args)
            
        if printfile!='':
            with open(printfile+'-parents.csv', 'w', newline='') as csvfile:
                writer(csvfile)
            with open(printfile+'-children.csv', 'w') as csvfile:
                writer(csvfile)
            with open(printfile+'-resolved.csv', 'w') as csvfile:
                writer(csvfile)
            
        elite = hyperrectangle(centre, f, -1, 0, lb, ub, np.inf)
        parents = np.array([elite])
        archive, prev_bests = np.array([], dtype=hyperrectangle), np.array([], dtype=hyperrectangle)
    
    _divide_hrect = _divide_vec if vectorizable is True else _divide_mp
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
        landlocked_resolved, edge_resolved = np.array([], dtype=hyperrectangle), np.array([], dtype=hyperrectangle)
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
            print(' ', end='\r', flush=True)
            print(f'it {i} - #hrects: {len(parents)}. Sorting Rectangles...', end='\r', flush=True)
            fev += len(new_hrects)
    
            # all hrects which do not have any children  
            childless = np.concatenate((new_hrects, archive))
            del new_hrects, archive #reduce memory load
    
            # generate array of list-index, cost
            fvs = np.array([(j, h.f) for j, h in enumerate(childless)], dtype=np.float64)
            gen_elite = childless[int(fvs[fvs[:,1].argmin(), 0])]
            if gen_elite.f < elite.f: 
                # optimal value may not be found in the youngest generation
                elite = gen_elite
    
            # maximum resolution rectangles 
            resolved_mask = _semibarren_speedup(list(childless), np.ones(ndim, dtype=np.bool_), log_min_l)
            total_vol -= sum([h.volume for h in childless[resolved_mask]])
            
            
            len_allresolved = resolved_mask.sum() + len(edge_resolved) + len(landlocked_resolved)
            #choose which method to use based on approximate no. of comparisons required
            if resolved_mask.sum() > 0:
                if resolved_mask.sum()*len_allresolved < resolved_mask.sum()*((~resolved_mask).sum()):
                    llresolved_mask = landlocked_bysum(
                        list(childless[resolved_mask]),
                        list(np.concatenate((childless[resolved_mask], 
                                             landlocked_resolved, 
                                             edge_resolved))),
                        bounds)
                    
                else: 
                    llresolved_mask = landlocked_bycontra(
                        list(childless[resolved_mask]),
                        list(childless[~resolved_mask]))
            else: 
                llresolved_mask = np.array([], dtype=np.bool_)
            if len(edge_resolved)>0:
                if len(edge_resolved)*len_allresolved < len(edge_resolved)*((~resolved_mask).sum()):
                    lledge_mask = landlocked_bysum(
                        list(edge_resolved),
                        list(np.concatenate((childless[resolved_mask], 
                                             landlocked_resolved, 
                                             edge_resolved))),
                        bounds)
                else: 
                    lledge_mask = landlocked_bycontra(
                        list(edge_resolved),
                        list(childless[~resolved_mask]))
            else: 
                lledge_mask = np.array([], dtype=np.bool_)
            landlocked_resolved = np.concatenate((landlocked_resolved,
                                                  edge_resolved[lledge_mask],
                                                  childless[resolved_mask][llresolved_mask]))
            
            edge_resolved = np.concatenate((edge_resolved[~lledge_mask], 
                                            childless[resolved_mask][~llresolved_mask]))

            childless = childless[~resolved_mask]

            if printfile != '':
                print(' ', end='\r', flush=True)
                print(f'it {i} - #hrects: {len(parents)}. Writing out to file. Do not Interrupt.', end='\r', flush=True)
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
                with open(printfile+'-resolved.csv', 'w', newline='') as csvfile:
                    resolved = np.concatenate((landlocked_resolved, edge_resolved))
                    if len(resolved) > 0:
                        printout = np.concatenate((np.array([(h.f, h.generation, h.cuts) for h in resolved]), 
                                                   np.array([h.centre for h in resolved])), 
                                                   axis=1)
                        writer(csvfile).writerows(printout)
                        del resolved
                del printout
                print(' ', end = '\r', flush=True)
                print(f'it {i} - #hrects: {len(parents)}. Sorting Rectangles... {" "*20}', end='\r', flush=True)
            
            # generate array of list-index, cost, and volume
            fvs = np.array([(j, h.f, h.volume) for j, h in enumerate(childless)], dtype=np.float64)
            # Make sure we haven't lost search space
            assert abs(1-sum(fvs[:, 2])/total_vol) < 1e-6, f'{sum(fvs[:, 2])} / {total_vol}' # tolerance for floating point 
            
            # sort list indices by cost
            fvs = fvs[fvs[:,1].argsort(), :]
            fs = fvs[:,0].astype(np.int64)

            # near-optimal rectangles
            nearoptimalmask = fvs[:, 1] <= near_optimal*elite.f
            best = fs[nearoptimalmask] 

            if locally_biased is True:
                best = best[:min(population, len(best))]
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
                # rotate splitting axes
                dims += rect_dim 
                dims %= ndim
                
                if len(best)>0:
                    # list indices of best rectangles which are semibarren
                    best_semibarr = best[_semibarren_speedup(list(childless[best]), dims, log_min_l)]
                    
                    best = np.setdiff1d(best, best[best_semibarr], assume_unique=True)
                    best = best[:min(len(best), population)]
                
                # get list-indices of childless hrects which are not to be split
                new_accepted = np.zeros(len(childless), dtype=np.bool_)
                new_accepted[best] = True

            if new_accepted.sum() == 0: 
                conv_count+=1
            else: 
                conv_count=0

            if new_accepted.sum() == 0 and conv_count >= conv_max:
                near_optimal_threshold = elite.f*near_optimal
                near_optimal_resolved = np.array([h.f < near_optimal_threshold for h in edge_resolved])

                if near_optimal_resolved.sum()>0:
                    eligible = np.ones(len(childless), dtype=np.bool_)
                    eligible[_semibarren_speedup(list(childless[eligible]), dims, log_min_l)] = False
                    if eligible.sum() > 0:
                        eligible[_borderheuristic(list(childless[eligible]), 
                                                  list(edge_resolved[near_optimal_resolved]))] = False
                    print(' ', end = '\r', flush=True)
                    print(f'it {i} - #hrects: {len(parents)}. Identifying near-optimal neighbours. Estimated time: ', end='', flush=True)  
                    if eligible.sum() <= cpu_count()*14 or eligible.sum()*near_optimal_resolved.sum() >= 10e10:
                        print('< a few minutes. ', end ='', flush=True)
                        new_accepted = sortrectangles(list(edge_resolved[near_optimal_resolved]), 
                                                      list(childless[eligible]))
                    else: 
                        time_test_range = cpu_count()*7
                        
                        timer_mask = np.zeros(len(eligible), dtype=np.bool_)
                        timer_mask[:_find_bool_indx(eligible, time_test_range)+1] = True
                        
                        sort_start = dt.datetime.now()
                        new_accepted = sortrectangles(list(edge_resolved[near_optimal_resolved]), 
                                                      list(childless[eligible*timer_mask]))
                        sort_time = (eligible.sum() - time_test_range)/time_test_range*(dt.datetime.now()-sort_start)
                        print(f'{sort_time}. Estimated end time: {dt.datetime.now() + sort_time}. ', end='', flush=True)
                        new_accepted = np.concatenate((new_accepted, 
                            sortrectangles(list(edge_resolved[near_optimal_resolved]), 
                                            list(childless[eligible*~timer_mask]))))

                    print('Done.', end ='\r', flush=True)
                    print(' '*150, end='\r', flush=True)
                    
                    eligible[eligible == True] = new_accepted
                    new_accepted=eligible
                else: 
                    new_accepted = np.zeros(len(childless), dtype=np.bool_)

            if new_accepted.sum() > 0:
                conv_count=0

            if new_accepted.sum() > MAXPARENTS: # prevents memory error
                _maxindx = _find_bool_indx(new_accepted, MAXPARENTS)
                new_accepted = new_accepted[_maxindx+1:] = False

            it_time = dt.datetime.now() - it_start
            if disp is True: 
                print(' ', end='\r', flush=True)
                print(f'it {i} - #hrects: {len(parents)}. Took: {it_time}. Best value: {elite.f}.', flush=True)
            if callback is not None:
                callback(elite)

            # update archive
            archive = childless[~new_accepted]
            # old parents are forgotten
            parents = childless[new_accepted]
            
            i+=1
        archive = np.concatenate((archive, landlocked_resolved, edge_resolved))
        if printfile != '':
            print(f'it {i} - #hrects: {len(parents)}. Writing out to file. Do not Interrupt.', end='\r', flush=True)
            with open(printfile+'-resolved.csv', 'w') as csvfile:
                writer(csvfile)
            with open(printfile+'-children.csv', 'w', newline='') as csvfile:
                if len(archive) > 0:
                    printout = np.concatenate((np.array([(h.f, h.generation, h.cuts) for h in archive]), 
                                               np.array([h.centre for h in archive])), 
                                               axis=1)
                    writer(csvfile).writerows(printout)
            print(' '*100, end='\r', flush=True)
        
        miter_adj += i
        mfev_adj += fev
        print(f'{"-"*50}\nprogram step\n{"-"*50}', flush=True)

    print('\n')
    return DirectResult(elite.centre, elite.f, fev, i, 
                          elite.lb, elite.ub, elite.volume, elite.volume/total_vol)                  

@njit
def _find_bool_indx(mask, count):
    """returns the index of the boolean mask such that there are {count} Trues 
    before it"""
    _mask_indx, _counter = -1, 0
    while _counter < count:
        _mask_indx+=1
        if mask[_mask_indx]:
            _counter += 1
    return _mask_indx

@njit 
def _sub_landlocked_bysum(h, pool, bounds):
    """ Returns True if h is landlocked by pool """
    """ Assumes h is of the smallest resolution in archive """
    faces = h.ndim*2 - (h.lb == bounds[0]).sum() - (h.ub == bounds[1]).sum()
    for h2 in pool:
        if hrects_border(h, h2):
            faces -= 1 
        if faces == 0:
            return True
    return False
            
@njit(parallel=True)
def landlocked_bysum(eligible, resolved, bounds):
    accepted = np.empty(len(eligible), dtype=np.bool_)
    for i in prange(len(eligible)):
        accepted[i] = _sub_landlocked_bysum(eligible[i], resolved, bounds)
    return accepted 

@njit 
def _sub_landlocked_bycontra(h, antipool):
    """ Returns True if h is landlocked by pool """
    for h2 in antipool: 
        if hrects_border(h, h2):
            return False
    return True

@njit(parallel=True)
def landlocked_bycontra(eligible, unresolved):
    accepted = np.empty(len(eligible), dtype=np.bool_)
    for i in prange(len(eligible)):
        accepted[i] = _sub_landlocked_bycontra(eligible[i], unresolved)
    return accepted

@njit(parallel=True)
def _semibarren_speedup(rects, dims, log_min_l):
    accepted = np.empty(len(rects), dtype=np.bool_)
    for i in prange(len(rects)):
        accepted[i] = hrect_is_semibarren(rects[i], dims, log_min_l)
    return accepted

@njit(parallel=True)
def sortrectangles(resolved, eligible):
    accepted=np.zeros(len(eligible), dtype=np.bool_)
    for i in prange(len(eligible)):
        h = eligible[i]
        for r in resolved:
            if hrects_border(h, r):
                accepted[i] = True
                break
    return accepted


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

    return accepted
    
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
        resolved = np.genfromtxt(restart+'-resolved.csv', delimiter=',', dtype=np.float64)
        if len(resolved) != 0:
            history = np.vstack((history, np.atleast_2d(resolved)))
        del resolved
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
        del fps, xps, lbps, ubps, parents

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
        
