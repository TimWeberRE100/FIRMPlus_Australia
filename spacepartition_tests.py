# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:12:38 2024

@author: u6942852
"""

import unittest
import numpy as np
from numba import njit 
from scipy.linalg import pascal

from spacepartition import *

class CustomAssertions:
    def assertArrayEqual(self, a, b, equal_nan=False, msg=None):
        if not a.shape == b.shape:
            self.fail(msg)
        if not np.isclose(a, b, equal_nan).all():
            self.fail(msg)
            
    def assertArraySetEqual(self, a, b, msg=None):
        if not set(a) == set(b):
            self.fail(msg)
    
    def assertArrayEmpty(self, arr, msg=None):
        if not arr.size==0:
            self.fail(msg)
            
    def assertNotRaises(self, exception, callable, *args, **kwds):
        try:
            callable(*args, **kwds)
        except exception:
            self.fail()

def f(args):
    return sum([x**2/i for i, x in enumerate(args, 1)])

def hyper_wrapper(centre, half_length):
    """simple wrapper for creating hyperrectangle """
    return hyperrectangle(centre.astype(np.float64), f(centre), 0, 0, np.array([]), half_length.astype(np.float64))
    
    
class TestHyperrectangleFuncs(unittest.TestCase, CustomAssertions):
    
    def setUp(self):
        # fully resolved 2d
        self.hrects2d = np.array([
            hyper_wrapper(np.array([25, 25]), np.array([25, 25])),
            hyper_wrapper(np.array([75, 25]), np.array([25, 25])),
            hyper_wrapper(np.array([25, 75]), np.array([25, 25])),
            hyper_wrapper(np.array([75, 75]), np.array([25, 25])),
            ])
        self.conns2d = np.array([
            [1,2], #hrect3d_2[0] connects to [1], and [2]
            [0,3], #hrect3d_2[1] connects to [0], and [3]
            [0,3],
            [1,2]])
        
        # fully resolved 3d
        self.hrects3d_1 = np.array([
            hyper_wrapper(np.array([25, 25, 25]), np.array([25, 25, 25])),
            hyper_wrapper(np.array([75, 25, 25]), np.array([25, 25, 25])),
            hyper_wrapper(np.array([25, 75, 25]), np.array([25, 25, 25])),
            hyper_wrapper(np.array([75, 75, 25]), np.array([25, 25, 25])),
            hyper_wrapper(np.array([25, 25, 75]), np.array([25, 25, 25])),
            hyper_wrapper(np.array([75, 25, 75]), np.array([25, 25, 25])),
            hyper_wrapper(np.array([25, 75, 75]), np.array([25, 25, 25])),
            hyper_wrapper(np.array([75, 75, 75]), np.array([25, 25, 25])),
            ])
        self.conns3d_1 = np.array([
            [1,2,4], #hrect3d_1[0] connects to [1], [2], and [4]
            [0,3,5], #hrect3d_1[1] connects to [0], [3], and [5]
            [0,3,6],
            [1,2,7],
            [0,5,6],
            [1,4,7],
            [2,4,7],
            [3,5,6]])
        
        # partially resolved 3d
        self.hrects3d_2 = np.array([
            hyper_wrapper(np.array([25, 25, 50]), np.array([25, 25, 50])),
            hyper_wrapper(np.array([25, 75, 50]), np.array([25, 25, 50])),
            hyper_wrapper(np.array([75, 50, 25]), np.array([25, 50, 25])),
            hyper_wrapper(np.array([75, 50, 75]), np.array([25, 50, 25])),
            ])
        self.conns3d_2 = np.array([
            [  1,2,3], #hrect3d_2[0] connects to [1], [2], and [3]
            [0,  2,3], #hrect3d_2[1] connects to [0], [2], and [3]
            [0,1,  3],
            [0,1,2  ]])
        
        # # garbled blend of parents and children
        self.hrects3d_3 = np.concatenate((self.hrects3d_1, self.hrects3d_2))
        self.conns3d_3 = np.array([
            # some conns are duplicated so that the shape of the array is constant 
            # note: all conns are valid, but children of parents do not neighbour
            [1, 2, 4, 9, 10, 1, 1],
            [0, 3, 5, 8, 11, 0, 0], 
            [0, 3, 6, 8, 10, 0, 0],
            [1, 2, 7, 9, 11, 1, 1],
            [0, 5, 6, 9, 11, 0, 0],
            [1, 4, 7, 8, 10, 1, 1],
            [2, 4, 7, 8, 11, 2, 2],
            [3, 5, 6, 9, 10, 3, 3],
            [9, 10, 11, 1, 2, 5, 6],
            [8, 10, 11, 0, 3, 4, 7],
            [8, 9, 11, 0, 2, 5, 7],
            [8, 9, 10, 1, 3, 4, 6]])
        
        hl = np.array([12.5, 12.5, 12.5])
        self.hrects3d_4 = np.array([
            hyper_wrapper(np.array([12.5, 12.5, 12.5]), hl), 
            hyper_wrapper(np.array([12.5, 12.5, 37.5]), hl), 
            hyper_wrapper(np.array([12.5, 37.5, 12.5]), hl), 
            hyper_wrapper(np.array([12.5, 37.5, 37.5]), hl), 
            hyper_wrapper(np.array([37.5, 12.5, 12.5]), hl), 
            hyper_wrapper(np.array([37.5, 12.5, 37.5]), hl), 
            hyper_wrapper(np.array([37.5, 37.5, 12.5]), hl), 
            hyper_wrapper(np.array([37.5, 37.5, 37.5]), hl), 
            hyper_wrapper(np.array([12.5, 12.5, 62.5]), hl), 
            hyper_wrapper(np.array([12.5, 12.5, 87.5]), hl), 
            hyper_wrapper(np.array([12.5, 37.5, 62.5]), hl), 
            hyper_wrapper(np.array([12.5, 37.5, 87.5]), hl), 
            hyper_wrapper(np.array([37.5, 12.5, 62.5]), hl), 
            hyper_wrapper(np.array([37.5, 12.5, 87.5]), hl), 
            hyper_wrapper(np.array([37.5, 37.5, 62.5]), hl), 
            hyper_wrapper(np.array([37.5, 37.5, 87.5]), hl), 
            hyper_wrapper(np.array([12.5, 62.5, 12.5]), hl), 
            hyper_wrapper(np.array([12.5, 62.5, 37.5]), hl), 
            hyper_wrapper(np.array([12.5, 87.5, 12.5]), hl), 
            hyper_wrapper(np.array([12.5, 87.5, 37.5]), hl), 
            hyper_wrapper(np.array([37.5, 62.5, 12.5]), hl), 
            hyper_wrapper(np.array([37.5, 62.5, 37.5]), hl), 
            hyper_wrapper(np.array([37.5, 87.5, 12.5]), hl), 
            hyper_wrapper(np.array([37.5, 87.5, 37.5]), hl), 
            hyper_wrapper(np.array([12.5, 62.5, 62.5]), hl), 
            hyper_wrapper(np.array([12.5, 62.5, 87.5]), hl), 
            hyper_wrapper(np.array([12.5, 87.5, 62.5]), hl), 
            hyper_wrapper(np.array([12.5, 87.5, 87.5]), hl), 
            hyper_wrapper(np.array([37.5, 62.5, 62.5]), hl), 
            hyper_wrapper(np.array([37.5, 62.5, 87.5]), hl), 
            hyper_wrapper(np.array([37.5, 87.5, 62.5]), hl), 
            hyper_wrapper(np.array([37.5, 87.5, 87.5]), hl), 
            hyper_wrapper(np.array([62.5, 12.5, 12.5]), hl), 
            hyper_wrapper(np.array([62.5, 12.5, 37.5]), hl), 
            hyper_wrapper(np.array([62.5, 37.5, 12.5]), hl), 
            hyper_wrapper(np.array([62.5, 37.5, 37.5]), hl), 
            hyper_wrapper(np.array([87.5, 12.5, 12.5]), hl), 
            hyper_wrapper(np.array([87.5, 12.5, 37.5]), hl), 
            hyper_wrapper(np.array([87.5, 37.5, 12.5]), hl), 
            hyper_wrapper(np.array([87.5, 37.5, 37.5]), hl), 
            hyper_wrapper(np.array([62.5, 12.5, 62.5]), hl), 
            hyper_wrapper(np.array([62.5, 12.5, 87.5]), hl), 
            hyper_wrapper(np.array([62.5, 37.5, 62.5]), hl), 
            hyper_wrapper(np.array([62.5, 37.5, 87.5]), hl), 
            hyper_wrapper(np.array([87.5, 12.5, 62.5]), hl), 
            hyper_wrapper(np.array([87.5, 12.5, 87.5]), hl), 
            hyper_wrapper(np.array([87.5, 37.5, 62.5]), hl), 
            hyper_wrapper(np.array([87.5, 37.5, 87.5]), hl), 
            hyper_wrapper(np.array([62.5, 62.5, 12.5]), hl), 
            hyper_wrapper(np.array([62.5, 62.5, 37.5]), hl), 
            hyper_wrapper(np.array([62.5, 87.5, 12.5]), hl), 
            hyper_wrapper(np.array([62.5, 87.5, 37.5]), hl), 
            hyper_wrapper(np.array([87.5, 62.5, 12.5]), hl), 
            hyper_wrapper(np.array([87.5, 62.5, 37.5]), hl), 
            hyper_wrapper(np.array([87.5, 87.5, 12.5]), hl), 
            hyper_wrapper(np.array([87.5, 87.5, 37.5]), hl), 
            hyper_wrapper(np.array([62.5, 62.5, 62.5]), hl), 
            hyper_wrapper(np.array([62.5, 62.5, 87.5]), hl), 
            hyper_wrapper(np.array([62.5, 87.5, 62.5]), hl), 
            hyper_wrapper(np.array([62.5, 87.5, 87.5]), hl), 
            hyper_wrapper(np.array([87.5, 62.5, 62.5]), hl), 
            hyper_wrapper(np.array([87.5, 62.5, 87.5]), hl), 
            hyper_wrapper(np.array([87.5, 87.5, 62.5]), hl), 
            hyper_wrapper(np.array([87.5, 87.5, 87.5]), hl)])
        
        
    def _test_hrectborders(self, hrects, conns):
        for i, row in enumerate(conns):
            # borders everything it should
            for j in row: 
                self.assertTrue(hrects_border(hrects[i], hrects[j]), f'i={i}, j={j}')
                self.assertTrue(hrects_border(hrects[j], hrects[i]), f'i={i}, j={j}')
            # does not border things it should not border
            # returns false when hrect is the same
            for j in set(range(len(hrects))) - set(row): 
                self.assertFalse(hrects_border(hrects[i], hrects[j]), f'i={i}, j={j}')
                self.assertFalse(hrects_border(hrects[j], hrects[i]), f'i={i}, j={j}')
        
    def test_hrectborders(self):
        self._test_hrectborders(self.hrects2d, self.conns2d)
        self._test_hrectborders(self.hrects3d_1, self.conns3d_1)
        self._test_hrectborders(self.hrects3d_2, self.conns3d_2)
        self._test_hrectborders(self.hrects3d_3, self.conns3d_3)
        
        self.assertTrue(find_neighbours(list(self.hrects3d_2), list(self.hrects3d_2)).all())
        
        for i, row in enumerate(self.conns3d_3):
            self.assertArraySetEqual(np.where(find_neighbours(list(self.hrects3d_3), [self.hrects3d_3[i]]))[0], row)
        
        for i, j in enumerate(range(2, 12)):
            result, slicer = np.zeros(12, bool), np.zeros(12, bool)
            slicer[i:j] = True
            result[~slicer] = find_neighbours(list(self.hrects3d_3[~slicer]), list(self.hrects3d_3[slicer]))
            self.assertArraySetEqual(np.where(result)[0], 
                                     set(self.conns3d_3[slicer].flatten()) - set(np.where(slicer)[0]), msg=f'i={i}, j={j}')                    
            
        for i, j in enumerate(range(3, 12)):
            result, slicer = np.zeros(12, bool), np.zeros(12, bool)
            slicer[i:j] = True
            result[~slicer] = find_neighbours(list(self.hrects3d_3[~slicer]), list(self.hrects3d_3[slicer]))
            self.assertArraySetEqual(np.where(result)[0], 
                                     set(self.conns3d_3[slicer].flatten()) - set(np.where(slicer)[0]), msg=f'i={i}, j={j}')    
        
    def test_hrectsemibarren(self):
        h = self.hrects2d[0]
        self.assertTrue(hrect_semibarren(h, np.array([True, True]), np.array([25., 25.])))
        self.assertTrue(hrect_semibarren(h, np.array([True, False]), np.array([25., 25.])))
        self.assertTrue(hrect_semibarren(h, np.array([0,1]), np.array([25., 25.])))
        
        #dims needs to be a valid slicer
        self.assertRaises(Exception, hrect_semibarren, h, [0, 1], np.array([25., 25.]))
        self.assertRaises(Exception, hrect_semibarren, h, [True, False], np.array([25., 25.]))
        #min_half_length needs to be broadcastable
        self.assertRaises(Exception, hrect_semibarren, h, np.array([True, True]), np.array([25., 25., 25.]))
        self.assertNotRaises(Exception, hrect_semibarren, h, np.array([True, True]), 25.)
        #min_half_length should not be slived by dims
        self.assertRaises(   Exception, hrect_semibarren, self.hrects3d_2[0], np.array([True, True, False]), np.array([25., 25.]))
        self.assertNotRaises(Exception, hrect_semibarren, self.hrects3d_2[0], np.array([True, True, False]), np.array([25., 25., 25.]))
        
        self.assertFalse(hrect_semibarren(self.hrects3d_2[0], np.ones(3, bool), np.array([25., 50., 25.])))
        self.assertFalse(hrect_semibarren(self.hrects3d_2[1], np.ones(3, bool), np.array([25., 50., 25.])))
        self.assertTrue(hrect_semibarren(self.hrects3d_2[2], np.ones(3, bool), np.array([25., 50., 25.])))
        self.assertTrue(hrect_semibarren(self.hrects3d_2[3], np.ones(3, bool), np.array([25., 50., 25.])))
        
        self.assertArrayEqual(
            semibarren_speedup(list(self.hrects3d_2), np.ones(3, bool), np.array([25., 50., 25.])),
            np.array([False, False, True, True]))
        
        self.assertArrayEqual(
            semibarren_speedup(list(self.hrects3d_2), np.array([0, 1]), np.array([25., 25., 25.])),
            np.array([True, True, False, False]))
        
        self.assertArrayEqual(
            semibarren_speedup(list(self.hrects3d_2), np.array([1, 2]), np.array([25., 25., 25.])),
            np.array([False, False, False, False]))
        
    
    def test_generatecentres(self):
        self.assertArrayEqual(np.unique(generate_centres(self.hrects2d[0], np.array([0,1])), axis=0),
                              np.array([[12.5, 12.5], 
                                        [12.5, 37.5], 
                                        [37.5, 12.5],
                                        [37.5, 37.5]]))
        self.assertArrayEqual(np.unique(generate_centres(self.hrects2d[0], np.array([0])), axis=0),
                              np.array([[12.5, 25.], 
                                        [37.5, 25.]]))
        self.assertArrayEqual(np.unique(generate_centres(self.hrects2d[0], np.array([1])), axis=0),
                              np.array([[25., 12.5], 
                                        [25., 37.5]]))
        
        
        self.assertArrayEqual(np.unique(generate_centres(self.hrects3d_1[-1], np.array([0,1])), axis=0),
                              np.array([[62.5, 62.5, 75.], 
                                        [62.5, 87.5, 75.], 
                                        [87.5, 62.5, 75.],
                                        [87.5, 87.5, 75.]]))
        
        self.assertArrayEqual(np.unique(generate_centres(self.hrects3d_1[-1], np.array([0,1,2])), axis=0),
                              np.array([[62.5, 62.5, 62.5], 
                                        [62.5, 62.5, 87.5], 
                                        [62.5, 87.5, 62.5], 
                                        [62.5, 87.5, 87.5],
                                        [87.5, 62.5, 62.5],
                                        [87.5, 62.5, 87.5],
                                        [87.5, 87.5, 62.5],
                                        [87.5, 87.5, 87.5]]))
        
    def test_landlockedbysum(self):
        bounds = (0, 100)
        # hrects in hrects2d 
        eligible = np.arange(3)
        # when all rectangles are members, all eligible hrects are landlocked
        self.assertTrue(landlocked_bysum(list(self.hrects2d[eligible]), list(self.hrects2d), bounds).all())
        # specific rectangles are landlocked
        self.assertArrayEqual(landlocked_bysum(list(self.hrects2d[eligible]), list(self.hrects2d[eligible]), bounds), 
                              np.array([True, False, False]))
        
        # hrects in hrects3d_4 which are children of hrects3d_1[0]
        eligible8 = np.arange(8)
        # when all rectangles are members, all eligible hrects are landlocked
        self.assertTrue(landlocked_bysum(list(self.hrects3d_4[eligible8]), list(self.hrects3d_4), bounds).all())
        # specific rectangles are landlocked
        self.assertArrayEqual(landlocked_bysum(list(self.hrects3d_4[eligible8]), list(self.hrects3d_4[eligible8]), bounds), 
                              np.array([True, False, False, False, False, False, False, False]))
        
        #hrects in hrects3d_4 which are childen of hrects3d_1[0, 1] 
        eligible16 = np.arange(16) 
        self.assertTrue(landlocked_bysum(list(self.hrects3d_4[eligible16]), list(self.hrects3d_4), bounds).all())
        self.assertArrayEqual(landlocked_bysum(list(self.hrects3d_4[eligible16]), list(self.hrects3d_4[eligible16]), bounds), 
                              np.array([True, True, False, False, False, False, False, False, 
                                        True, True, False, False, False, False, False, False]))
        self.assertArrayEqual(landlocked_bysum(list(self.hrects3d_4[eligible8]), list(self.hrects3d_4[eligible16]), bounds), 
                              np.array([True, True, False, False, False, False, False, False]))

    def test_landlockedbycontra(self):
        # hrects in hrects2d 
        eligible = np.arange(3)
        non_members = np.arange(3,4)
        # when all rectangles are members, all eligible hrects are landlocked
        # self.assertTrue(landlocked_bycontra(list(self.hrects2d[eligible]), list()).all())
        # specific rectangles are landlocked
        self.assertArrayEqual(landlocked_bycontra(list(self.hrects2d[eligible]), list(self.hrects2d[non_members])), 
                              np.array([True, False, False]))
        
        # hrects in hrects3d_4 which are children of hrects3d_1[0]
        eligible8 = np.arange(8)
        non_members8 = np.arange(8,64)
        # when all rectangles are members, all eligible hrects are landlocked
        # self.assertTrue(landlocked_bycontra(list(self.hrects3d_4[eligible8]), list()).all())
        # specific rectangles are landlocked
        self.assertArrayEqual(landlocked_bycontra(list(self.hrects3d_4[eligible8]), list(self.hrects3d_4[non_members8])), 
                              np.array([True, False, False, False, False, False, False, False]))
        
        #hrects in hrects3d_4 which are childen of hrects3d_1[0, 1] 
        eligible16 = np.arange(16) 
        non_members16 = np.arange(16,64)

        # self.assertTrue(landlocked_bycontra(list(self.hrects3d_4[eligible16]), list()).all())
        self.assertArrayEqual(landlocked_bycontra(list(self.hrects3d_4[eligible16]), list(self.hrects3d_4[non_members16])), 
                              np.array([True, True, False, False, False, False, False, False, 
                                        True, True, False, False, False, False, False, False]))
        self.assertArrayEqual(landlocked_bycontra(list(self.hrects3d_4[eligible8]), list(self.hrects3d_4[non_members16])), 
                              np.array([True, True, False, False, False, False, False, False]))
        

class TestAuxiliaries(unittest.TestCase, CustomAssertions):
    def test_norm(self):
        # normalisation does nothing with true bounds 0-1 since normed bounds are also 0=1
        arr = np.arange(5)-2
        lb = np.zeros(5)
        ub = np.ones(5)
        self.assertArrayEqual(normalise(arr, lb, ub), arr)
        arr = np.random.rand(5) 
        self.assertArrayEqual(normalise(arr, lb, ub), arr)
        
        ub = 2*ub
        self.assertArrayEqual(normalise(arr, lb, ub), arr / 2)

        lb = -1*np.ones(5)
        self.assertArrayEqual(normalise(arr, lb, ub), (arr+1)/3)
        
        lb = np.arange(5)
        ub = 10*np.ones(5)
        arr = np.array([1, 3.7, 6, 7.9, 9.4])
        self.assertArrayEqual(normalise(arr, lb, ub), np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
        
        arr = np.array([8, 6.4, 5.2, 4.4, 4])
        self.assertArrayEqual(normalise(arr, lb, ub), np.array([0.8, 0.6, 0.4, 0.2, 0]))

    def test_unnorm_c(self):
        # normalisation does nothing with true bounds 0-1 since normed bounds are also 0=1
        lb = np.zeros(5)
        ub = np.ones(5)
        arr = np.random.rand(5) 
        self.assertArrayEqual(unnormalise_c(arr, lb, ub), arr)
        arr = np.arange(5)-1
        self.assertArrayEqual(unnormalise_c(arr, lb, ub), arr)
                
        lb = np.arange(5)
        ub = 10*np.ones(5)
        arr = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        self.assertArrayEqual(unnormalise_c(arr, lb, ub), np.array([1, 3.7, 6, 7.9, 9.4]))

        arr = np.array([0.8, 0.6, 0.4, 0.2, 0])
        self.assertArrayEqual(unnormalise_c(arr, lb, ub), np.array([8, 6.4, 5.2, 4.4, 4]))

    def test_unnorm_hl(self):
        # normalisation does nothing with true bounds 0-1 since normed bounds are also 0=1
        lb = np.zeros(5)
        ub = np.ones(5)
        arr = np.random.rand(5) 
        self.assertArrayEqual(unnormalise_hl(arr, lb, ub), arr)
        arr = np.arange(5)-1
        self.assertArrayEqual(unnormalise_hl(arr, lb, ub), arr)
                
        lb = np.arange(5)
        ub = 10*np.ones(5)
        arr = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        self.assertArrayEqual(unnormalise_hl(arr, lb, ub), np.array([1, 2.7, 4, 4.9, 5.4]))

        arr = np.array([0.8, 0.6, 0.4, 0.2, 0])
        self.assertArrayEqual(unnormalise_hl(arr, lb, ub), np.array([8, 5.4, 3.2, 1.4, 0]))


    def test_norm_interactions(self):
        ub = np.random.rand(10)*10
        lb = np.random.rand(10)
        arr = np.random.rand(10)*5
        
        #normalise is inverse of unnormalise_c and vice versa
        self.assertArrayEqual(normalise(unnormalise_c(arr, lb, ub), lb, ub), arr)
        self.assertArrayEqual(unnormalise_c(normalise(arr, lb, ub), lb, ub), arr)
        
        #normalise is inverse of unnormlise_hl - offset and vice versa
        self.assertArrayEqual(unnormalise_hl(normalise(arr, lb, ub), lb, ub), arr-lb)
        self.assertArrayEqual(normalise(unnormalise_hl(arr, lb, ub), lb, ub), arr-lb/(ub-lb))

    def test_bool_matrix(self):
        # negative is not allowed
        self.assertRaises(Exception, generate_boolmatrix, -1)
        # 0 is empty
        self.assertArrayEmpty(generate_boolmatrix(0))
        # we can check 1 exactly 
        bm = generate_boolmatrix(1)
        self.assertTrue((bm==np.array([[True], [False]])).all() or 
                        (bm==np.array([[False], [True]])).all())
        
        # check some features of larger dimensionnal ones since we can't easily
        # check them directly
        for ndim in (2, 4, 8):
            bm = generate_boolmatrix(ndim)
            
            #expected shape
            self.assertEqual(bm.shape[0], 2**ndim)
            self.assertEqual(bm.shape[1], ndim)
            
            # one row of all True and one row of all False
            self.assertEqual(bm.all(axis=1).sum(), 1)
            self.assertEqual((~bm).all(axis=1).sum(), 1)
            
            # equal numbers of True and False
            self.assertEqual(bm.sum(), bm.size/2)
            
            # equal numbers of True and False in each column
            self.assertArrayEqual(bm.sum(axis=0), (2**(ndim-1))*np.ones(ndim)) 
            
            # expected distribution of True/False sums of each row
            self.assertArrayEqual(np.unique(bm.sum(axis=1), return_counts=True)[1],
                                  np.diagonal(np.fliplr(pascal(ndim+1))))
        
    def test_factor2(self):
        """This is a rather ridiculous function which is nevertheless vital to 
        the restart ability (specifically being able to calculate the 
        half_length of a hyper rectangle given only the centre and the box 
        bounds of the whole problem).
        The function should return the highest power of 2 which is a factor 
        of the argument supplied (AKA the ruler function)"""
                
        for i in range(1, 16):
            n=2**i
            self.assertEqual(factor2(n), n, msg=f"n={n}")
            self.assertEqual(factor2(n+1), 1, msg=f"n={n}")
            self.assertEqual(factor2(n-1), 1, msg=f"n={n}")
        
        self.assertEqual(factor2(6), 2)
        self.assertEqual(factor2(12), 4)
        self.assertEqual(factor2(24), 8)
        self.assertEqual(factor2(48), 16)
    
    def test_find_bool_indx(self):
        fs = np.zeros(10, dtype=bool)
        ts = np.ones(10, dtype=bool)
        arr = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
        
        # function only valid for n >= 1
        # function returns whole array when cannot find the full count of Trues
        for n in (1, 5, 10, np.inf):
            self.assertArrayEqual(fs[:find_bool_indx(fs, n)], fs, msg=f"n={n}")
        
        for n in range(1, 11):
            self.assertArrayEqual(ts[:find_bool_indx(ts, n)], np.ones(n, dtype=bool), msg=f"n={n}")
        
        for n in range(1, 6):
            self.assertArrayEqual(arr[:find_bool_indx(arr, n)], arr[:min(2*n, 10)], msg=f"n={n}")
                
    
    
if __name__ == '__main__':
    unittest.main()