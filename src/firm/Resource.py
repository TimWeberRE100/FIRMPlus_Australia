# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 16:34:32 2025

@author: u6942852
"""
import numpy as np

pv_resource=np.array([
    6.385, # N1
    2.985, # N2
    6.85, # N3 
    8, # N4  
    2.256, # N5  
    1.028, # N6  
    0, # N7  
    0, # N8  
    0.516, # N9  
    1.1, # Q1  
    8, # Q2  
    3.4, # Q3  
    6.9, # Q4  
    8, # Q5  
    7.533, # Q6  
    2.2, # Q7  
    6.992, # Q8  
    6.1, # Q9  
    0.1, # S1  
    4, # S2  
    1.3, # S3  
    0, # S4  
    2.9, # S5  
    6.5, # S6  
    3.4, # S7  
    5, # S8  
    4, # S9  
    0.3, # T1  
    0.15, # T2  
    0.15, # T3  
    1, # V1  
    4.7, # V2  
    0.4, # V3  
    0, # V4  
    0.5, # V5  
    1.7, # V6  
    ], np.float64)



onw_resource=np.array([
    0, # N1
    7.4, # N2
    3, # N3 
    5.1, # N4  
    3.9, # N5  
    1, # N6  
    0, # N7  
    0.3, # N8  
    1.4, # N9  
    2.28, # Q1  
    18.6, # Q2  
    0, # Q3  
    3.8, # Q4  
    3.9, # Q5  
    3.5, # Q6  
    1.1, # Q7  
    5.6, # Q8  
    3.4, # Q9  
    3.2, # S1  
    1.4, # S2  
    4.6, # S3  
    1.4, # S4  
    0, # S5  
    2.4, # S6  
    0, # S7  
    2.3, # S8  
    1.5, # S9  
    1.4, # T1  
    5, # T2  
    3.4, # T3  
    0, # V1  
    0, # V2  
    2.6, # V3  
    3.442, # V4  
    2, # V5  
    1.6, # V6  
    ], np.float64)

onw_quality=np.array([
    5, # N1
    3, # N2
    3, # N3 
    4, # N4  
    5, # N5  
    5, # N6  
    5, # N7  
    2, # N8  
    4, # N9  
    1, # Q1  
    2, # Q2  
    5, # Q3  
    4, # Q4  
    4, # Q5  
    3, # Q6  
    5, # Q7  
    3, # Q8  
    5, # Q9  
    3, # S1  
    5, # S2  
    3, # S3  
    3, # S4  
    5, # S5  
    2, # S6  
    5, # S7  
    3, # S8  
    3, # S9  
    2, # T1  
    1, # T2  
    1, # T3  
    5, # V1  
    5, # V2  
    2, # V3  
    3, # V4  
    3, # V5  
    4, # V6  
    ], np.int64)

offwfl_resource=np.array([
    7.42, # N10 # class B
    5.696, # N11 # class B
    0, # N12  
    7.032, # S10 # class A  
    26.15, # T4 # class A
    0, # T5  
    5, # V7 # class A
    3.33, # V8 # class A
    ], np.float64)

offwfx_resource=np.array([
    0, # N10 # class E
    0.148, # N11 # class B
    0, # N12  
    20.428, # S10 # class A 
    14.4, # T4 # class A
    0, # T5  
    54.996, # V7 # class A
    0.78, # V8 # class A
    ], np.float64)