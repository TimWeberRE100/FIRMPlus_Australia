from dataclasses import dataclass

@dataclass
class Parameters:
    y: int  # years
    p: bool # profiling
    
@dataclass
class DE_Hyperparameters:
    i: int # iterations
    p: int # population
    m: float|tuple[float, float] # mutation
    r: float # recombination
    v: int # verbose
    s: tuple[int, float] # stagnation
    f: int # print to file
    

