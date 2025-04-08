from dataclasses import dataclass

@dataclass
class Parameters:
    s: int  # scenario
    y: int  # years
    p: bool # profiling
    
    def __iter__(self):
        return iter((self.s, self.y, self.p))
    
    
@dataclass
class DE_Hyperparameters:
    i: int # iterations
    p: int # population
    m: float|tuple[float, float] # mutation
    r: float # recombination
    v: int # verbose
    s: tuple[int, float] # stagnation
    f: int # print to file
    

