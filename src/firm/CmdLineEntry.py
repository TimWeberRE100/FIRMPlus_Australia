import click
from firm.Parameters import Parameters, DE_Hyperparameters
from rich.console import Console # type: ignore
from rich.text import Text # type: ignore
from rich.table import Table # type: ignore
from psutil import cpu_count
from time import perf_counter


@click.command
@click.option(
    "-s",
    "--scenario",
    default=21,
    type=click.IntRange(0),
    required=False,
    show_default=True,
    help="Scenario to run",
)
def statistics(
    scenario: int 
    ):
    import numpy as np 
    try: 
        x = np.genfromtxt(f"Results/Optimisation_resultx{scenario}.csv", 
                          delimiter=",")
    except FileNotFoundError as e:
        print("No solution found. Run optimisation first.")
        raise e 
    from firm.Statistics import Information
    Information(x)
    
    
@click.command
@click.option(
    "-s",
    "--scenario",
    default=21,
    type=click.IntRange(0),
    required=False,
    show_default=True,
    help="Scenario to run",
)
@click.option(
    "-y", 
    "--years", 
    default=-1,
    type=click.IntRange(-1), 
    required=False,   
    show_default=True, 
    help="no. of years to model",
)
@click.option(
    "-n", 
    "--number", 
    default=3, 
    type=click.IntRange(1),
    show_default=True, 
    required=False, 
    help="How many batches",
    )
@click.option(
    "-e", 
    "--evals", 
    default=cpu_count(True)*3, 
    type=click.IntRange(1),
    show_default=True, 
    required=False, 
    help="How many evaluations per batch",
    )
def benchmark(
    scenario: int,
    years: int,
    number: int, 
    evals: int, 
):
    
    print("Running Benchmarking...", end="")
    from firm.Benchmark import Benchmark
    from firm.Input import Solution_data
    from firm.Costs import Raw_Costs
    
    parameters = Parameters(scenario, years, False)
    sd = Solution_data(*parameters)
    cost_model = Raw_Costs(sd).CostFactors()
    #compile
    Benchmark(2, sd, cost_model)
    start = perf_counter()
    for i in range(number): 
        Benchmark(evals, sd, cost_model)
    time = perf_counter() - start
    print(f"\rBenchmarking took {time/number} per parallel batch of {evals} ({time/number/evals} per eval).")

@click.command
@click.option(
    "-s",
    "--scenario",
    default=21,
    type=click.IntRange(0),
    required=False,
    show_default=True,
    help="Scenario to run",
)
@click.option(
    "-y", 
    "--years", 
    default=-1,
    type=click.IntRange(-1), 
    required=False,   
    show_default=True, 
    help="no. of years to model",
)
def profile(
    scenario: int,
    years: int,
):
    #%%
    print("Running Profiling...", end="")
    from firm.Benchmark import profile
    from firm.Input import Solution_data
    from firm.Costs import Raw_Costs
    from firm.Utils import zero_safe_division
    parameters = Parameters(scenario, years, True)
    sd = Solution_data(*parameters)
    cost_model = Raw_Costs(sd).CostFactors()
    
    solution, time, ctwt = profile(sd.x0, sd, cost_model, False) # compile
    solution, time, ctwt = profile(sd.x0, sd, cost_model, False)
    ctwt = ctwt*1000 # seconds to microseconds
    print("\r", " "*25, "\r")
    print("""Warning: 
profile measures cpu-time. 
Table times are wall-time apportioned over cpu-cycles
Not all of the profiling overhead is accounted for. 
    empirically, unprofiled time is ~30% faster 
Profiling overhead is not evenly split between components
    higher level functions (e.g. Transmission) which call lower level funcs
    have higher proportions of overhead attached
          """)
    table = Table(title="Profile Results")
    
    table.add_column("Function")
    table.add_column("Calls")
    table.add_column("Cpu-cycles")
    table.add_column("Cpu-cycles per call")
    table.add_column("Apportioned Time (ms)")
    table.add_column("Time per call")

    calls_profile = sum((getattr(solution, item) for item in dir(solution) if item.startswith('calls_')))
    time_profile = calls_profile * solution.profile_overhead

    table.add_row(
        "Profiler overhead", 
        str(calls_profile), 
        str(time_profile), 
        str(solution.profile_overhead), 
        str(ctwt*time_profile), 
        str(ctwt*solution.profile_overhead), 
        )    
    table.add_row(
        "Transmission", 
        str(solution.calls_transmission), 
        str(solution.time_transmission), 
        str(zero_safe_division(
            solution.time_transmission,
            solution.calls_transmission)), 
        str(ctwt*solution.time_transmission), 
        str(ctwt*zero_safe_division(
            solution.time_transmission,
            solution.calls_transmission)), 
        )
    table.add_row(
        "Flexible",
        str(solution.calls_backfill), 
        str(solution.time_backfill), 
        str(zero_safe_division(
            solution.time_backfill,
            solution.calls_backfill)),
        str(ctwt*solution.time_backfill), 
        str(ctwt*zero_safe_division(
            solution.time_backfill,
            solution.calls_backfill)),
        )
    # table.add_row(
    #     "Basic Sim",
    #     str(solution.calls_basic), 
    #     str(solution.time_basic), 
    #     str(zero_safe_division(
    #         solution.time_basic,
    #         solution.calls_basic)), 
    #     str(ctwt*solution.time_basic), 
    #     str(ctwt*zero_safe_division(
    #         solution.time_basic,
    #         solution.calls_basic)), 
    #     )
    table.add_row(
        "Interconnection0",
        str(solution.calls_interconnection0),
        str(solution.time_interconnection0),
        str(zero_safe_division(
            solution.time_interconnection0, 
            solution.calls_interconnection0)),
        str(ctwt*solution.time_interconnection0),
        str(ctwt*zero_safe_division(
            solution.time_interconnection0, 
            solution.calls_interconnection0)),
        )
    table.add_row(
        "Interconnection1",
        str(solution.calls_interconnection1),
        str(solution.time_interconnection1),
        str(zero_safe_division(
            solution.time_interconnection1,
            solution.calls_interconnection1)),
        str(ctwt*solution.time_interconnection1),
        str(ctwt*zero_safe_division(
            solution.time_interconnection1,
            solution.calls_interconnection1)),
        )
    table.add_row(
        "Interconnection2",
        str(solution.calls_interconnection2),
        str(solution.time_interconnection2),
        str(zero_safe_division(
            solution.time_interconnection2,
            solution.calls_interconnection2)),
        str(ctwt*solution.time_interconnection2),
        str(ctwt*zero_safe_division(
            solution.time_interconnection2,
            solution.calls_interconnection2)),
        )
    table.add_row(
        "Interconnection3",
        str(solution.calls_interconnection3),
        str(solution.time_interconnection3),
        str(zero_safe_division(
            solution.time_interconnection3,
            solution.calls_interconnection3)),
        str(ctwt*solution.time_interconnection3),
        str(ctwt*zero_safe_division(
            solution.time_interconnection3,
            solution.calls_interconnection3)),
        )
    table.add_row(
        "storage behav",
        str(solution.calls_storage_behavior), 
        str(solution.time_storage_behavior), 
        str(zero_safe_division(
            solution.time_storage_behavior,
            solution.calls_storage_behavior)), 
        str(ctwt*solution.time_storage_behavior), 
        str(ctwt*zero_safe_division(
            solution.time_storage_behavior,
            solution.calls_storage_behavior)), 
        )
    table.add_row(
        "storage behav t",
        str(solution.calls_storage_behaviort), 
        str(solution.time_storage_behaviort), 
        str(zero_safe_division(
            solution.time_storage_behaviort,
            solution.calls_storage_behaviort)), 
        str(ctwt*solution.time_storage_behaviort), 
        str(ctwt*zero_safe_division(
            solution.time_storage_behaviort,
            solution.calls_storage_behaviort)), 
        )
    table.add_row(
        "spill/def",
        str(solution.calls_spilldef), 
        str(solution.time_spilldef), 
        str(zero_safe_division(
            solution.time_spilldef,
            solution.calls_spilldef)), 
        str(ctwt*solution.time_spilldef), 
        str(ctwt*zero_safe_division(
            solution.time_spilldef,
            solution.calls_spilldef)), 
        )
    table.add_row(
        "spill/def t",
        str(solution.calls_spilldeft), 
        str(solution.time_spilldeft), 
        str(zero_safe_division(
            solution.time_spilldeft,
            solution.calls_spilldeft)), 
        str(ctwt*solution.time_spilldeft), 
        str(ctwt*zero_safe_division(
            solution.time_spilldeft,
            solution.calls_spilldeft)), 
        )
    table.add_row(
        "soc",
        str(solution.calls_update_soc),
        str(solution.time_update_soc),
        str(zero_safe_division(
            solution.time_update_soc,
            solution.calls_update_soc)),
        str(ctwt*solution.time_update_soc),
        str(ctwt*zero_safe_division(
            solution.time_update_soc,
            solution.calls_update_soc)),
        )
    table.add_row(
        "soc t",
        str(solution.calls_update_soct), 
        str(solution.time_update_soct), 
        str(zero_safe_division(
            solution.time_update_soct,
            solution.calls_update_soct)), 
        str(ctwt*solution.time_update_soct), 
        str(ctwt*zero_safe_division(
            solution.time_update_soct,
            solution.calls_update_soct)), 
        )
    table.add_row(
        "unbalanced",
        str(solution.calls_unbalanced), 
        str(solution.time_unbalanced), 
        str(zero_safe_division(
            solution.time_unbalanced,
            solution.calls_unbalanced)), 
        str(ctwt*solution.time_unbalanced), 
        str(ctwt*zero_safe_division(
            solution.time_unbalanced,
            solution.calls_unbalanced)), 
        )
    table.add_row(
        "unbalanced t",
        str(solution.calls_unbalancedt),
        str(solution.time_unbalancedt),
        str(zero_safe_division(
            solution.time_unbalancedt,
            solution.calls_unbalancedt)),
        str(ctwt*solution.time_unbalancedt),
        str(ctwt*zero_safe_division(
            solution.time_unbalancedt,
            solution.calls_unbalancedt)),
        )
    
    print("\r", " "*30, sep='', end="")
    results = Table(title="Solution Result")
    results.add_column("")
    results.add_column("Value")
    results.add_row(
        "LCOE",
        str(solution.LCOE),
        )
    results.add_row(
        "Penalties",
        str(solution.Penalties),
        )
    results.add_row(
        "Wall-time", 
        str(time),
    )


    console = Console()
    console.print(table)
    console.print(results)

#%%

@click.command
@click.option(
    "-s",
    "--scenario",
    default=21,
    type=click.IntRange(0),
    required=False,
    show_default=True,
    help="Scenario to run",
)
@click.option(
    "-y",
    "--years",
    default=1,
    type=click.IntRange(-1, 10, clamp=True),
    required=False,
    show_default=True,
    help="No. of years to simulate. -1 indicates max",
)
@click.option(
    "-i",
    "--iterations",
    default=1000,
    show_default=True,
    type=click.IntRange(1, 4000),
    required=False,
    help="Maximum iterations",
)
@click.option(
    "-p",
    "--popsize",
    default=50,
    show_default=True,
    type=click.IntRange(2, 1000),
    required=False,
    help="Population size",
)
@click.option(
    "-m", 
    "--mutation", 
    default=None, 
    show_default=True, 
    type=click.FloatRange(0., 2.), 
    required=False, 
    help="Mutation factor"
)
@click.option(
    "-d", 
    "--dither", 
    nargs=2,
    default=(None, None), 
    type=click.Tuple([click.FloatRange(0., 2.), click.FloatRange(0., 2.)]),
    required=False, 
    help="Mutation factor dither range (overrides --mutation)"
)
@click.option(
    "-r", 
    "--recombination", 
    default=0.4, 
    show_default=True, 
    type=click.FloatRange(0., 1.), 
    required=False, 
    help="Recombination factor"
)
@click.option(
    "-v",
    "--progress",
    default=1,
    type=click.IntRange(0),
    show_default=True,
    required=False,
    help="Print progress to console",
)
@click.option(
    "-e", 
    "--stagnation", 
    default=(5, 0.1),
    type=click.Tuple([click.IntRange(0), click.FloatRange(0)]), 
    show_default=True, 
    required=False, 
    help="Stagnation condition: (no. of its, minimum improvement)"
    )
@click.option(
    "-f", 
    "--fileprint", 
    default=1,
    type=click.IntRange(0), 
    show_default=True, 
    required=False, 
    help="Frequency to print to file"
    )
def optimise(
    scenario: int,
    years: int,
    iterations: int,
    popsize: int,
    mutation: float,
    dither: tuple[float, float],
    recombination: float,
    progress: int,
    stagnation: tuple[int, float],
    fileprint: int,
):
    
    param = Parameters(
        s = scenario,
        y = years,
        p = False,
    )
    hyperparam = DE_Hyperparameters(
        i = iterations, 
        p = popsize, 
        m = (mutation if mutation is not None else 
             dither if dither[0] is not None and dither[1] is not None else 
             (0.5, 1.0)),
        r = recombination,
        v = progress,
        s = stagnation,
        f = fileprint,
        )
    
    from firm.Input import Solution_data
    from firm.Optimisation import Optimise
    
    solution_data = Solution_data(*param)
  
    result, time = Optimise(solution_data, hyperparam)
    # print(result.x)

@click.command
@click.option(
    "-s",
    "--scenario",
    default=21,
    type=click.IntRange(0),
    required=False,
    show_default=True,
    help="Scenario to run",
)
@click.option(
    "-y",
    "--years",
    default=1,
    type=click.IntRange(-1, 10, clamp=True),
    required=False,
    show_default=True,
    help="No. of years to simulate. -1 indicates max",
)
@click.option(
    "-i",
    "--iterations",
    default=1000,
    show_default=True,
    type=click.IntRange(1, 4000),
    required=False,
    help="Maximum iterations",
)
@click.option(
    "-p",
    "--popsize",
    default=50,
    show_default=True,
    type=click.IntRange(2, 1000),
    required=False,
    help="Population size",
)
@click.option(
    "-m", 
    "--mutation", 
    default=None, 
    show_default=True, 
    type=click.FloatRange(0., 2.), 
    required=False, 
    help="Mutation factor"
    )
@click.option(
    "-d", 
    "--dither", 
    nargs=2,
    default=(None, None), 
    type=click.Tuple([click.FloatRange(0., 2.), click.FloatRange(0., 2.)]),
    required=False, 
    help="Mutation factor dither range (overrides --mutation)"
    )
@click.option(
    "-r", 
    "--recombination", 
    default=0.4, 
    show_default=True, 
    type=click.FloatRange(0., 1.), 
    required=False, 
    help="Recombination factor"
    )
@click.option(
    "-v",
    "--progress",
    default=1,
    type=click.IntRange(0),
    show_default=True,
    required=False,
    help="Print progress to console",
)
@click.option(
    "-e", 
    "--stagnation", 
    default=(5, 0.1),
    type=click.Tuple([click.IntRange(0), click.FloatRange(0)]), 
    show_default=True, 
    required=False, 
    help="Stagnation condition: (no. of its, minimum improvement)"
    )
@click.option(
    "-f", 
    "--fileprint", 
    default=1,
    type=click.IntRange(0), 
    show_default=True, 
    required=False, 
    help="Frequency to print to file"
    )
def polish(
    scenario: int,
    years: int,
    iterations: int,
    popsize: int,
    mutation: float,
    dither: tuple[float, float],
    recombination: float,
    progress: int,
    stagnation: tuple[int, float],
    fileprint: int,
):
    import numpy as np 
    try: 
        x0 = np.genfromtxt(f"Results/Optimisation_resultx{scenario}.csv", 
                          delimiter=",")
    except FileNotFoundError as e:
        print("No solution found. Run optimisation first.")
        raise e 
    param = Parameters(
        s = scenario,
        y = years,
        p = False,
    )
    hyperparam = DE_Hyperparameters(
        i = iterations, 
        p = popsize, 
        m = (mutation if mutation is not None else 
             dither if dither[0] is not None and dither[1] is not None else 
             (0.5, 1.0)),
        r = recombination,
        v = progress,
        s = stagnation,
        f = fileprint,
        )
    
    from firm.Input import Solution_data
    from firm.Optimisation import Polish
    
    solution_data = Solution_data(*param)
    
    result, time = Polish(x0, solution_data, hyperparam)
    print(result.x)

@click.command
def info():
    console = Console()
    text = Text(
        r"""
 _____ ___ ____  __  __ ____  _           
|  ___|_ _|  _ \|  \/  |  _ \| |_   _ ___ 
| |_   | || |_) | |\/| | |_) | | | | / __|
|  _|  | ||  _ <| |  | |  __/| | |_| \__ \
|_|   |___|_| \_\_|  |_|_|   |_|\__,_|___/
"""
    )
    console.print(text, style="cornflower_blue")
    version_string = "Version 0.0.1"
    console.print(version_string, style="cornflower_blue")

@click.group
def Entry():
    pass

Entry.add_command(benchmark)
Entry.add_command(profile)
Entry.add_command(optimise)
Entry.add_command(polish)
Entry.add_command(statistics)
Entry.add_command(info)

