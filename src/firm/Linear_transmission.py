"""
This module is not used in the optimisation.
It is used to verify that Fill.py works as intended
    i.e. provides a good approximation of a perfect Fill
"""

from datetime import datetime as dt

import numpy as np
import pyomo.environ as pyo

# import cplex


def build_model(solution):

    b_net = np.array(
        [
            [0, 3],  # FNQ-QLD
            [1, 3],  # NSW-QLD
            [1, 4],  # NSW-SA
            [1, 6],  # NSW-VIC
            [2, 4],  # NT-SA
            [4, 7],  # SA-WA
            [5, 6],  # TAS-VIC
        ],
        dtype=np.int64,
    )[solution.network_mask]

    pos_export_lines = [np.where(b_net[:, 0] == n)[0] + 1 for n in solution.Nodel_int]  # pyomo uses 1-indexing
    neg_export_lines = [np.where(b_net[:, 1] == n)[0] + 1 for n in solution.Nodel_int]  # pyomo uses 1-indexing

    print("Instantiating optimiser:", dt.now())
    model = pyo.ConcreteModel()

    # GCHydro = solution.CHydro.sum()
    # GCBio = solution.CBio.sum()

    model.t = pyo.RangeSet(solution.intervals)
    model.n = pyo.RangeSet(solution.nodes)
    model.l = pyo.RangeSet(solution.nhvdc)

    model.charge = pyo.Var(model.t, model.n, domain=pyo.NonNegativeReals)
    model.discharge = pyo.Var(model.t, model.n, domain=pyo.NonNegativeReals)
    model.storage = pyo.Var(model.t, model.n, domain=pyo.NonNegativeReals)
    model.deficit = pyo.Var(model.t, model.n, domain=pyo.NonNegativeReals)
    # model.hydro =   pyo.Var(model.t, model.n, domain=pyo.NonNegativeReals)
    # model.bio =     pyo.Var(model.t, model.n, domain=pyo.NonNegativeReals)

    model.hvdc_pos = pyo.Var(model.t, model.l, domain=pyo.NonNegativeReals)
    model.hvdc_neg = pyo.Var(model.t, model.l, domain=pyo.NonNegativeReals)

    model.constr_charge_power_upper = pyo.Constraint(
        model.t, model.n, rule=lambda m, t, n: m.charge[t, n] <= solution.CPHP[n - 1]
    )
    # model.constr_charge_power_lower = pyo.Constraint(model.t, model.n, rule=lambda m, t, n: m.charge[t, n] >= 0)
    model.constr_discharge_power_upper = pyo.Constraint(
        model.t, model.n, rule=lambda m, t, n: m.discharge[t, n] <= solution.CPHP[n - 1]
    )
    # model.constr_discharge_power_lower = pyo.Constraint(model.t, model.n, rule=lambda m, t, n: m.discharge[t, n] >= 0)
    model.constr_storage_energy_upper = pyo.Constraint(
        model.t, model.n, rule=lambda m, t, n: m.storage[t, n] <= solution.CPHS[n - 1]
    )
    # model.constr_storage_energy_lower = pyo.Constraint(model.t, model.n, rule=lambda m, t, n: m.storage[t, n] >= 0)

    # model.constr_hydro_power_upper = pyo.Constraint(model.t, rule=lambda m, t: m.hydro[t] <= GCHydro)
    # model.constr_hydro_power_lower = pyo.Constraint(model.t, rule=lambda m, t: m.hydro[t] >= 0)
    # model.constr_bio_power_upper = pyo.Constraint(model.t, rule=lambda m, t: m.bio[t] <= GCBio)
    # model.constr_bio_power_lower = pyo.Constraint(model.t, rule=lambda m, t: m.bio[t] >= 0)

    # model.constr_max_hydro = pyo.Constraint(rule=lambda m: pyo.summation(m.hydro)*solution.resolution/solution.years <= solution.Hydro_res)
    # model.constr_max_bio   = pyo.Constraint(rule=lambda m: pyo.summation(m.bio)*solution.resolution/solution.years <= solution.Bio_res)

    model.constr_hvdc_power_import = pyo.Constraint(
        model.t, model.l, rule=lambda m, t, l: m.hvdc_pos[t, l] <= solution.CHVDC[l - 1]
    )
    model.constr_hvdc_power_export = pyo.Constraint(
        model.t, model.l, rule=lambda m, t, l: m.hvdc_neg[t, l] <= solution.CHVDC[l - 1]
    )

    def constr_state_of_charge(m, t, n):
        if t == 1:
            return (
                m.storage[t, n]
                == 0.5 * solution.CPHS[n - 1]
                - m.discharge[t, n] * solution.resolution
                + m.charge[t, n] * solution.resolution * solution.efficiency
            )
        else:
            return (
                m.storage[t, n]
                == m.storage[t - 1, n]
                - m.discharge[t, n] * solution.resolution
                + m.charge[t, n] * solution.resolution * solution.efficiency
            )

    model.constr_storage_state_of_charge = pyo.Constraint(model.t, model.n, rule=constr_state_of_charge)

    def expr_energy_balance(m, t, n):
        return (
            solution.MLoad[t - 1, n - 1].sum()
            + m.charge[t, n]
            - solution.MPV[t - 1, n - 1]
            - solution.MWind[t - 1, n - 1]
            - solution.CBaseload[n - 1]
            # - m.hydro[t]
            # - m.bio[t]
            - m.discharge[t, n]
            + sum((m.hvdc_pos[t, l] - m.hvdc_neg[t, l] for l in pos_export_lines[n - 1]))
            + sum((m.hvdc_neg[t, l] - m.hvdc_pos[t, l] for l in neg_export_lines[n - 1]))
            - m.deficit[t, n]
        )

    model.energy_balance = pyo.Expression(model.t, model.n, rule=expr_energy_balance)
    model.constr_energy_balance = pyo.Constraint(model.t, model.n, rule=lambda m, t, n: m.energy_balance[t, n] <= 0)

    model.obj = pyo.Objective(rule=lambda m: pyo.summation(m.deficit))
    # model.obj = pyo.Objective(rule=lambda m: pyo.summation(m.gas))

    return model


def optimise_model(model):
    start = dt.now()
    print("Optimising. Start:", start)
    optimiser = pyo.SolverFactory("gurobi")
    optimiser.solve(model)
    end = dt.now()
    print("Optimisation took:", end - start)
    return model


if __name__ == "__main__":
    from firm.Input import Solution, x0

    solution = Solution(x0)
    model = build_model(solution)
    model = optimise_model(model)
