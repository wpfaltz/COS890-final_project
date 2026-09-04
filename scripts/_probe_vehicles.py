"""Throwaway probe: how long does Gurobi's own MIP need to find/prove the minimal
fleet size for c102 (100 customers) under the compact formulation? Not part of the
package; delete after use.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrptw.cuts import fix_infeasible_arcs
from vrptw.formulation import build_model
from vrptw.instance import load_solomon_instance

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "solomon" / "raw"

inst = load_solomon_instance(RAW_DIR / "c102.txt", num_customers=100)
model, v = build_model(inst, fixed_vehicles=None, objective="vehicles")
fix_infeasible_arcs(inst, v["x"], model)
model.Params.OutputFlag = 1
model.Params.TimeLimit = 300.0
model.Params.MIPGap = 0.0
t0 = time.time()
model.optimize()
print(f"status={model.Status} solcount={model.SolCount} objval={model.ObjVal if model.SolCount else None} "
      f"objbound={model.ObjBound} nodecount={model.NodeCount} elapsed={time.time()-t0:.1f}s")
