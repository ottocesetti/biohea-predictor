# ============================================================
# FILE: thermodynamics/thermocalc_engine.py
# ============================================================

import json
import hashlib
from pathlib import Path
import numpy as np


class ThermoCalcEngine:

    FRAMEWORK_VERSION = "18.2"

    def __init__(
        self,
        cache_dir,
        database="TCHEA5",
        database_version="unknown",
        thermocalc_version="unknown",
        equilibrium_mode="equilibrium",
        metastable=False,
        suspended_phases=None,
        global_minimization=True,
        solver_options=None,
        cache_failures=False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.database = database
        self.database_version = database_version
        self.thermocalc_version = thermocalc_version
        self.equilibrium_mode = equilibrium_mode
        self.metastable = metastable
        self.suspended_phases = suspended_phases or []
        self.global_minimization = global_minimization
        self.solver_options = solver_options or {}
        self.cache_failures = cache_failures

    def _canonical_composition(self, composition_dict):
        total = sum(float(v) for v in composition_dict.values())
        if total <= 0:
            raise ValueError("Invalid composition: total fraction <= 0.")

        return {k: float(v) / total for k, v in sorted(composition_dict.items())}

    def _make_key(self, composition_dict, temperature, pressure):
        key_dict = {
            "composition": self._canonical_composition(composition_dict),
            "temperature": float(temperature),
            "pressure": float(pressure),
            "database": self.database,
            "database_version": self.database_version,
            "thermocalc_version": self.thermocalc_version,
            "equilibrium_mode": self.equilibrium_mode,
            "metastable": self.metastable,
            "suspended_phases": sorted(self.suspended_phases),
            "global_minimization": self.global_minimization,
            "solver_options": self.solver_options,
            "framework_version": self.FRAMEWORK_VERSION,
        }
        key_json = json.dumps(key_dict, sort_keys=True)
        return hashlib.sha256(key_json.encode()).hexdigest()

    def get_cache_path(self, key):
        return self.cache_dir / f"{key}.json"

    def _classify_failure(self, error):
        msg = str(error).lower()
        if "timeout" in msg:
            return "timeout"
        if "conver" in msg or "diverg" in msg or "minim" in msg:
            return "solver_convergence"
        if "infeasible" in msg or "no equilibrium" in msg or "condition" in msg:
            return "infeasible_equilibrium"
        if "license" in msg or "tc_python" in msg or "import" in msg:
            return "license_or_import"
        return "unknown_failure"

    def _metadata(self, temperature, pressure):
        return {
            "temperature": float(temperature),
            "pressure": float(pressure),
            "database": self.database,
            "database_version": self.database_version,
            "thermocalc_version": self.thermocalc_version,
            "equilibrium_mode": self.equilibrium_mode,
            "metastable": self.metastable,
            "suspended_phases": self.suspended_phases,
            "global_minimization": self.global_minimization,
            "solver_options": self.solver_options,
            "framework_version": self.FRAMEWORK_VERSION,
        }

    def _failure_output(self, error, temperature, pressure):
        return {
            "GM": np.nan,
            "HM": np.nan,
            "SM": np.nan,
            "stable_phases": [],
            "phase_fractions": {},
            "status": "failure",
            "failure_type": self._classify_failure(error),
            "failure_message": str(error),
            "metadata": self._metadata(temperature, pressure),
        }

    def calculate(self, composition_dict, temperature=1200, pressure=101325):
        composition_dict = self._canonical_composition(composition_dict)
        key = self._make_key(composition_dict, temperature, pressure)
        cache_path = self.get_cache_path(key)

        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        try:
            from tc_python import TCPython

            with TCPython() as start:
                system = (
                    start.select_database_and_elements(
                        self.database, list(composition_dict.keys())
                    ).get_system()
                )
                calculation = system.with_single_equilibrium_calculation()

                elements = list(composition_dict.keys())
                for el in elements[:-1]:
                    calculation.set_condition(f"X({el})", float(composition_dict[el]))

                calculation.set_condition("T", float(temperature))
                calculation.set_condition("P", float(pressure))

                result = calculation.calculate()
                stable_phases = result.get_stable_phases()
                phase_fractions = {}

                for phase in stable_phases:
                    phase_fractions[phase] = float(result.get_value_of(f"NP({phase})"))

                output = {
                    "GM": float(result.get_value_of("GM")),
                    "HM": float(result.get_value_of("HM")),
                    "SM": float(result.get_value_of("SM")),
                    "stable_phases": stable_phases,
                    "phase_fractions": phase_fractions,
                    "status": "success",
                    "failure_type": "none",
                    "failure_message": "",
                    "metadata": self._metadata(temperature, pressure),
                }

        except Exception as e:
            output = self._failure_output(e, temperature, pressure)

        if output["status"] == "success" or self.cache_failures:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)

        return output
