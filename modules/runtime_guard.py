"""Runtime guards for Streamlit Cloud.

Keeps the expensive MLB model alive across Streamlit reruns and caps Monte Carlo
work so clicking the Analyze button does not retrain/allocate everything again.
"""

import copy

import modules.ml_mlb as ml_mod
import modules.montecarlo_mlb as mc_mod

_ORIGINAL_PREDICTOR = ml_mod.PredictorMLMLB
_ORIGINAL_MC = mc_mod.simular_partido_mlb


class CachedPredictor(_ORIGINAL_PREDICTOR):
    _cached_state = None
    _cached_key = None

    @staticmethod
    def _data_key(df_batting, df_pitching, df_games):
        def last_value(df, col):
            try:
                if df is not None and not df.empty and col in df.columns:
                    return str(df[col].iloc[-1])
            except Exception:
                pass
            return ""

        return (
            len(df_batting) if df_batting is not None else 0,
            len(df_pitching) if df_pitching is not None else 0,
            len(df_games) if df_games is not None else 0,
            last_value(df_batting, "Season"),
            last_value(df_pitching, "Season"),
            last_value(df_games, "Date"),
        )

    def entrenar(self, df_batting, df_pitching, df_games):
        key = self._data_key(df_batting, df_pitching, df_games)
        if self.__class__._cached_key == key and self.__class__._cached_state is not None:
            self.__dict__.update(copy.copy(self.__class__._cached_state))
            return bool(self.entrenado)

        ok = super().entrenar(df_batting, df_pitching, df_games)
        if ok:
            self.__class__._cached_key = key
            self.__class__._cached_state = copy.copy(self.__dict__)
        return ok


def capped_montecarlo(*args, **kwargs):
    requested = int(kwargs.get("num_simulaciones", 20000) or 20000)
    kwargs["num_simulaciones"] = min(requested, 20000)
    return _ORIGINAL_MC(*args, **kwargs)


def install_runtime_guards():
    ml_mod.PredictorMLMLB = CachedPredictor
    mc_mod.simular_partido_mlb = capped_montecarlo
