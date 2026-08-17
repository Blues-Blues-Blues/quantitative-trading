"""超参数搜索空间定义（权重、阈值、窗口）。

参数组（与 strategy.signals / indicators.microstructure 的初始化入参一一对应）：

1. 权重（需归一化和为 1）：
   - W_OFSS [0.2, 0.6] / W_CPS [0.1, 0.5] / W_INST [0.0, 0.4] / W_NORTH [0.0, 0.3]
   采样策略：前三个权重独立 uniform 采样，W_NORTH = 1 - (W_OFSS+W_CPS+W_INST) 由
   归一化推导；W_NORTH 越出 [0, 0.3] 的 trial 由约束机制判为不可行 ——
   既不破坏 SignalSynthesizer 的「权重和 == 1」断言，又尊重各自范围。
2. 阈值：TH_MS_BULL / TH_MS_EXIT / TH_LOCK / TH_PURITY / TH_GLOBAL_MIN / TH_ADR_MIN
3. 窗口：WIN_INST（分钟，→ inst_window）/ WIN_CHIP_OLD（天，→ chip_window）/
   WIN_HOLD_MAX（分钟，→ win_hold_max）
"""

from typing import Dict, List, Optional, Sequence, Tuple

import optuna

# 权重参数名（顺序即 SignalSynthesizer.weights 的 (W_OFSS, W_CPS, W_INST, W_NORTH)）
WEIGHT_NAMES: Tuple[str, ...] = ("w_ofss", "w_cps", "w_inst", "w_north")


class SearchSpace:
    """Optuna 搜索空间：`suggest(trial)` 采样 → 代码层参数字典。"""

    def __init__(
        self,
        w_ofss: Tuple[float, float] = (0.2, 0.6),
        w_cps: Tuple[float, float] = (0.1, 0.5),
        w_inst: Tuple[float, float] = (0.0, 0.4),
        w_north: Tuple[float, float] = (0.0, 0.3),
        th_ms_bull: Tuple[float, float] = (0.3, 0.8),
        th_ms_exit: Tuple[float, float] = (-0.4, 0.0),
        th_lock: Tuple[float, float] = (0.4, 0.8),
        th_purity: Tuple[float, float] = (0.1, 0.5),
        th_global_min: Tuple[float, float] = (-0.8, -0.2),
        th_adr_min: Tuple[float, float] = (0.3, 0.7),
        win_inst: Tuple[int, int] = (1, 20),
        win_chip_old: Tuple[int, int] = (5, 20),
        win_hold_max: Tuple[int, int] = (10, 120),
    ) -> None:
        self.w_ofss, self.w_cps = w_ofss, w_cps
        self.w_inst, self.w_north = w_inst, w_north
        self.th_ms_bull, self.th_ms_exit = th_ms_bull, th_ms_exit
        self.th_lock, self.th_purity = th_lock, th_purity
        self.th_global_min, self.th_adr_min = th_global_min, th_adr_min
        self.win_inst, self.win_chip_old = win_inst, win_chip_old
        self.win_hold_max = win_hold_max

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------

    def suggest(self, trial: optuna.Trial) -> Dict[str, object]:
        """从 trial 采样一组参数，返回代码层参数字典。

        :return: 可直接解包给 SignalSynthesizer / MicroStructure / 回测引擎的字典
        """
        w_ofss = trial.suggest_float("w_ofss", *self.w_ofss)
        w_cps = trial.suggest_float("w_cps", *self.w_cps)
        w_inst = trial.suggest_float("w_inst", *self.w_inst)
        w_north = 1.0 - w_ofss - w_cps - w_inst  # 归一化：和恒为 1

        params: Dict[str, object] = {
            # 策略层（SignalSynthesizer）
            "weights": (w_ofss, w_cps, w_inst, w_north),
            "th_ms_bull": trial.suggest_float("th_ms_bull", *self.th_ms_bull),
            "th_ms_exit": trial.suggest_float("th_ms_exit", *self.th_ms_exit),
            "th_lock": trial.suggest_float("th_lock", *self.th_lock),
            "th_purity": trial.suggest_float("th_purity", *self.th_purity),
            "th_global_min": trial.suggest_float("th_global_min", *self.th_global_min),
            "th_adr_min": trial.suggest_float("th_adr_min", *self.th_adr_min),
            "win_hold_max": trial.suggest_int("win_hold_max", *self.win_hold_max),
            # 窗口：WIN_INST → inst_window；WIN_CHIP_OLD → chip_window（因子层）
            "inst_window": trial.suggest_int("win_inst", *self.win_inst),
            "chip_window": trial.suggest_int("win_chip_old", *self.win_chip_old),
        }
        trial.set_user_attr("w_north", w_north)
        return params

    # ------------------------------------------------------------------
    # 可行性
    # ------------------------------------------------------------------

    def weight_ranges(self) -> List[Tuple[float, float]]:
        """四维权重各自的范围（顺序同 WEIGHT_NAMES）。"""
        return [self.w_ofss, self.w_cps, self.w_inst, self.w_north]

    def is_feasible(self, params: Dict[str, object],
                    atol: float = 1e-9) -> bool:
        """权重是否同时满足「和为 1」与「各自范围」。"""
        w = tuple(float(x) for x in params["weights"])
        if not abs(sum(w) - 1.0) <= atol:
            return False
        for value, (lo, hi) in zip(w, self.weight_ranges()):
            if not (lo - atol <= value <= hi + atol):
                return False
        return True
