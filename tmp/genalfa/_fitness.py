from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ._alpha import _Alpha

__all__ = ["_Fitness"]


class _Fitness:
    """
    This class is used to calculate the fitness of the alpha.
    """

    def __init__(
        self,
        alpha: _Alpha,
        data: pd.DataFrame,
    ):
        if not isinstance(alpha, _Alpha):
            raise ValueError("alpha must be an instance of _Alpha")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")

        if not alpha.validate_alpha():
            raise ValueError("alpha is not valid")

        self._alpha: _Alpha = alpha
        self._data: pd.DataFrame = data
        self._expression: str = alpha.get_expression()

        self._function_mapping: Dict[str,
                                     Any] = self._alpha.get_function_mapping()

    def _rank_by_stock(
        self,
        computed_series: pd.Series,
        ascending: bool = False,
    ) -> Any:
        """
        This function is used to rank the computed series by stock.
        """
        self._data["tmp_rank_by_stock"] = computed_series
        return self._data.groupby("Date_only")["tmp_rank_by_stock"].rank(
            ascending=ascending,
            pct=True,
        )

    def _compute_returns(
        self,
        computed_series: pd.Series,
    ) -> Dict[str, np.float64]:
        """
        This function is used to compute the returns of the computed series.
        """
        self._data["tmp_returns"] = self._rank_by_stock(
            computed_series=computed_series,
            ascending=False,
        )
        return_data: Dict[str, np.float64] = dict()

        column_to_return: List[str] = [
            "Intra_diff",
            "Intra_diff_till_12",
            "Intra_diff_12_to_14",
        ]

        filtered_data: Any = self._data[self._data["tmp_returns"] < 0.03].copy(
        )

        filtered_list_grouped: Any = filtered_data.groupby("Date_only").agg(
            {
                "Intra_diff": "mean",
                "Intra_diff_till_12": "mean",
                "Intra_diff_12_to_14": "mean",
                "Intra_diff_autosquareoff": "mean",
            }
        )

        for column in column_to_return:
            return_data[column] = np.abs(filtered_list_grouped[column].sum()) - (
                filtered_list_grouped[column].count() * 0.05
            )

        filtered_data: Any = self._data[self._data["tmp_returns"] > 0.97].copy(
        )

        filtered_list_grouped: Any = filtered_data.groupby("Date_only").agg(
            {
                "Intra_diff": "mean",
                "Intra_diff_till_12": "mean",
                "Intra_diff_12_to_14": "mean",
                "Intra_diff_autosquareoff": "mean",
            }
        )

        for column in column_to_return:
            return_data[f"{column}_desc"] = np.abs(
                filtered_list_grouped[column].sum()
            ) - (filtered_list_grouped[column].count() * 0.05)

        return return_data

    def get_score(
        self,
    ) -> Dict[str, np.float64]:
        """
        This function is used to get the score of the alpha.
        """
        context: Dict[str, Any] = self._function_mapping.copy()

        for variable in self._alpha.get_all_variable_nodes():
            variable_name: str = variable.get_name()
            if variable_name in self._data.columns:
                context[variable_name] = self._data[variable_name]

        computed_series: Any = eval(self._expression, context)

        return self._compute_returns(computed_series=computed_series)
