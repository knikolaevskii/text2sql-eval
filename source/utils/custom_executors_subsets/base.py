from abc import ABC, abstractmethod
import collections
from typing import Optional

import numpy as np
import pandas as pd
import re
from pandas.testing import assert_frame_equal, assert_series_equal

def subset_df(
    df_sub: pd.DataFrame,
    df_super: pd.DataFrame,
    question: str,
) -> bool:
    """
    Checks if df_sub is a subset of df_super.
    """
    if df_sub.empty:
        return False  # handle cases for empty dataframes

    # make a copy of df_super so we don't modify the original while keeping track of matches
    df_super_temp = df_super.copy(deep=True)
    matched_columns = []
    df_sub = deduplicate_columns(df_sub)
    df_super_temp = deduplicate_columns(df_super_temp)
    for col_sub_name in df_sub.columns:
        col_match = False
        for col_super_name in df_super_temp.columns:
            col_sub = df_sub[col_sub_name].sort_values().reset_index(drop=True)
            col_super = (
                df_super_temp[col_super_name].sort_values().reset_index(drop=True)
            )

            try:
                assert_series_equal(
                    col_sub, col_super, check_dtype=False, check_names=False
                )
                col_match = True
                matched_columns.append(col_super_name)
                # remove col_super_name to prevent us from matching it again
                df_super_temp = df_super_temp.drop(columns=[col_super_name])
                break
            except AssertionError:
                continue

        if not col_match:
            return False

    df_sub_normalized = normalize_table(df_sub, question)

    # get matched columns from df_super, and rename them with columns from df_sub, then normalize
    df_super_matched = df_super[matched_columns].rename(
        columns=dict(zip(matched_columns, df_sub.columns))
    )
    df_super_matched = normalize_table(
        df_super_matched, question
    )

    try:
        assert_frame_equal(df_sub_normalized, df_super_matched, check_dtype=False)
        return True
    except AssertionError:
        return False

def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    if len(cols) != len(set(cols)):
        duplicates = [
            item for item, count in collections.Counter(cols).items() if count > 1
        ]
        for dup in duplicates:
            indices = [i for i, x in enumerate(cols) if x == dup]
            for i in indices:
                cols[i] = f"{dup}_{i}"
        df.columns = cols
    return df

def normalize_table(
    df: pd.DataFrame, question: str
) -> pd.DataFrame:
    """
    Normalizes a dataframe by:
    1. removing all duplicate rows
    2. sorting columns in alphabetical order
    3. sorting rows using values from first column to last (if query_category is not 'order_by' and question does not ask for ordering)
    4. resetting index
    """
    # remove duplicate rows, if any
    df = df.drop_duplicates()

    # sort columns in alphabetical order of column names
    sorted_df = df.reindex(sorted(df.columns), axis=1)

    # check if query_category is 'order_by' and if question asks for ordering
    has_order_by = False
    pattern = re.compile(r"\b(order|sort|arrange)\b", re.IGNORECASE)
    in_question = re.search(pattern, question.lower())  # true if contains
    if in_question:
        has_order_by = True

    if not has_order_by:
        # sort rows using values from first column to last
        sorted_df = sorted_df.sort_values(by=list(sorted_df.columns))

    # reset index
    sorted_df = deduplicate_columns(sorted_df)
    sorted_df = sorted_df.reset_index(drop=True)
    return sorted_df

def compare_df(
    df_gold: pd.DataFrame,
    df_gen: pd.DataFrame,
    question: str,
) -> bool:
    """
    Compares two dataframes and returns True if they are the same, else False.
    query_gold and query_gen are the original queries that generated the respective dataframes.
    """
    # drop duplicates to ensure equivalence
    try:
        is_equal = df_gold.values == df_gen.values
        if is_equal.all():
            return True
    except:
        try:
            is_equal = df_gold.values == df_gen.values
            if is_equal:
                return True
        except:
            pass

    df_gold = normalize_table(df_gold, question)
    df_gen = normalize_table(df_gen, question)

    # perform same checks again for normalized tables
    if df_gold.shape != df_gen.shape:
        return False
    # fill NaNs with -99999 to handle NaNs in the dataframes for comparison
    df_gen.fillna(-99999, inplace=True)
    df_gold.fillna(-99999, inplace=True)
    is_equal = df_gold.values == df_gen.values
    try:
        return is_equal.all()
    except:
        return is_equal

class BaseExecutor(ABC):
    
    @abstractmethod
    def execute_sql(self, sql: str, dsn_or_db_path: str) -> dict:
        return {"result": None, "execution_time": None, "error": None}

    def match_sqls(
        self, predicted_sql: str, gold_sql: str, dsn_or_db_path: str, question: Optional[str] = None
    ) -> dict:
        """
        Match SQLs and return both exact match and subset match results.
        
        Returns:
            dict: {
                "result": int,        # 1 for exact match, 0 otherwise
                "subset_match": int,  # 1 for subset match, 0 otherwise
                "error": str or None  # Error message if any
            }
        """
        prediction = self.execute_sql(sql=predicted_sql, dsn_or_db_path=dsn_or_db_path)
        gold = self.execute_sql(sql=gold_sql, dsn_or_db_path=dsn_or_db_path)

        if prediction["error"]:
            return {
                "result": 0,
                "subset_match": 0,
                "error": prediction["error"],
            }
        prediction_list = prediction["result"]
        gold_list = gold["result"]
        
        is_match = set(prediction_list) == set(gold_list)
        if is_match:
            return {
                "result": int(is_match),
                "subset_match": 1,
                "error": None ,
            }
        
        # Use question if provided, otherwise empty string
        question_str = question if question is not None else ""
        
        # Check exact match first
        # exact = compare_df(gold["result_df"], prediction["result_df"], question_str)
        
        # if exact:
        #     return {
        #         "result": 1,
        #         "subset_match": 1,  # If exact match, subset is also true
        #         "error": None,
        #     }
        
        # Check subset match (gold is subset of prediction)
        subset = subset_df(gold["result_df"], prediction["result_df"], question_str)
        
        if subset:
            return {
                "result": 0,
                "subset_match": 1,
                "error": None
            }
        else:
            return {
                "result": 0,
                "subset_match": 0,
                "error": "Table mismatch",
            }

    def clean_abnormal(self, input: list[float]) -> list[float]:
        input_array = np.asarray(input)
        mean = np.mean(input_array)
        std = np.std(input_array)
        return [x for x in input_array if mean - 3 * std < x < mean + 3 * std]

    def iterated_execution(
        self,
        predicted_sql: str,
        gold_sql: str,
        dsn_or_db_path: str,
        num_iterations: int,
    ) -> dict:
        is_match = self.match_sqls(
            predicted_sql=predicted_sql,
            gold_sql=gold_sql,
            dsn_or_db_path=dsn_or_db_path,
        )

        if is_match["result"] == 1:
            diff_list = [
                self.execute_sql(sql=predicted_sql, dsn_or_db_path=dsn_or_db_path)[
                    "execution_time"
                ]
                / self.execute_sql(sql=gold_sql, dsn_or_db_path=dsn_or_db_path)[
                    "execution_time"
                ]
                for _ in range(num_iterations)
            ]
            processed_diff_list = self.clean_abnormal(diff_list)
            return {
                "result": sum(processed_diff_list) / len(processed_diff_list),
                "subset_match": 1,  # Add subset_match for consistency
                "error": None,
            }
        elif is_match["subset_match"] == 1:
            # For VES, subset matches might get partial credit
            return {
                "result": 0.5,  # Partial credit for subset match
                "subset_match": 1,
                "error": None,
            }
        else:
            return {
                "result": 0, 
                "subset_match": 0,
                "error": is_match["error"]
            }