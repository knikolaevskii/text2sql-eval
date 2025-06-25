import math
import traceback
from pathlib import Path
from typing import Optional, Union
import itertools

from func_timeout import FunctionTimedOut, func_timeout
from tqdm.auto import tqdm

from utils.custom_executors_subsets import BaseExecutor
from premsql.utils import save_to_json


class Text2SQLEvaluator:
    def __init__(
        self, executor: BaseExecutor, experiment_path: Union[str, Path]
    ) -> None:
        self.executor = executor
        self.experiment_path = Path(experiment_path)

    def _execute_model(
        self,
        metric_name: str,
        generated_sql: str,
        gold_sql: str,
        dsn_or_db_path: str,
        meta_time_out: Optional[int] = 1000,
        num_iterations: Optional[int] = None,
        debug: Optional[bool] = False,
        question: Optional[str] = None,
    ):
        assert metric_name in ["accuracy", "ves"], "Invalid metric name"
        try:
            if metric_name == "accuracy":
                result = func_timeout(
                    meta_time_out,
                    self.executor.match_sqls,
                    args=(generated_sql, gold_sql, dsn_or_db_path),
                )
                # print(result)
            elif metric_name == "ves":
                num_iterations = 10 if num_iterations is None else num_iterations
                result = func_timeout(
                    meta_time_out,
                    self.executor.iterated_execution,
                    args=(generated_sql, gold_sql, dsn_or_db_path, num_iterations),
                )
            else:
                raise ValueError(f"Invalid metric name: {metric_name}")

            return {
                metric_name: result["result"],
                "subset_match": result["subset_match"],  # Add subset tracking
                "error": result["error"],
            }
        except FunctionTimedOut as e:
            return {
                metric_name: 0,
                "subset_match": 0,
                "error": f"Function Timed out: {e}",
            }
        except Exception as e:
            if debug:
                traceback.print_exc()

            return {
                metric_name: 0,
                "subset_match": 0,
                "error": f"Exception: {e}",
            }
    
    # find start and end index of { } in a string. return (start, end) if found, else return (-1, -1)
    def find_bracket_indices(self, s: str, start_index: int = 0) -> "tuple[int, int]":
        start = s.find("{", start_index)
        end = s.find("}", start + 1)
        if start == -1 or end == -1:
            return (-1, -1)
        return (start, end)

    # extrapolate all possible queries from a query with { } in it
    def get_all_minimal_queries(self, query: str) -> "list[str]":
        """
        extrapolate all possible queries
        - split by semicolon. this is to accommodate queries where joins to other tables are also acceptable.
        - expand all column permutations if there are braces { } in it. eg:
        ```sql
            SELECT {user.id, user.name} FROM user;
        ```
        Would be expanded to:
        ```sql
            SELECT user.id FROM user;
            SELECT user.name FROM user;
            SELECT user.id, user.name FROM user;
        ```
        """
        queries = query.split(";")
        result_queries = []
        for query in queries:
            query = query.strip()
            if query == "":
                continue
            start, end = self.find_bracket_indices(query, 0)
            if (start, end) == (-1, -1):
                result_queries.append(query)
                continue
            else:
                # get all possible column subsets
                column_options = query[start + 1 : end].split(",")
                column_combinations = list(
                    itertools.chain.from_iterable(
                        itertools.combinations(column_options, r)
                        for r in range(1, len(column_options) + 1)
                    )
                )
                for column_tuple in column_combinations:
                    left = query[:start]
                    column_str = ", ".join(column_tuple)
                    right = query[end + 1 :]
                    # change group by size dynamically if necessary
                    right = right.replace("GROUP BY {}", f"GROUP BY {column_str}")
                    result_queries.append(left + column_str + right)
        return result_queries

    def execute(
        self,
        metric_name: str,
        model_responses: list[dict],
        filter_by: Optional[str] = None,
        num_iterations: Optional[int] = 10,
        meta_time_out: Optional[int] = 10,
        debug: Optional[bool] = False,
    ) -> dict:
        data_with_results = []

        for response in tqdm(model_responses, total=len(model_responses)):
            generated_sql = response["generated"]
            gold_sql = response["SQL"]
            db_path = response["db_path"]
            question = response["question"]

            minimal_gold_queries = self.get_all_minimal_queries(gold_sql)
            best_result = None
            best_subset_result = None

            for gold_variant in minimal_gold_queries:
                result = self._execute_model(
                    metric_name=metric_name,
                    generated_sql=generated_sql,
                    gold_sql=gold_variant,
                    dsn_or_db_path=db_path,
                    num_iterations=num_iterations,
                    meta_time_out=meta_time_out,
                    debug=debug,
                    question=question,
                )

                # Early stop if exact match is found
                if result[metric_name] == 1:
                    data_with_results.append({**response, **result})
                    break
                # Track best subset match
                elif result["subset_match"] == 1:
                    best_subset_result = result
                else:
                    best_result = result

            else:
                # No early break: use best available result
                if best_subset_result is not None:
                    data_with_results.append({**response, **best_subset_result})
                elif best_result is not None:
                    data_with_results.append({**response, **best_result})
                else:
                    # This shouldn't happen, but just in case
                    data_with_results.append({**response, **{
                        metric_name: 0,
                        "subset_match": 0,
                        "error": "No valid result found"
                    }})

        execution_result = {}
        
        if filter_by:
            if filter_by not in data_with_results[0]:
                raise KeyError(f"Filter key: {filter_by} is not found in responses")

            filter_values = {response[filter_by] for response in data_with_results}
            total_responses = len(data_with_results)
            overall_metric = 0.0
            overall_subset_metric = 0.0
            overall_db_errors = 0
            overall_logic_errors = 0
            overall_successes = 0
            overall_subset_matches = 0

            for value in filter_values:
                filtered_responses = [
                    response
                    for response in data_with_results
                    if response[filter_by] == value
                ]
                
                # Compute metric and error counts for this filter value
                metric_result = self.compute_metric_with_errors(
                    results=filtered_responses, metric_name=metric_name
                )
                
                execution_result[value] = metric_result
                
                # Weight the overall metrics by the number of responses in this group
                weight = len(filtered_responses) / total_responses
                overall_metric += metric_result[f"{metric_name}_percentage"] * weight
                overall_subset_metric += metric_result["subset_percentage"] * weight
                overall_db_errors += metric_result["db_error_count"]
                overall_logic_errors += metric_result["logic_error_count"]
                overall_successes += metric_result["success_count"]
                overall_subset_matches += metric_result["subset_count"]

            execution_result["overall"] = {
                f"{metric_name}_percentage": overall_metric,
                "subset_percentage": overall_subset_metric,
                "success_count": overall_successes,
                "subset_count": overall_subset_matches,
                "logic_error_count": overall_logic_errors,
                "db_error_count": overall_db_errors,
                "success_percentage": (overall_successes / total_responses) * 100 if total_responses > 0 else 0,
                "subset_match_percentage": (overall_subset_matches / total_responses) * 100 if total_responses > 0 else 0,
                "logic_error_percentage": (overall_logic_errors / total_responses) * 100 if total_responses > 0 else 0,
                "db_error_percentage": (overall_db_errors / total_responses) * 100 if total_responses > 0 else 0
            }
        else:
            metric_result = self.compute_metric_with_errors(
                results=data_with_results, metric_name=metric_name
            )
            execution_result["overall"] = metric_result

        save_to_json(
            json_object=execution_result,
            save_path=self.experiment_path / f"{metric_name}.json",
        )

        # also save the data_with_results
        save_to_json(
            json_object=data_with_results,
            save_path=self.experiment_path / "predict_eval.json",
        )
        return execution_result

    def compute_metric_with_errors(self, results: list[dict], metric_name: str) -> dict:
        """
        Compute metric with simplified error classification and subset tracking
        
        Error Types:
        - Logic Errors: "Table mismatch" (SQL executed but wrong results)
        - DB Errors: All other errors (SQL couldn't execute, timeouts, exceptions, etc.)
        - Successes: Perfect matches (exact matches)
        - Subset Matches: Partial matches where gold is subset of predicted
        
        Returns:
            dict: Contains metric percentage, subset percentage, success count, subset count, 
                  logic errors, db errors and their percentages
        """
        total_queries = len(results)
        
        # Count different types of errors and successes
        db_error_count = 0        # All errors except table mismatch
        logic_error_count = 0     # Table mismatch only
        success_count = 0         # Perfect match (exact)
        subset_count = 0          # Subset match but not exact
        
        for result in results:
            error_msg = result.get("error", "")
            exact_match = result.get(metric_name, 0)
            subset_match = result.get("subset_match", 0)
            
            # Handle None values for error_msg
            if error_msg is None:
                error_msg = ""
            
            if exact_match == 1:
                # Exact match - highest priority
                success_count += 1
            elif subset_match == 1:
                # Subset match but not exact
                subset_count += 1
            elif not error_msg or not error_msg.strip():
                # No error but also no match - this shouldn't happen but handle it
                logic_error_count += 1
            elif error_msg == "Table mismatch":
                # SQL executed but results don't match at all
                logic_error_count += 1
            else:
                # All other errors are DB errors (syntax, timeouts, exceptions, etc.)
                db_error_count += 1
        
        # Compute the actual metric (exact matches only)
        if metric_name == "accuracy":
            metric_value = sum(res[metric_name] for res in results) / total_queries * 100
            subset_value = sum(res.get("subset_match", 0) for res in results) / total_queries * 100
        elif metric_name == "ves":
            total_ratio = 0.0
            subset_ratio = 0.0
            for result in results:
                if result[metric_name] == 1:
                    total_ratio += math.sqrt(result[metric_name]) * 100
                if result.get("subset_match", 0) == 1:
                    subset_ratio += math.sqrt(result.get("subset_match", 0)) * 100
            metric_value = total_ratio / total_queries
            subset_value = subset_ratio / total_queries
        else:
            raise ValueError(f"Invalid metric name: {metric_name}")
        
        return {
            f"{metric_name}_percentage": metric_value,
            "subset_percentage": subset_value,
            "success_count": success_count,
            "subset_count": subset_count,
            "logic_error_count": logic_error_count,
            "db_error_count": db_error_count,
            "success_percentage": (success_count / total_queries) * 100 if total_queries > 0 else 0,
            "subset_match_percentage": (subset_count / total_queries) * 100 if total_queries > 0 else 0,
            "logic_error_percentage": (logic_error_count / total_queries) * 100 if total_queries > 0 else 0,
            "db_error_percentage": (db_error_count / total_queries) * 100 if total_queries > 0 else 0
        }