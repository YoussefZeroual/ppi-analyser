# analysis/results.py

import re
import json
import logging
import importlib

from ppi_analyser.state import SessionState
from ppi_analyser.config import RESULT_COLUMNS

logger = logging.getLogger(__name__)


def clean_results(results: list, expression: str, conv: str) -> list:
    cleaned_results = []
    for item in results:
        try:
            if not item or item.strip() == '':
                cleaned_results.append(None)
                continue
        except AttributeError as e:
            logger.warning("clean_results AttributeError: %s", e)

        item = item.strip()
        item = re.sub(r'^```json\s*', '', item)
        item = re.sub(r'^```\s*', '', item)
        item = re.sub(r'\s*```$', '', item)

        json_match = re.search(r'\{.*\}', item, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            json_str = json_str.replace('\\', '')
            json_str = re.sub(r',\s*\n\s*}', '}', json_str)
            json_str = re.sub(r',\s*\n\s*]', ']', json_str)
            try:
                cleaned_results.append(json.loads(json_str))
            except json.JSONDecodeError:
                try:
                    json_str_fixed = re.sub(r'(?<!\\)"', r'\"', json_str)
                    cleaned_results.append(json.loads(json_str_fixed))
                except Exception:
                    try:
                        prop_match = re.search(r'"Propriété"\s*:\s*"([^"]+)"', json_str, re.IGNORECASE)
                        just_match = re.search(r'"Justification"\s*:\s*"([^"]*(?:"[^"]+)*)"', json_str, re.DOTALL | re.IGNORECASE)
                        if prop_match and just_match:
                            cleaned_results.append({
                                "Propriété": prop_match.group(1),
                                "Justification": just_match.group(1)
                            })
                        else:
                            cleaned_results.append(None)
                    except Exception:
                        cleaned_results.append(None)
        else:
            cleaned_results.append(None)

    return cleaned_results


def create_df(results: list, df_index: int, expression: str, conv: str, state: SessionState):
    np = importlib.import_module('numpy')
    pd = importlib.import_module('pandas')

    c = clean_results(results, expression, conv)
    if c is None:
        logger.warning("Results json empty, returning empty df")
        return pd.DataFrame()

    try:
        df = pd.DataFrame(c)
        state.dfs.append(c)
    except (AttributeError, TypeError) as e:
        logger.error("Could not create df from cleaned results: %s", e)
        return pd.DataFrame()

    flat_values = df.values.flatten(order='C')
    result_df = pd.DataFrame([flat_values]).reset_index(drop=True)

    try:
        result_df.columns = RESULT_COLUMNS
    except ValueError as e:
        logger.error(
            "Error renaming columns at df_index %d — expected %d, got %d: %s",
            df_index, len(RESULT_COLUMNS), len(result_df.columns), e
        )

    if state.err_list:
        for idx in sorted(state.err_list):
            result_df.loc[idx + 0.5] = [np.nan] * len(RESULT_COLUMNS)
        result_df = result_df.sort_index().reset_index(drop=True)

    return result_df
