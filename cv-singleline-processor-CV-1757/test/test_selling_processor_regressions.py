import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


SELLING_PROCESSOR = Path(__file__).resolve().parents[2] / "selling_processor.py"


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture(scope="module")
def selling_processor():
    class Placeholder:
        pass

    stubs = {
        "metrics": _module("metrics"),
        "db": _module("db", BigTableClient=Placeholder, BigQueryClient=Placeholder),
        "services": _module("services"),
        "services.model_interfaces": _module("services.model_interfaces"),
        "services.model_interfaces.model_interface_base": _module(
            "services.model_interfaces.model_interface_base",
            SellingModelBase=Placeholder,
        ),
        "services.model_interfaces.common_model_functions": _module(
            "services.model_interfaces.common_model_functions",
            retry_model_predict=lambda *args, **kwargs: (500, None),
        ),
        "configuration": _module("configuration", Settings=Placeholder),
        "shapely": _module("shapely"),
        "shapely.geometry": _module(
            "shapely.geometry",
            Polygon=Placeholder,
            box=lambda *args: Placeholder(),
        ),
    }

    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(
            "selling_processor_regression_tests",
            SELLING_PROCESSOR,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

    module.print_log = lambda *args, **kwargs: None
    yield module
    sys.modules.pop(spec.name, None)


def _sequence_rows():
    return [
        {"SKU_NBR": "1000000123", "LOCATIONID": 1, "CURR_RETL_AMT": 10.00},
        {"SKU_NBR": "1000000456", "LOCATIONID": 2, "CURR_RETL_AMT": 20.00},
        {"SKU_NBR": "1000000789", "LOCATIONID": 3, "CURR_RETL_AMT": 30.00},
    ]


def test_row_column_conversion_supports_ragged_grids(selling_processor):
    price_grid = [["1.00"], ["2.00", "3.00"]]

    assert selling_processor.grid_position_to_flat_index(price_grid, 0, 0) == 0
    assert selling_processor.grid_position_to_flat_index(price_grid, 1, 0) == 1
    assert selling_processor.grid_position_to_flat_index(price_grid, 1, 1) == 2


def test_sequence_freshness_detects_expected_count_and_location_gaps(selling_processor):
    rows = _sequence_rows()

    assert selling_processor.sequence_data_needs_refresh([]) is True
    assert selling_processor.sequence_data_needs_refresh(rows) is False
    assert selling_processor.sequence_data_needs_refresh(
        [dict(rows[0], EXPECTED_RECORD_COUNT=4), rows[1], rows[2]]
    ) is True
    assert selling_processor.sequence_data_needs_refresh([rows[0], rows[2]]) is True


def test_empty_candidate_lookup_accepts_production_sequence_columns(
    selling_processor,
):
    rows = _sequence_rows()

    matches = selling_processor.find_candidates_empty({"dollars": 20}, rows)

    assert matches == [rows[1]]
    assert selling_processor._sequence_retail_amount(rows[1]) == 20.0
