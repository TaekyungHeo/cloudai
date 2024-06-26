# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Dict

import pytest
from cloudai.__main__ import handle_dry_run_and_run

SLURM_TEST_SCENARIOS = [
    {
        "path": Path("conf/v0.6/general/test_scenario/sleep.toml"),
        "expected_dirs_number": 3,
    },
    {
        "path": Path("conf/v0.6/general/test_scenario/ucc_test.toml"),
        "expected_dirs_number": 1,
    },
]


@pytest.mark.parametrize("scenario", SLURM_TEST_SCENARIOS, ids=lambda x: str(x))
def test_slurm(tmp_path: Path, scenario: Dict):
    test_scenario_path = scenario["path"]
    expected_dirs_number = scenario.get("expected_dirs_number")

    handle_dry_run_and_run(
        "dry-run",
        Path("conf/v0.6/general/system/example_slurm_cluster.toml"),
        Path("conf/v0.6/general/test_template"),
        Path("conf/v0.6/general/test"),
        test_scenario_path,
        tmp_path,
    )

    results_output = list(tmp_path.glob("*"))[0]
    test_dirs = list(results_output.iterdir())

    if expected_dirs_number is not None:
        assert len(test_dirs) == expected_dirs_number, "Dirs number in output is not as expected"

    for td in test_dirs:
        assert td.is_dir(), "Invalid test directory"
        assert "Tests." in td.name, "Invalid test directory name"
