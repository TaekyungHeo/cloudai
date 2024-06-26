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

import os

from cloudai._core.job_status_result import JobStatusResult
from cloudai._core.job_status_retrieval_strategy import JobStatusRetrievalStrategy


class NcclTestJobStatusRetrievalStrategy(JobStatusRetrievalStrategy):
    """Strategy to retrieve job status for NCCL tests by checking the contents of 'stdout.txt'."""

    def get_job_status(self, output_path: str) -> JobStatusResult:
        """
        Determine the job status by examining 'stdout.txt' in the output directory.

        Args:
            output_path (str): Path to the directory containing 'stdout.txt'.

        Returns:
            JobStatusResult: The result containing the job status and an optional error message.
        """
        stdout_path = os.path.join(output_path, "stdout.txt")
        if os.path.isfile(stdout_path):
            with open(stdout_path, "r") as file:
                content = file.read()
                if "# Out of bounds values" in content and "# Avg bus bandwidth" in content:
                    return JobStatusResult(is_successful=True)
                missing_indicators = []
                if "# Out of bounds values" not in content:
                    missing_indicators.append("'# Out of bounds values'")
                if "# Avg bus bandwidth" not in content:
                    missing_indicators.append("'# Avg bus bandwidth'")
                error_message = (
                    f"Missing success indicators in {stdout_path}: {', '.join(missing_indicators)}. "
                    "These keywords are expected to be present in stdout.txt, usually towards the end of the file. "
                    f"Please ensure the NCCL test ran to completion. You can run the generated sbatch script manually "
                    f"and check if {stdout_path} is created and contains the expected keywords."
                )
                return JobStatusResult(is_successful=False, error_message=error_message)
        return JobStatusResult(
            is_successful=False,
            error_message=(
                f"stdout.txt file not found in the specified output directory {output_path}. "
                "This file is expected to be created as a result of the NCCL test run. "
                "Please ensure the NCCL test was executed properly and that stdout.txt is generated. "
                f"You can run the generated NCCL test command manually and verify the creation of {stdout_path}."
            ),
        )
