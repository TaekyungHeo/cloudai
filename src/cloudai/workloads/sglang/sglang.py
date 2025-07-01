# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from cloudai.core import DockerImage, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition


class SGLangServerArgs(BaseModel):
    """Arguments for SGLang server."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())

    model_path: str = Field(default="nvidia/Llama-3.1-8B-Instruct-FP8", alias="model-path")
    trust_remote_code: bool = Field(default=True, alias="trust-remote-code")
    tp: int = 8
    disable_radix_cache: bool = Field(default=True, alias="disable-radix-cache")


class SGLangClientArgs(BaseModel):
    """Arguments for SGLang client."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    backend: str = "sglang"
    dataset_name: str = Field(default="random", alias="dataset-name")
    num_prompts: int = Field(default=3000, alias="num-prompts")
    random_input: int = Field(default=1024, alias="random-input")
    random_output: int = Field(default=1024, alias="random-output")
    random_range_ratio: float = Field(default=0.5, alias="random-range-ratio")
    max_concurrency: int = Field(default=1000, alias="max-concurrency")


class SGLangCmdArgs(CmdArgs):
    """Command line arguments for SGLang test."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    docker_image_url: str
    server: SGLangServerArgs = SGLangServerArgs()
    client: SGLangClientArgs = SGLangClientArgs()


class SGLangTestDefinition(TestDefinition):
    """Test definition for SGLang."""

    cmd_args: SGLangCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def installables(self) -> list[Installable]:
        return [self.docker_image]

    @property
    def hugging_face_home_path(self) -> Path:
        raw = self.extra_env_vars.get("HF_HOME")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("HF_HOME must be set and non-empty")
        path = Path(raw)
        if not path.is_dir():
            raise FileNotFoundError(f"HF_HOME path not found at {path}")
        return path

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        stdout_path = tr.output_path / "stdout.txt"
        if stdout_path.exists():
            # TODO: Add proper success indicators based on SGLang output
            return JobStatusResult(is_successful=True)

        return JobStatusResult(
            is_successful=False,
            error_message=(
                f"stdout.txt file not found in the specified output directory {tr.output_path}. "
                "This file is expected to be created as a result of the SGLang test run. "
                "Please ensure the SGLang test was executed properly and that stdout.txt is generated. "
                f"You can run the generated SGLang test command manually and verify the creation of {stdout_path}. "
                "If the issue persists, contact the system administrator."
            ),
        )
