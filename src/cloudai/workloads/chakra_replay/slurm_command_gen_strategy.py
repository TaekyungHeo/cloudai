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
from typing import Any, Dict, List, Union, cast

import toml

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy
from cloudai.workloads.chakra_replay import ChakraReplayTestDefinition


class ChakraReplaySlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for ChakraReplay on Slurm systems."""

    def _container_mounts(self, tr: TestRun) -> list[str]:
        tdef: ChakraReplayTestDefinition = cast(ChakraReplayTestDefinition, tr.test.test_definition)
        if tdef.cmd_args.trace_dir:
            return [f"{tdef.cmd_args.trace_dir}:{tdef.cmd_args.trace_dir}"]
        return []

    def _parse_slurm_args(
        self, job_name_prefix: str, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, tr)

        tdef: ChakraReplayTestDefinition = cast(ChakraReplayTestDefinition, tr.test.test_definition)
        base_args.update({"image_path": tdef.docker_image.installed_path})

        return base_args

    def _filter_config_data(
        self, cmd_args: Dict[str, Union[str, List[str]]]
    ) -> Dict[str, Dict[str, Union[str, int, bool]]]:
        config_data = {}

        sections = {
            "trace": {"directory": "trace_dir"},
            "replay": {"warmup_iters": "warmup_iters", "iters": "iters"},
            "profiler": {"enabled": "profiler.enabled"},
            "backend": {"name": "backend.name"},
            "logging": {"level": "logging.level"},
            "git_repo": {"url": "git_repo.url", "commit": "git_repo.commit"},
        }

        for section, fields in sections.items():
            section_data = {key: cmd_args[value] for key, value in fields.items() if value in cmd_args}
            if section_data:
                config_data[section] = section_data

        return config_data

    def _write_toml_config(self, cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun) -> str:
        config_path = Path("/tmp/chakra_replay_config.toml")
        config_data = self._filter_config_data(cmd_args)

        with config_path.open("w") as toml_file:
            toml.dump(config_data, toml_file)

        return str(config_path)

    def generate_test_command(
        self, env_vars: Dict[str, str], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        config_path = self._write_toml_config(cmd_args, tr)
        return ["comm_replay", f"--config {config_path}"]
