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
from typing import Any, Dict
from unittest.mock import Mock

import pytest

from cloudai.core import Test, TestRun
from cloudai.systems.slurm import SlurmSystem
from cloudai.workloads.sglang import (
    SGLangClientArgs,
    SGLangCmdArgs,
    SGLangServerArgs,
    SGLangSlurmCommandGenStrategy,
    SGLangTestDefinition,
)


class TestSGLangSlurmCommandGenStrategy:
    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem) -> SGLangSlurmCommandGenStrategy:
        return SGLangSlurmCommandGenStrategy(slurm_system)

    @pytest.fixture
    def server_args_base(self) -> Dict[str, Any]:
        return {
            "model_path": "deepseek-ai/DeepSeek-R1",
            "trust_remote_code": True,
            "tp": 8,
            "disable_radix_cache": True,
        }

    @pytest.fixture
    def client_args_base(self) -> Dict[str, Any]:
        return {
            "backend": "sglang",
            "dataset_name": "random",
            "num_prompts": 3000,
            "random_input": 1024,
            "random_output": 1024,
            "random_range_ratio": 0.5,
            "max_concurrency": 1000,
        }

    @pytest.fixture
    def test_run(self, tmp_path: Path, server_args_base: Dict[str, Any], client_args_base: Dict[str, Any]) -> TestRun:
        server_args = SGLangServerArgs(**server_args_base)
        client_args = SGLangClientArgs(**client_args_base)

        tdef = SGLangTestDefinition(
            name="sglang_test",
            description="SGLang server test",
            test_template_name="default_template",
            cmd_args=SGLangCmdArgs(
                docker_image_url="registry.example.com/sglang:latest",
                server=server_args,
                client=client_args,
            ),
            extra_env_vars={"HF_HOME": str(tmp_path / "hf_home")},
            extra_cmd_args={},
        )

        # Create HF_HOME directory
        hf_home = tmp_path / "hf_home"
        hf_home.mkdir(parents=True)

        test = Test(test_definition=tdef, test_template=Mock())
        return TestRun(
            test=test,
            num_nodes=1,
            nodes=[],
            output_path=tmp_path / "output",
            name="sglang-job",
        )

    @pytest.mark.parametrize(
        "server_args_override,expected_flags,unexpected_flags",
        [
            (
                {},  # Default config
                [
                    "--model-path deepseek-ai/DeepSeek-R1",
                    "--trust-remote-code",
                    "--tp 8",
                    "--disable-radix-cache",
                    "--cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 1024",
                ],
                [],
            ),
            (
                {  # Modified config
                    "model_path": "meta-llama/Llama-2-7b",
                    "tp": 4,
                    "trust_remote_code": False,
                },
                [
                    "--model-path meta-llama/Llama-2-7b",
                    "--tp 4",
                ],
                ["--trust-remote-code"],
            ),
        ],
    )
    def test_gen_server_command(
        self,
        cmd_gen_strategy: SGLangSlurmCommandGenStrategy,
        test_run: TestRun,
        server_args_base: Dict[str, Any],
        server_args_override: Dict[str, Any],
        expected_flags: list[str],
        unexpected_flags: list[str],
    ) -> None:
        server_args = {**server_args_base, **server_args_override}
        test_run.test.test_definition.cmd_args.server = SGLangServerArgs(**server_args)

        command = cmd_gen_strategy._generate_server_command(test_run)

        assert "python -m sglang.launch_server" in command

        for flag in expected_flags:
            assert flag in command

        for flag in unexpected_flags:
            assert flag not in command

    @pytest.mark.parametrize(
        "client_args_override,expected_flags",
        [
            (
                {},  # Default config
                [
                    "--backend sglang",
                    "--dataset-name random",
                    "--num-prompts 3000",
                    "--random-input 1024",
                    "--random-output 1024",
                    "--random-range-ratio 0.5",
                    "--max-concurrency 1000",
                ],
            ),
            (
                {  # Modified config
                    "num_prompts": 5000,
                    "max_concurrency": 2000,
                    "random_range_ratio": 0.75,
                },
                [
                    "--backend sglang",
                    "--dataset-name random",
                    "--num-prompts 5000",
                    "--random-input 1024",
                    "--random-output 1024",
                    "--random-range-ratio 0.75",
                    "--max-concurrency 2000",
                ],
            ),
        ],
    )
    def test_gen_client_command(
        self,
        cmd_gen_strategy: SGLangSlurmCommandGenStrategy,
        test_run: TestRun,
        client_args_base: Dict[str, Any],
        client_args_override: Dict[str, Any],
        expected_flags: list[str],
    ) -> None:
        client_args = {**client_args_base, **client_args_override}
        test_run.test.test_definition.cmd_args.client = SGLangClientArgs(**client_args)

        command = cmd_gen_strategy._generate_client_command(test_run)

        assert "python -m sglang.bench_serving" in command

        for flag in expected_flags:
            assert flag in command
