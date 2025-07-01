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
from typing import Dict, List, Optional, Union, cast

from cloudai.core import TestRun
from cloudai.systems.slurm import SlurmCommandGenStrategy

from .sglang import SGLangTestDefinition


class SGLangSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for SGLang tests on Slurm systems."""

    def _container_mounts(self, tr: TestRun) -> list[str]:
        td = cast(SGLangTestDefinition, tr.test.test_definition)
        mounts = [
            f"{td.hugging_face_home_path}:{td.hugging_face_home_path}",
        ]

        script_host = (tr.output_path / "run.sh").resolve()
        script_container = "/opt/run.sh"
        self._generate_wrapper_script(script_host, tr)
        mounts.append(f"{script_host}:{script_container}")

        return mounts

    def _generate_wrapper_script(self, script_path: Path, tr: TestRun) -> None:
        server_cmd = " ".join(self._generate_server_command(tr))
        client_cmd = " ".join(self._generate_client_command(tr))

        lines = [
            "#!/bin/bash",
            "",
            'echo "Starting SGLang server-client workflow..."',
            "",
            'echo "Launching server in background..."',
            "# Start server in background and redirect output to separate files",
            f"{server_cmd} > /cloudai_run_results/server_stdout.txt 2> /cloudai_run_results/server_stderr.txt &",
            "server_pid=$!",
            'echo "Server started with PID: $server_pid"',
            "",
            "# Function to check if server is ready",
            "check_server_ready() {",
            '    if grep -q "The server is fired up and ready to roll!" /cloudai_run_results/server_stderr.txt; then',
            "        return 0",
            "    fi",
            "    return 1",
            "}",
            "",
            'echo "Waiting for server to be ready..."',
            "# Wait for server to be ready with timeout",
            "timeout=300  # 5 minutes timeout",
            "elapsed=0",
            "while ! check_server_ready; do",
            "    if [ $((elapsed % 10)) -eq 0 ]; then",
            '        echo "Still waiting for server... (${elapsed}s elapsed)"',
            "    fi",
            "    sleep 1",
            "    elapsed=$((elapsed + 1))",
            "    if [ $elapsed -ge $timeout ]; then",
            '        echo "ERROR: Timeout waiting for server to be ready after ${timeout}s"',
            '        echo "Last few lines of server stderr:"',
            "        tail -n 20 /cloudai_run_results/server_stderr.txt",
            "        kill $server_pid",
            "        exit 1",
            "    fi",
            "done",
            "",
            'echo "Server is ready! Starting client..."',
            "# Run client command with separate output files",
            f"{client_cmd} > /cloudai_run_results/client_stdout.txt 2> /cloudai_run_results/client_stderr.txt",
            "client_exit_code=$?",
            'echo "Client finished with exit code: $client_exit_code"',
            "",
            'echo "Shutting down server..."',
            "# Kill the server now that client is done",
            "kill $server_pid",
            'echo "Server shutdown complete"',
            "",
            "# Exit with the client's exit code",
            'echo "Workflow complete. Exiting with client exit code: $client_exit_code"',
            "exit $client_exit_code",
        ]

        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("\n".join(lines), encoding="utf-8")
        script_path.chmod(0o755)

    def generate_test_command(
        self, env_vars: Dict[str, Union[str, List[str]]], cmd_args: Dict[str, Union[str, List[str]]], tr: TestRun
    ) -> List[str]:
        return ["/opt/run.sh"]

    def image_path(self, tr: TestRun) -> Optional[str]:
        tdef: SGLangTestDefinition = cast(SGLangTestDefinition, tr.test.test_definition)
        return str(tdef.docker_image.installed_path)

    def gen_srun_prefix(self, tr: TestRun, use_pretest_extras: bool = False) -> List[str]:
        srun_prefix = super().gen_srun_prefix(tr, use_pretest_extras)
        num_nodes, _ = self.get_cached_nodes_spec(tr)
        srun_prefix.extend(
            [
                f"--ntasks={num_nodes}",
                "--ntasks-per-node=1",
            ]
        )
        return srun_prefix

    def _generate_server_command(self, tr: TestRun) -> List[str]:
        tdef: SGLangTestDefinition = cast(SGLangTestDefinition, tr.test.test_definition)
        config = tdef.cmd_args.server
        args = config.model_dump(by_alias=True, exclude_none=True)

        cmd_parts = ["python -m sglang.launch_server"]
        for k, v in args.items():
            if k == "cuda_graph_bs":
                continue
            if isinstance(v, bool):
                if v:
                    cmd_parts.append(f"--{k}")
            else:
                cmd_parts.append(f"--{k} {v}")

        cmd_parts.append("--cuda-graph-bs 1 2 4 8 16 32 64 128 256 512 1024")  # TODO: make this configurable

        if tr.test.extra_cmd_args:
            cmd_parts.append(tr.test.extra_cmd_args)

        return cmd_parts

    def _generate_client_command(self, tr: TestRun) -> List[str]:
        tdef: SGLangTestDefinition = cast(SGLangTestDefinition, tr.test.test_definition)
        config = tdef.cmd_args.client
        args = config.model_dump(by_alias=True, exclude_none=True)

        cmd_parts = ["python -m sglang.bench_serving"]
        cmd_parts.extend(f"--{k} {v}" for k, v in args.items())

        if tr.test.extra_cmd_args:
            cmd_parts.append(tr.test.extra_cmd_args)

        return cmd_parts
