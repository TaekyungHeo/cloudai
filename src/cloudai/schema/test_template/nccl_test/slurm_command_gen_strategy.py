# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Dict, List

from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy

from .slurm_install_strategy import NcclTestSlurmInstallStrategy


class NcclTestSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """Command generation strategy for NCCL tests on Slurm systems."""

    def _parse_slurm_args(
        self,
        job_name_prefix: str,
        env_vars: Dict[str, str],
        cmd_args: Dict[str, str],
        num_nodes: int,
        nodes: List[str],
    ) -> Dict[str, Any]:
        base_args = super()._parse_slurm_args(job_name_prefix, env_vars, cmd_args, num_nodes, nodes)

        image_path = self.docker_image_cache_manager.ensure_docker_image(
            self.docker_image_url,
            NcclTestSlurmInstallStrategy.SUBDIR_PATH,
            NcclTestSlurmInstallStrategy.DOCKER_IMAGE_FILENAME,
        ).docker_image_path

        container_mounts = ""
        if "NCCL_TOPO_FILE" in env_vars and "DOCKER_NCCL_TOPO_FILE" in env_vars:
            nccl_graph_path = Path(env_vars["NCCL_TOPO_FILE"]).resolve()
            nccl_graph_file = env_vars["DOCKER_NCCL_TOPO_FILE"]
            container_mounts = f"{nccl_graph_path}:{nccl_graph_file}"
        elif "NCCL_TOPO_FILE" in env_vars:
            del env_vars["NCCL_TOPO_FILE"]

        base_args.update({"image_path": image_path, "container_mounts": container_mounts})

        return base_args

    def generate_test_command(
        self, env_vars: Dict[str, str], cmd_args: Dict[str, str], extra_cmd_args: str
    ) -> List[str]:
        srun_command_parts = [f"/usr/local/bin/{cmd_args['subtest_name']}"]
        nccl_test_args = [
            "nthreads",
            "ngpus",
            "minbytes",
            "maxbytes",
            "stepbytes",
            "op",
            "datatype",
            "root",
            "iters",
            "warmup_iters",
            "agg_iters",
            "average",
            "parallel_init",
            "check",
            "blocking",
            "cudagraph",
        ]
        for arg in nccl_test_args:
            if arg in cmd_args:
                srun_command_parts.append(f"--{arg} {cmd_args[arg]}")

        if extra_cmd_args:
            srun_command_parts.append(extra_cmd_args)

        return srun_command_parts
