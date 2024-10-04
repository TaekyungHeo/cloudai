#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class Plugin:
    """
    A class to represent plugins actions in a test scenario.

    Attributes
        name (str): Name of the plugin task.
        command (str): Shell command to be executed in the plugin.
    """

    def __init__(self, name: str, command: str) -> None:
        """
        Initialize a Plugin instance.

        Args:
            name (str): Name of the plugin task.
            command (str): Command to execute as part of the plugin.
        """
        self.name = name
        self.command = command

    def __repr__(self) -> str:
        """Return a string representation of the Plugin instance."""
        return f"Plugin(name={self.name}, command={self.command})"

    def run(self) -> None:
        """Execute the command in the plugin."""
        print(f"Executing command '{self.command}' for task '{self.name}'")
