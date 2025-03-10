# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES All rights reserved.

import logging
import re
import subprocess
from pathlib import Path
from typing import Tuple, cast

import pandas as pd

from cloudai import TestDefinition
from cloudai.report_generator.tool.csv_report_tool import CSVReportTool

from .nccl import NCCLTestDefinition


class NcclTestPredictionReportGenerator:
    """Generate NCCL test predictor reports by extracting and analyzing performance data."""

    def __init__(self, collective_type: str, output_path: Path, test_definition: TestDefinition):
        self.collective_type = collective_type
        self.output_path = output_path
        self.stdout_path = output_path / "stdout.txt"
        self.test_definition = cast(NCCLTestDefinition, test_definition)
        self.venv_path = self._get_venv_path()
        assert self.test_definition.predictor is not None
        self.predictor = self.test_definition.predictor
        self.predictor_path = self._get_predictor_path()

    def _get_venv_path(self) -> Path:
        venv_path = self.predictor.venv_path
        if venv_path is None:
            logging.warning(
                f"Virtual environment is missing for {self.predictor.git_repo.url}. "
                "Ensure installation is completed before running the test."
            )
            return Path()
        return venv_path

    def _get_predictor_path(self) -> Path:
        repo_path = self.predictor.git_repo.installed_path
        if repo_path is None:
            logging.warning(
                "Local clone of git repository "
                f"{self.predictor.git_repo.url} is missing. "
                "Ensure installation is completed before running the test."
            )
            return Path()
        return repo_path / "maya/predictor"

    def generate(self) -> None:
        maya_inference_path = self.venv_path / "bin" / "maya_inference"

        if not self.venv_path.exists():
            logging.warning(
                f"Virtual environment directory {self.venv_path} does not exist. "
                "Ensure the correct environment is set up before running the test. Skipping report generation."
            )
            return

        if not maya_inference_path.exists():
            logging.warning(
                f"maya_inference executable not found at {maya_inference_path}. "
                "Ensure the virtual environment is correctly set up and dependencies are installed. "
                "Skipping report generation."
            )
            return

        gpu_type, num_devices, num_ranks = self._extract_device_info()
        df = self._extract_performance_data(gpu_type, num_devices, num_ranks)

        if df.empty:
            logging.warning(
                "No valid NCCL performance data extracted from stdout. "
                "Ensure the NCCL test ran successfully before generating predictions."
            )
            return

        self._store_intermediate_data(df.drop(columns=["gpu_type", "measured_dur"]))
        predictions = self._run_predictor(gpu_type, maya_inference_path)

        if predictions.empty:
            logging.warning("Prediction output is empty. Skipping report generation.")
            return

        self._generate_prediction_report(df, predictions)

    def _extract_device_info(self) -> Tuple[str, int, int]:
        gpu_type = "Unknown"
        num_ranks = 0
        device_indices = {}

        if not self.stdout_path.is_file():
            logging.warning(
                f"stdout file {self.stdout_path} not found. "
                "Ensure the NCCL test was executed successfully before generating the report."
            )
            return gpu_type, 0, 0

        with self.stdout_path.open(encoding="utf-8") as file:
            for line in file:
                if "Rank" in line and "device" in line and "NVIDIA" in line:
                    num_ranks += 1

                    if device_match := re.search(r"on\s+([\w\d\-.]+)\s+device\s+(\d+)", line):
                        host, device_index = device_match.groups()
                        device_indices[host] = max(device_indices.get(host, -1), int(device_index))

                    if gpu_match := re.search(r"NVIDIA\s+([A-Z0-9]+)", line):
                        gpu_type = gpu_match.group(1).strip()

        num_devices = max(device_indices.values(), default=-1) + 1 if device_indices else 0
        logging.debug(f"Extracted GPU Type: {gpu_type}, Devices per Node: {num_devices}, Ranks: {num_ranks}")
        return gpu_type, num_devices, num_ranks

    def _extract_performance_data(self, gpu_type: str, num_devices: int, num_ranks: int) -> pd.DataFrame:
        if not self.stdout_path.is_file():
            return pd.DataFrame()

        extracted_data = [
            [gpu_type, num_devices, num_ranks, float(match.group(1)), round(float(match.group(2)), 2)]
            for line in self.stdout_path.open(encoding="utf-8")
            if (
                match := re.match(r"^\s*(\d+)\s+\d+\s+\S+\s+\S+\s+[-\d]+\s+\S+\s+\S+\s+\S+\s+\d+\s+(\S+)", line.strip())
            )
        ]

        if not extracted_data:
            logging.debug("No valid NCCL performance data found in stdout.")
            return pd.DataFrame()

        return pd.DataFrame(
            extracted_data, columns=["gpu_type", "num_devices_per_node", "num_ranks", "message_size", "measured_dur"]
        )

    def _store_intermediate_data(self, df: pd.DataFrame) -> None:
        csv_path = self.output_path / "cloudai_nccl_test_prediction_input.csv"
        df.to_csv(csv_path, index=False)
        logging.debug(f"Stored intermediate predictor input data at {csv_path}")

    def _run_predictor(self, gpu_type: str, maya_inference_path: Path) -> pd.DataFrame:
        config_path = self.predictor_path / f"conf/{gpu_type}/{self.collective_type}.yaml"
        model_path = self.predictor_path / f"weights/{gpu_type}/{self.collective_type}.pkl"
        input_csv = self.output_path / "cloudai_nccl_test_prediction_input.csv"
        output_csv = self.output_path / "cloudai_nccl_test_prediction_output.csv"

        missing_files = [path for path in [config_path, model_path, input_csv] if not path.exists()]
        if missing_files:
            for file in missing_files:
                logging.warning(
                    f"Missing required file for inference: {file}. "
                    "Ensure predictor configuration and model files are correctly set up."
                )
            logging.warning("Skipping prediction due to missing required files.")
            return pd.DataFrame()

        command = [
            str(maya_inference_path),
            "--config",
            str(config_path),
            "--model",
            str(model_path),
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--log-level",
            "INFO",
        ]

        logging.debug(f"Running maya_inference with command: {' '.join(command)}")

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.debug(f"maya_inference output:\n{result.stdout}")

        except subprocess.CalledProcessError as e:
            logging.warning(
                f"maya_inference execution failed. Error message:\n{e.stderr}\n"
                "Check predictor logs and ensure model/config files are correct."
            )
            return pd.DataFrame()

        if not output_csv.exists():
            logging.warning(
                f"Expected output CSV {output_csv} not found after inference. "
                "Check maya_inference execution logs for errors."
            )
            return pd.DataFrame()

        predictions = pd.read_csv(output_csv)

        required_columns = {"num_devices_per_node", "num_ranks", "message_size", "dur"}
        missing_columns = required_columns - set(predictions.columns)
        if missing_columns:
            logging.warning(
                f"Missing required columns in prediction output: {', '.join(missing_columns)}. "
                "Ensure maya_inference is generating the correct format."
            )
            return pd.DataFrame()

        predictions.rename(columns={"dur": "predicted_dur"}, inplace=True)
        predictions["predicted_dur"] = predictions["predicted_dur"].round(2)

        return predictions[["num_devices_per_node", "num_ranks", "message_size", "predicted_dur"]]

    def _generate_prediction_report(self, df: pd.DataFrame, predictions: pd.DataFrame) -> None:
        df = df.merge(predictions, on="message_size", how="left")

        df["error_ratio"] = ((df["measured_dur"] - df["predicted_dur"]).abs() / df["measured_dur"]).round(2)

        csv_report_tool = CSVReportTool(self.output_path)
        csv_report_tool.set_dataframe(df[["message_size", "predicted_dur", "measured_dur", "error_ratio"]])
        csv_report_tool.finalize_report(Path("cloudai_nccl_test_prediction_csv_report.csv"))

        logging.debug("Saved predictor-based performance prediction report to CSV.")
