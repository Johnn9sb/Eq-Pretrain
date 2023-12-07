import seisbench
import seisbench.util
from seisbench.data.base import BenchmarkDataset, WaveformDataWriter

from pathlib import Path
import h5py
import pandas as pd
import numpy as np


class CWBSN(BenchmarkDataset):
    def __init__(self, loading_method="full", **kwargs):

        self.loading_method = loading_method

        # ======================= #
        # TODO: citation, license #
        # ======================= #
        citation = ()
        license = ""
        super().__init__(citation=citation, license=license, **kwargs)

    def _download_dataset(self, writer: WaveformDataWriter, basepath=None, **kwargs):
        path = self.path

        # CWBSN: containing 2012~2021
        years = [str(y) for y in range(2012, 2022)]

        if basepath is None:
            raise ValueError("No cached version of CWBSN found. ")

        basepath = Path(basepath)

        # Data format
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": 100,
            "measurement": "velocity/acceleration",
            "unit": "cmps/cmps2",
        }

        # 檢查 10 年資料有無缺漏，並下載缺漏的年份資料
        missing_metadata, missing_waveform = self.check(basepath, years)

        # ============================================= #
        # TODO: 之後上傳檔案到網路上後，新增下載方式的 code
        # ============================================= #

        # Loading dataset
        if self.loading_method == "simple":
            print("Simple loading...")
            writer = self.simple_load(writer, basepath, years)
        elif self.loading_method == "single":
            print("Single loading...")
            writer = self.single_load(writer, basepath, years)
        elif self.loading_method == "event":
            writer = self.event_load(writer, basepath, years)
        elif self.loading_method == "full":
            print("Fully loading...")
            writer = self.full_load(writer, basepath, years)

    @staticmethod
    def set_split(year):
        if year == "2021" or year == "2020":
            return "test"
        elif year == "2019":
            return "dev"
        else:
            return "train"

    @staticmethod
    def check(basepath, years):
        metadata = []
        waveform = []

        for y in years:
            path = "metadata_" + str(y) + ".csv"
            if not (basepath / path).is_file():
                metadata.append(y)

            path = "chunks_" + str(y) + ".hdf5"
            if not (basepath / path).is_file():
                waveform.append(y)

        return metadata, waveform

    def full_load(self, writer, basepath, years):
        # full load: 全部都 load 進 dataset

        total_trace = 0
        for y in years:
            print("years: ", y)
            # Loading metadata (ex. 2020)，之後一次 load 所有年份的資料進來
            meta_path = "metadata_" + y + ".csv"
            metadata = pd.read_csv(basepath / meta_path)

            metadata["split"] = self.set_split(y)

            # Adding traces (ex. 2020)，之後一次 load 所有年份的資料進來
            hdf5_path = "chunks_" + y + ".hdf5"
            with h5py.File(basepath / hdf5_path) as f:
                gdata = f["data"]
                for _, row in metadata.iterrows():
                    row = row.to_dict()

                    # Adding trace only when waveform is available
                    waveforms = gdata[row["trace_name"]][()]

                    writer.add_trace(row, waveforms)
                    total_trace += 1

        # Total number of traces
        writer.set_total(total_trace)

        return writer

    def simple_load(self, writer, basepath, years):
        # simple load: 把 trace_completeness=1 的都 load 進來

        total_trace = 0
        for y in years:
            print("years: ", y)
            # Loading metadata (ex. 2020)，之後一次 load 所有年份的資料進來
            meta_path = "metadata_" + y + ".csv"
            metadata = pd.read_csv(basepath / meta_path)

            metadata["split"] = self.set_split(y)

            # Adding traces (ex. 2020)，之後一次 load 所有年份的資料進來
            hdf5_path = "chunks_" + y + ".hdf5"
            with h5py.File(basepath / hdf5_path) as f:
                gdata = f["data"]
                for _, row in metadata.iterrows():
                    row = row.to_dict()

                    # Adding trace only when waveform is available
                    if row["trace_completeness"] == 1:
                        waveforms = gdata[row["trace_name"]][()]

                        writer.add_trace(row, waveforms)
                        total_trace += 1

        # Total number of traces
        writer.set_total(total_trace)

        return writer

    def single_load(self, writer, basepath, years):
        # single load: load 只包含一個事件的 traces

        total_trace = 0
        for y in years:
            print("years: ", y)
            # Loading metadata (ex. 2020)，之後一次 load 所有年份的資料進來
            meta_path = "metadata_" + y + ".csv"
            metadata = pd.read_csv(basepath / meta_path)

            metadata["split"] = self.set_split(y)

            # Adding traces (ex. 2020)，之後一次 load 所有年份的資料進來
            hdf5_path = "chunks_" + y + ".hdf5"
            with h5py.File(basepath / hdf5_path) as f:
                gdata = f["data"]
                for _, row in metadata.iterrows():
                    row = row.to_dict()

                    # Adding trace only when waveform is available
                    if (
                        row["trace_completeness"] == 1
                        and row["trace_number_of_event"] == 1
                    ):
                        waveforms = gdata[row["trace_name"]][()]

                        writer.add_trace(row, waveforms)
                        total_trace += 1

        # Total number of traces
        writer.set_total(total_trace)

        return writer
