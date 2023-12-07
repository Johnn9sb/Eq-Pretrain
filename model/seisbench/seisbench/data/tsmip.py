import seisbench
import seisbench.util
import seisbench.data as sbd
from seisbench.data.base import BenchmarkDataset, WaveformDataWriter

from pathlib import Path
import h5py
import pandas as pd
import numpy as np


class TSMIP(BenchmarkDataset):
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

        # TSMIP: containing 2011~2020
        years = [str(y) for y in range(2011, 2021)]

        if basepath is None:
            raise ValueError("No cached version of TSMIP found. ")

        basepath = Path(basepath)

        # Data format
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": 200,
            "measurement": "acceleration",
            "unit": "cmps2",
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
        if year == "2019":
            return "dev"
        elif year == "2020":
            return "test"
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

    def event_load(self, writer, basepath, years):
        # 以 event 作為基本單位 load 進 dataset

        total_trace = 0
        for y in years:
            print("years: ", y)

            # Loading metadata (ex. 2020)，之後一次 load 所有年份的資料進來
            meta_path = "metadata_" + y + ".csv"
            metadata = pd.read_csv(basepath / meta_path)

            # set split
            metadata["split"] = self.set_split(y)

            # Adding traces (ex. 2020)，之後一次 load 所有年份的資料進來
            hdf5_path = "chunks_" + y + ".hdf5"
            with h5py.File(basepath / hdf5_path) as f:
                gdata = f["data"]

                prev_event = ""
                event_metadata = {}
                waveform = []
                station_info = []
                p_arrival = []

                for i, row in metadata.iterrows():
                    row = row.to_dict()

                    # get event name
                    tmp = row["trace_name"]
                    tmp = tmp.split("_")[:2]
                    cur_event = "_".join(tmp)

                    # check whether the trace belong to previous event
                    if cur_event != prev_event:

                        # add previous event into writer
                        if prev_event != "" or i == len(metadata) - 1:
                            waveform = self.padding(waveform, p_arrival)
                            waveform = np.array(waveform)

                            station_info = np.array(station_info)

                            event_metadata["Event_metadata"] = station_info
                            event_metadata["trace_name"] = prev_event
                            event_metadata["split"] = row["split"]

                            # add to writer
                            print(
                                f"waveform: {waveform.shape}, station_info: {station_info.shape}"
                            )

                            writer.add_trace(event_metadata, waveform)
                            total_trace += 1

                            if i == len(metadata) - 1:
                                break

                        event_metadata = {}
                        event_metadata["Origin_Time(GMT+0)"] = row["source_origin_time"]
                        event_metadata["Latitude(o)"] = row["source_latitude_deg"]
                        event_metadata["Longitude(o)"] = row["source_longitude_deg"]
                        event_metadata["Depth(km)"] = row["source_depth_km"]
                        event_metadata["Mag"] = row["source_magnitude"]

                        waveform = []
                        waveform.append(gdata[row["trace_name"]][()])

                        p_arrival = []
                        p_arrival.append(row["trace_p_arrival_sample"])

                        station_info = []
                        station_info.append(
                            [
                                row["station_latitude_deg"],
                                row["station_longitude_deg"],
                                row["station_elevation_m"],
                            ]
                        )

                        prev_event = cur_event
                    else:
                        waveform.append(gdata[row["trace_name"]][()])
                        station_info.append(
                            [
                                row["station_latitude_deg"],
                                row["station_longitude_deg"],
                                row["station_elevation_m"],
                            ]
                        )
                        p_arrival.append(row["trace_p_arrival_sample"])

        # Total number of traces
        writer.set_total(total_trace)

        return writer

    @staticmethod
    def padding(waveform, p_arrival):
        # for event_load, padding different traces into same waveform length to store into hdf5

        # find the longest length of traces
        max_len = 0
        for w in range(len(waveform)):
            if waveform[w].shape[1] > max_len:
                max_len = waveform[w].shape[1]

        # padding
        padded_waveform = []
        for i, w in enumerate(waveform):
            pad_len = max_len - w.shape[1]

            if pad_len == 0:
                padded_waveform.append(w)

                continue

            tri = p_arrival[i]
            wave = np.empty((3, max_len))

            mean = (
                np.mean(w[:, : tri - 200], axis=1)
                if tri - 200 >= 0
                else np.mean(w[:, :tri], axis=1)
            )
            mean[np.isnan(mean)] = 0

            wave[0] = np.hstack((w[0], mean[0].repeat(pad_len)))
            wave[1] = np.hstack((w[1], mean[1].repeat(pad_len)))
            wave[2] = np.hstack((w[2], mean[2].repeat(pad_len)))

            padded_waveform.append(wave)

            del wave

        return padded_waveform
