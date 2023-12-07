import seisbench
import seisbench.util
from seisbench.data.base import BenchmarkDataset, WaveformDataWriter

from pathlib import Path
import h5py
import pandas as pd
import numpy as np

class CWBSN_noise(BenchmarkDataset):
    def __init__(self, level=-1, **kwargs):
        if level == -1:
            self.level = ['3', '4']
        elif level == 3:
            self.level = ['3']
        elif level == 4:
            self.level = ['4']

        citation = ()
        license = ""

        super().__init__(citiation=citation, license=license, **kwargs)

    def _download_dataset(self, writer: WaveformDataWriter, basepath=None, **kwargs):
        path = self.path

        if basepath is None:
            raise ValueError('No cached version of CWBSN found. ')

        basepath = Path(basepath)

        # Data format
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": 100,
            "measurement": "velocity/acceleration",
            "unit": "cmps/cmps2",
        }

        self.load(writer, basepath)

    @staticmethod
    def set_split(year):
        if year == "21" or year == "20":
            return "test"
        elif year == "19":
            return "dev"
        else:
            return "train"

    def load(self, writer, basepath):
        total_trace = 0

        for l in self.level:
            print('level: ', l)

            meta_path = 'metadata_level' + l + '.csv'
            metadata = pd.read_csv(basepath / meta_path)

            hdf5_path = 'CWB_noise_level' + l + '.hdf5'
            with h5py.File(basepath / hdf5_path) as f:
                gdata = f['data']
                for _, row in metadata.iterrows():
                    row = row.to_dict()
                    row['split'] = self.set_split(row['trace_name'][:2])
                    
                    waveforms = gdata[row['trace_name']][()]

                    writer.add_trace(row, waveforms)
                    total_trace += 1

        writer.set_total(total_trace)

