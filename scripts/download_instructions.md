# CICIoT2023 download

1. Download the CICIoT2023 dataset from the Canadian Institute for Cybersecurity:
   https://www.unb.ca/cic/datasets/iotdataset-2023.html
   (registration or a request form may be required).
2. Extract the CSV files into `data/raw/`. The loader expects one CSV per
   capture/attack type with a `Label` column.
3. `data/` is gitignored; do not commit the dataset.
