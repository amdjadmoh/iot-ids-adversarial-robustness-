# CICIoT2023 download

## What this repository actually expects

This study does **not** start from the 169 per-capture CSVs of the official
release. It uses a **pre-split** copy of CICIoT2023 that already ships the three
splits as separate files:

```text
data/raw/
  train/train.csv
  validation/validation.csv
  test/test.csv
```

`src/data.py` looks for files whose stem is `train`, `validation` or `test` and,
when it finds them, adopts those roles **verbatim**: no sampling-based
re-partitioning, no capture-level holdout, no leakage-repair pass. The fractions
in `configs/default.yaml` (`split.val_fraction`, `split.test_fraction`) and the
`split.capture_holdout_when_splitting` switch are only consulted on the fallback
path, i.e. when the raw directory instead contains per-capture CSVs.

Consequences to keep in mind when reading `results/`:

- There is **no capture-level separation between splits**. Flows from one capture
  may appear in both train and test, so the baseline numbers may be optimistic
  relative to a proper unseen-capture evaluation. See the Limitations section of
  `README.md` and `report/report.md`.
- The split is therefore *fixed by whoever packaged this copy*, not by this
  repository. `results/split_summary.json` and `results/split_files.csv` record
  the row counts and the source file behind every row actually used.

### TODO (owner)

The exact provenance of this pre-split copy is **not** recorded anywhere in the
repository and is deliberately not guessed here. Fill in all of the following:

- [ ] Where the archive was obtained (URL, mirror, course or lab distribution).
- [ ] The packaging date / dataset version, if the distributor states one.
- [ ] The exact archive filename and any checksum you can compute.
- [ ] How the three files were produced from the original per-capture CSVs
      (which rows went where, and whether any resampling or relabelling happened).
- [ ] The licence or terms that apply to this redistribution.

Until that list is complete, treat any claim about the split as unverified.

## If you want to rebuild from the official release instead

1. Request the dataset from the Canadian Institute for Cybersecurity:
   https://www.unb.ca/cic/datasets/iotdataset-2023.html
   The download form is at http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/.
   Registration or a request form may be required.
2. Extract the `CSV/` directory into `data/raw/`, **without** the pre-split
   `train/`, `validation/` and `test/` files described above. The loader searches
   recursively, so keeping subdirectories is fine.
3. Expect 169 CSV captures, about 46 numeric features plus a `label` column, and
   roughly 13 GB uncompressed. The loader samples per capture and never loads the
   full dataset into memory.
4. `data/` is gitignored; do not commit the dataset.

On that path the loader creates the splits itself, keeping capture files intact
when a label has at least three of them. Numbers produced that way will differ
from the ones in `results/`, because the splits are different.

The official header is:

```text
flow_duration,Header_Length,Protocol Type,Duration,Rate,Srate,Drate,
fin_flag_number,syn_flag_number,rst_flag_number,psh_flag_number,
ack_flag_number,ece_flag_number,cwr_flag_number,ack_count,syn_count,
fin_count,urg_count,rst_count,HTTP,HTTPS,DNS,Telnet,SMTP,SSH,IRC,TCP,
UDP,DHCP,ARP,ICMP,IPv,LLC,Tot sum,Min,Max,AVG,Std,Tot size,IAT,Number,
Magnitue,Radius,Covariance,Variance,Weight,label
```
