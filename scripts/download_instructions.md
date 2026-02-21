# CICIoT2023 download

1. Request the dataset from the Canadian Institute for Cybersecurity:
   https://www.unb.ca/cic/datasets/iotdataset-2023.html
   The download form is at http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/.
   Registration or a request form may be required.
2. Extract the `CSV/` directory into `data/raw/`. The loader searches recursively,
   so keeping subdirectories is fine.
3. Expect 169 CSV captures, about 46 numeric features plus a `label` column, and
   roughly 13 GB uncompressed. The loader samples per capture and never loads the
   full dataset into memory.
4. `data/` is gitignored; do not commit the dataset.

The official header is:

```text
flow_duration,Header_Length,Protocol Type,Duration,Rate,Srate,Drate,
fin_flag_number,syn_flag_number,rst_flag_number,psh_flag_number,
ack_flag_number,ece_flag_number,cwr_flag_number,ack_count,syn_count,
fin_count,urg_count,rst_count,HTTP,HTTPS,DNS,Telnet,SMTP,SSH,IRC,TCP,
UDP,DHCP,ARP,ICMP,IPv,LLC,Tot sum,Min,Max,AVG,Std,Tot size,IAT,Number,
Magnitue,Radius,Covariance,Variance,Weight,label
```
