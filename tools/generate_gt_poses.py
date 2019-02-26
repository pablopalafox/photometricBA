import os
import numpy as np
import pandas as pd

## Read timestamps of our SfM dataset
timestamps = []
input_timestamps_filepath = "data/euroc_V1/timestamps.txt"
with open(input_timestamps_filepath, 'r') as f:
    for line in f:
        timestamps.append(int(line.rstrip('\n')))

## Read file with ALL GT poses
gt_poses_filepath = "data/mav0/state_groundtruth_estimate0/data.csv"
df = pd.read_csv(gt_poses_filepath)
gt_poses_timestamps = df[[df.columns[0]]].values

print(len(gt_poses_timestamps))

## Go through our dataset
# with open('data/mav0/state_groundtruth_estimate0/gt_poses_timestamps.txt', 'w') as file:
#     for timestamp in timestamps:
#         closes_timestamp = min(gt_poses_timestamps, key=lambda x: abs(x - timestamp))
#         closes_timestamp = str(closes_timestamp[0])
#         file.write(str(closes_timestamp) + "\n")


selected_gt_timestamps = []
with open('data/mav0/state_groundtruth_estimate0/gt_poses_timestamps.txt', 'r') as file:
    for line in file:
        selected_gt_timestamps.append(int(line.rstrip('\n')))


print(len(selected_gt_timestamps))
print(len(timestamps))

diffs = []
for i in range(len(selected_gt_timestamps)):
    diff = abs(selected_gt_timestamps[i] - timestamps[i])
    diffs.append(diff)
    print(selected_gt_timestamps[i], timestamps[i], diff)

