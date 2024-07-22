#! /bin/bash

cd data

#unzip the rapid movement dataset
unzip ugv_rapid_movement_dataset.zip
rm ugv_rapid_movement_dataset.zip

#unzip and prepare the seen dataset
unzip ugv_seen_dataset.zip
rm ugv_seen_dataset.zip

#unzip and prepare the unseen dataset
unzip ugv_unseen_dataset.zip
rm ugv_unseen_dataset.zip
