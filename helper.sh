#!/bin/bash

# Define the three lists of arguments
arg_list_1=(100 200 400 800 1600 3200 6400 12800)
arg_list_2=({0..31})
arg_list_3=(0.7 0.98 0.99 0.999 1)

# Loop through the arguments and execute the Python script with each combination
for arg1 in "${arg_list_1[@]}"; do
  for arg2 in "${arg_list_2[@]}"; do
    for arg3 in "${arg_list_3[@]}"; do
      python3 ./Tommaso/simulation_Tommaso.py "$arg1" "$arg2" "$arg3"
    done
  done
done