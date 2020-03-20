#!/bin/bash

rm -rf logs

for f in parameters/*
do
  echo Experiment with $f parameters started
  python experiment.py --parameters=$f
  echo Ended!
done