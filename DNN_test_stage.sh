#!/bin/bash

alpha=(0.0 0.25 0.5 0.75 1.0)
n_epochs=(30 30 30 30 30)
n_iter=(10 10 10 10 10)
tipo_reverb=(1 2 3 4 5 6)
#n_epochs=(1 1 1 1 1)
#n_iter=(1 1 1 1 1)
for idx in $(seq 0 4); do
	for jdx in $(seq 0 5); do
		echo "alpha = " ${alpha[idx]} " tipo_reverb = " ${tipo_reverb[jdx]}
		python test-stage-DNN-Chromagrams.py ${n_epochs[idx]} ${n_iter[idx]} ${alpha[idx]} ${tipo_reverb[jdx]}
	done
done
echo "Estamos ready"
exit 0
