#!/bin/bash

n_epochs=$1
# n_iter=$2
#alpha=(0.0 0.25 0.5 0.75 1.0)
alpha=0.0
n_alpha=${#alpha[@]}
#tipo_reverb=(1 2 3 4 5) # Anecoica regrabaciones.
tipo_reverb=1
n_reverb=${#tipo_reverb[@]}
echo "Parámetros seleccionados: "
echo "Número de Epochs: "${n_epochs}
echo "Etapa de Entrenamiento."
for idx in $(seq 0 $((n_alpha-1))); do
	echo "alpha = "${alpha[idx]} " n_epochs = "${n_epochs} 
	python main-dnnchromagrams.py ${n_epochs} ${alpha[idx]}
done

echo "Etapa de Test"
for idx in $(seq 0 $((n_alpha-1))); do
	for jdx in $(seq 0 $((n_reverb-1))); do
		echo "alpha = " ${alpha[idx]} " tipo_reverb = " ${tipo_reverb[jdx]}
		python test-stage-DNN-Chromagrams.py ${n_epochs} ${alpha[idx]} ${tipo_reverb[jdx]}
	done
done
echo "Etapa de Evaluación del Sistema"
for idx in $(seq 0 $((n_alpha-1))); do
	echo "Evaluaciones para alpha = " ${alpha[idx]}
	python evaluation-stage-DNN-Chromagrams.py ${n_epochs} ${alpha[idx]}

done

exit 0
