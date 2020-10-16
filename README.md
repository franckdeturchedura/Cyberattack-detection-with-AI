# Cyberattack detection with AI - ESIEE Paris Project x Thalès

## The aim

This project, in collaboration with Thalès, had to perform an attack recognition with AI from some probes (that we had to determine).

We finally focused on the most obvious ones, which are the CPU's temp, the nbr of processes in progress, the number of bits sent and received, and some others.

Results on our machine are quite good. However, since these probes send different values depending on the hardware, we have to train the network for each environment we want to use.

The network is a FFNN (Feed Forward Neural Network). To improve our results, we should use the logs (with an natural language processing or some other algorithms to extract the interesting information) with maybe a RNN (Recurrent Neural Network) to take into account previous sates (no more Markov assumptions).

## How to test it 

As said previously, you will have first to get data for your environment. This repository don't focus on this, but the scripts with the good probes are here available (`script_bash_auto.py`). By running the `run.py` script, the network will be trained with the `.npy` dataset given.

Then in de "Déploiement" folder is the script to make it run on you device.

