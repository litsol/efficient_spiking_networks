# Efficient Spiking Networks


## Description

This code implements an Adaptive Spiking Recurrent Neural
Network. This is an adaptation of the Google Speech Commands (GSC)
porton of Bojian Yin's original [Adaptive (SRNN) Spiking Recurrent
NeuralNetwork](https://github.com/byin-cwi/Efficient-spiking-networks.git)
code. The data input, partitioning and preprocessing machinery has
been rewritten and no longer depends on a custom pre-installed GSC
dataset, but uses the standard GSC distribution.  Experiment
parameters, formerly hard-coded in the simulation code have beem moved
to a separate TOML configuration file.


## Getting Started

### Dependencies

We use the [Python Development
Manager (PDM)](https://pdm.fming.dev/latest/) to manage our package and
its dependencies. The PDM will manage all software dependencies.


### Installing

After cloning the repository, running PDM in the repository is all
that is necessary to download and install the other software
dependencies.


```
git clone https://github.com/litsol/efficient_spiking_networks.git
cd efficient_spiking_networks
pdm setup
```


The first time you run a simulation the software will download the GSC
dataset into the directory you specify as **dataroot** in the
configuration file read by the software. This **dataroot** directory
must exist for the dowload to succeed. Consider the default
configuration file src/GSC/config-v1-cuda.toml. The **dataroot**
parameter is assigned a relative path in the same directory named
'google.' Either create this directory or create a symbolic link to an
existing directory elsewhere for this configuration to succeed.


### Executing program

Before starting a simulation add the NVIDIA cudnn and nccl libraries
to your LD_LIBRARY_PATH. Adjusts the paths as necessary.



```
export LD_LIBRARY_PATH=<Path to the repositry>/efficient_spiking_networks/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:<Path to the repositry>/efficient_spiking_networks/.venv/lib/python3.10/site-packages/nvidia/nccl/lib
```


The simulation executable's name is srnn.py; it resides in the src/GSC
directory. Use pdm to launch it. The simulation executable requires
one argument - a TOML configuration file.


```
cd src/GSC
pdm run srnn.py config-v1-cuda.toml
pdm run srnn.py config-v1-cuda.toml 2>&1 | tee config-v1-cuda.log # direct all logging output to a log file
```


## Authors

[Michael A. Guravage](mailto:guravage@literatesolutions.com)

[Sander M. Bohté](mailto:S.M.Bohte@cwi.nl)

## Version History

* 0.1
   * Initial Release


## License

This project is licensed under the Creative Commons & Mozilla Public Licenses - see the LICENSE.md file for details


## Acknowledgments

[Adaptive (SRNN) Spiking Recurrent Neural Network](https://github.com/byin-cwi/Efficient-spiking-networks.git)


## References

[1]. Bojian Yin, Federico Corradi, Sander M. Bohté. **Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks**

[2]. Bojian Yin, Federico Corradi, Sander M. Bohté. **Effective and efficient computation with multiple-timescale spiking recurrent neural networks**
