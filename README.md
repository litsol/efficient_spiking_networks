# Efficient Spiking Networks


## Description

This code implements an Adaptive Spiking Recurrent Neural
Network. This is an adaptation of the Google Speech Commands (GSC)
portion of Bojian Yin's original [Adaptive (SRNN) Spiking Recurrent
Neural Network](https://github.com/byin-cwi/Efficient-spiking-networks.git)
code. The data input, partitioning and prepossessing machinery has
been rewritten and no longer depends on a custom pre-installed GSC
dataset, but uses the standard GSC distribution.  Experiment
parameters, formerly hard-coded in the simulation code have been moved
to a separate TOML configuration file.


## Getting Started

### Dependencies

We use the [Python Development
Manager (PDM)](https://pdm.fming.dev/latest/) to manage our package and
its dependencies. The PDM will manage all software dependencies.


### Installing

After cloning the repository, running PDM in the repository is all
that is necessary to download and install the other software and
data dependencies.

The first time you run a simulation the software will download the GSC
dataset into the directory named **dataroot** in the TOML
configuration file. This **dataroot** directory
must exist for the download to succeed.

Consider the default configuration file src/GSC/config-v1-cuda.toml.
The **dataroot** parameter there is assigned a relative path named
'google.'  For this configuration to succeed, create this directory,
create a symbolic link to an existing directory elsewhere, or assign
**dataroot** an exhaustive path of your choice.


```
git clone https://github.com/litsol/efficient_spiking_networks.git
cd efficient_spiking_networks
mkdir src/GSC/google
pdm setup
```


### Executing program

Before starting a simulation add the NVIDIA cudnn and nccl libraries
to your LD_LIBRARY_PATH. Adjust the paths as necessary.


```
export LD_LIBRARY_PATH=<Path to the repositry>/efficient_spiking_networks/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:<Path to the repositry>/efficient_spiking_networks/.venv/lib/python3.10/site-packages/nvidia/nccl/lib
```

The simulation program's name is srnn.py; it resides in the src/GSC
directory. The simulation program requires one argument - a TOML
configuration file. Here is the contents of the default configuration
file config-v1-cuda.toml.


```
[data]
    dataroot = "google"
    gsc_url = "speech_commands_v0.01"

[srnn]
    network_size = 256
    learning_rate = 3e-3
    epochs = 30
    batch_size = 32
    size = 16000
    sample_rate = 16000
    bias = true

[mel]
    delta_order = 2
    fmax = 4000
    fmin = 20
    n_mels = 40
    stack = true

[logger]
    level = "INFO"

[cuda]
    cuda = true
```


Use PDM to run a simulation.


```
cd src/GSC
pdm run srnn.py config-v1-cuda.toml
pdm run srnn.py config-v1-cuda.toml 2>&1 | tee config-v1-cuda.log # direct all logging output to a log file
```


## Authors

[Bojian Yin](mailto:byin@cwi.nl)

[Sander M. Bohté](mailto:S.M.Bohte@cwi.nl)

[Michael A. Guravage](mailto:guravage@literatesolutions.com)



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
