# InstaCD Benchmarks
Following is a list of steps for how to setup the InstaCD benchmarking project. 

## Dependencies
The various benchmarking algorithms hosted in this project require a number of dependecies in both R and python, and leverage the rpy2 package in order to interoperate.

The requirements are specified in the associated requirements.txt for Python, and the associated requirements-*.R files for the R requirements. Further, many of the R dependecies require use of Bioconductor to install those packages. 

Therefore, we have built a docker image to orchestrate all these rquirements and dependencies and highly recommned leveraging this image.

### Getting the data
We leverage a fork of the Example-Causal-Datasets repository that is included as a submodule. Therefore, before you continue please make sure your submodules for this prooject have been fetched with the following command:
``` bash
git submodule --init update && cd submdules/example-causal-datasets && git checkout master
```

## Code Organization
All of the algorithms and source code for running the benchmarks are located in the instacd_benchmarks module. However, we leverage runners in order to organize and run various benchmarks. These runners are located in the runners directory. 

## Docker
To build the docker image run the following command:
```bash
docker build -t instacd:benchmarks .
```
Or you can pull the benchmarks image from Docker hub using:

``` bash 
docker pull yakaboskic/instacd:benchmarks
```

### Running a Runner
Using docker run, we can you run an experiment with
``` bash
docker run -v /tmp/data:/data instacd:benchmarks python3 runners/globe_runner.py --sheet_name continuous --format continuous --result_file /data/results.csv --network_result_file /data/network_results.dat
```
Notice that I have mounted a volume to the container with -v /tmp/data:/data and also used that associated container path as the place to store result files. This allows me to collect results as they come in as some experiments can take a while. 

### Using apptainer
If you are using apptainer on your computer/server instead of Docker you can use the following commands to build a sif file and run the command above using apptainer. 
``` bash
apptainer build instacd-benchmarks.sif docker://yakaboskic/instacd:benchmarks
```
Then youu can run the equivalent docker run command with:
``` bash
apptainer exec --bind /tmp/data:/data instacd-benchmarks.sif /bin/bash -c "cd /usr/src/app && python3 runners/globe_runner.py --sheet_name continuous --format continuous --result_file /data/results.csv --network_result_file /data/network_results.dat"
```
