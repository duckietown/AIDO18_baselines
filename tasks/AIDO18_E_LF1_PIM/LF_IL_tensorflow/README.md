# AIDO18 template for the lane following task (LF)

## Imitation learning from real-world logs


For ease of use a Makefile is provided to support all steps in using this template. 


## Create new virtual environment with the dependencies

Creates virtual environment and installs all the necessary dependencies.

```
make install
``` 

## Download logs

Searches and downloads the logs(bag files) which are defined in the download_logs.py script.

```
make download
```

## Preprocessing of logs and save into HDF5 files

Extracts topics from bag files, synchronizes images with velocities and saves the data into HDF5 files.

```
make preprocess
```

## Train CNN

Trains CNN using the extracted images and velocities from the logs.

```
make learn
``` 

## Freeze TensorFlow graph and build image

Freezes TensorFlow graph and builds docker image.

```
make build-image
```

## Install NCSDK v2.05 and compile TensorFlow graph to Movidius graph

Installs the SDK tools of NCSDK v2.05 to enable compilaton to Movidius graphs. Then, compile TensorFlow graph to Movidius graph. 

```
make build-real-local-laptop
```

## Access RPi and install NCSDK v2.05 (only API)

Install only the SDK API in RPi, in order to be able to run Movidius graphs using the Movidius stick in RPi.

```
make build-real-local-rpi
```

## Access RPi and run lane-following-demo

Run lane following demo using the trained CNN

```
demo-real-local-rpi
```

