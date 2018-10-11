# AIDO18 template for the lane following task (LF)

## Imitation learning from real-world logs


For ease of use a Makefile is provided to support all steps in using this template. 


## Create new virtual environment with the dependencies

Create new virtual environment and install all the necessary dependencies.

Type: `make install` 

## Download logs

Search and download the logs(bag files) which are defined in the download_logs.py script.

Type: `make download`

## Preprocessing of logs and save into HDF5 files

Extract topics from bag files, synchronize images with velocities and save the data into HDF5 files.

Type: `make preprocess`

## Train CNN

Train CNN using the extracted images and velocities from the logs.

Type: `make learn` 

## Freeze TensorFlow graph and build image

Freeze TensorFlow graph and build docker image.

Type: `make build-image`

## Install NCSDK v2.05 and compile TensorFlow graph to Movidius graph

Install the SDK tools of NCSDK v2.05 to enable compilation to Movidius graphs. Then, compile TensorFlow graph to Movidius graph. 

Type: `make build-real-local-laptop`

## Access RPi and install NCSDK v2.05 (only API)

Install only the SDK API in RPi, in order to be able to run Movidius graphs using the Movidius stick in RPi.

Type: `make build-real-local-rpi`

## Access RPi and run lane-following-demo

Run lane following demo using the trained CNN.

Type: `demo-real-local-rpi`

