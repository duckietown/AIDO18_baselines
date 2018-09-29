# AIDO18 template for the lane following task (LF)

## Imitation learning from real-world logs


For ease of use a Makefile is provided to support all steps in using this template. 


## Installation of dependencies

Type

```
make install
``` 

## Downloading of logs

Type

```
make download
```

## Preprocessing of logs by extracting into HDF5

Type

```
make preprocess
```

## Train CNN

Type

```
make learn
``` 

## Freeze TensorFlow graph and build image

Type

```
make build-image
```

## Install NCSDK v2.05 in laptop and compile TensorFlow frozen graph to a movidius graph

Type

```
make build-real-local-laptop
```

## Access RPi and install NCSDK v2.05 (only API)

Type

```
make build-real-local-rpi
```

## Access RPi and run lane-following-demo

Type

```
demo-real-local-rpi
```

