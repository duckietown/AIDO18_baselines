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

## Preprocessing of logs by extracting into local folder

Type

```
make preprocess
```  

## Train the network on provided imitation learning data

Type

```
make learn
```  


## Run the network in the local simulator environment

Type

```
make evaluate-sim-local
```  
