## Dataset


    rosrun easy_logs download 20180108135529_a313 20180108141006_a313 20180108141155_a313 20180108141448_a313 20180108141719_a313


    rosrun easy_logs find 20180108135529_a313 

## Learn

First download logs to `logdir`(a function is provided in dataset.py, but the ipfs gateways are not reliable enough yet)






Then:

`./extract_data.py -s logidr -t datadir`

`./train.py -s datadir -t modeldir`

## Execute
Choose one model from the trained above. `modelpath` should point to it.

`./execute.py -m modelpath`

Although this script is not yet written, as most of it depends on communication over the ZMQs.




