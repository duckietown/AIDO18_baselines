VIRTUALENV = LF_IL_virtualenv
AG_VERSION = 2018.8.18

USERNAME = megaduck
DUCKIEBOT = megabot08
PASSWORD = megacity

# Makefile to structure work on the AI-DO baselines.
#
# - install: Installs dependencies need to perform reinforcement learning
# - learn: Performs reinforcement learning and saves the learned agent

install:
				virtualenv $(VIRTUALENV); \
				. $(VIRTUALENV)/bin/activate; \
				pip install -e .;

download:
				. $(VIRTUALENV)/bin/activate; \
				python src/download_logs.py;

preprocess:
				. $(VIRTUALENV)/bin/activate; \
				bash -c "source ~/duckietown/environment.sh; python src/extract_data.py;"; \


learn:
				. $(VIRTUALENV)/bin/activate; \
				python src/cnn_training_tensorflow.py;
				#cd $(AGENT); \
		        #docker-compose pull && docker-compose up
		        #TODO: make sure learned agent is saved

build-image:
				. $(VIRTUALENV)/bin/activate; \
                python src/freeze_graph.py; \
                docker build --tag duckietown/il-baseline-tensorflow:$(AG_VERSION) .;

evaluate-sim-local:
				AGENT_VERSION=$(AG_VERSION); \
                                docker-compose up;


build-real-local-laptop:
				
				. $(VIRTUALENV)/bin/activate; \
				wget -nc https://ncs-forum-uploads.s3.amazonaws.com/ncsdk/ncsdk-02_05_00_02-full/ncsdk-2.05.00.02.tar.gz; \
				if [ -d ncsdk-2.05.00.02 ]; then \
					echo "NCSDK is installed"; \
				else \
					tar xvf ncsdk-2.05.00.02.tar.gz; \
					cd ncsdk-2.05.00.02; \
					make install; \
				fi \

				echo "Compile TensorFlow graph to Movidius graph"; \
				. $(VIRTUALENV)/bin/activate; \
				mkdir movidius_graph; \
				mvNCCompile -s 12 tensorflow_logs/batch=100,lr=0.0001,optimizer=GDS,epochs=1000/frozen_graph/frozen_graph.pb -in x -on ConvNet/fc_layer_2/BiasAdd -o movidius_graph/lane_following.graph; \
				echo "Frozen graph is compiled to Movidius graph and saved in ./movidius_graph/ directory"; \

# for Breandan: 
# copy the <CNN REPO> to duckietown org
# make it generate an image called "rpi-cnn-lane-following" that 
#  1) inherits from the appropriate base image
#  2) install the ncsdk (as below)
#  3) install the ROS packages and launch file in <CNN REPO>
#  4) has CMD roslaunch cnn_lane_following /cnn_lane_following/cnn_lane_following.launch veh:=$(VEHICLE_NAME) local:=false 
build-real-local-rpi:
				
				sudo apt-get install sshpass
				sshpass -p '$(PASSWORD)' ssh $(USERNAME)@$(DUCKIEBOT) '\
				wget -nc https://ncs-forum-uploads.s3.amazonaws.com/ncsdk/ncsdk-02_05_00_02-full/ncsdk-2.05.00.02.tar.gz; \
				[ ! -d ncsdk-2.05.00.02 ] && tar xvf ncsdk-2.05.00.02.tar.gz; \
				cd ncsdk-2.05.00.02; \
				sudo make api; \
				sudo apt-get install python-opencv; \
				'

# create a Dockerfile which does the following:
# Dockerfile = '''
# FROM duckietown/rpi-cnn-lane-following
# COPY movidius_graph to /movidius_graph

demo-real-local-rpi:
				
				sshpass -p '$(PASSWORD)' ssh $(USERNAME)@$(DUCKIEBOT) '\
				bash -c "source ~/duckietown/environment.sh; roslaunch cnn_lane_following cnn_lane_following.launch veh:=$(VEHICLE_NAME) local:=false" '


evaluate-real-local:

				ls # still todo

				ls # image to run only camera and wheel driver nodes
				ls # image to run inverse kinematics nodes
				ls # image to run cnn_node
				ls # script to evaluate in real city

submit:
				ls # to be determined
