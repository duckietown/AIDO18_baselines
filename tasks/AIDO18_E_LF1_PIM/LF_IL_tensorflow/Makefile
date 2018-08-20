VIRTUALENV = LF_IL_virtualenv
# Makefile to structure work on the AI-DO baselines.
#
# - install: Installs dependencies need to perform reinforcement learning
# - learn: Performs reinforcement learning and saves the learned agent

install:
				virtualenv $(VIRTUALENV); \
				source $(VIRTUALENV)/bin/activate; \
				pip install -e .;

download:
				source $(VIRTUALENV)/bin/activate; \
				python src/download_logs.py;

preprocess:
				source $(VIRTUALENV)/bin/activate; \
				python src/extract_data.py;

learn:
				source $(VIRTUALENV)/bin/activate; \
				python src/cnn_training_tensorflow.py;
				#cd $(AGENT); \
        #docker-compose pull && docker-compose up
        #TODO: make sure learned agent is saved

container:
				source $(VIRTUALENV)/bin/activate; \
				python src/freeze_graph.py;
        #TODO: command to create new container

evaluate-sim-local:
        git clone https://github.com/duckietown/gym-duckietown
				cd $(AGENT) \
        docker run duckietown/$(AGENT)

evaluate-real-local:
				ls # still todo

submit:
				ls # to be determined
