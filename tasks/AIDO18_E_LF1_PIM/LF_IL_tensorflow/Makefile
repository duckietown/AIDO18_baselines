VIRTUALENV = LF_IL_virtualenv
AG_VERSION = 2018.8.18
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

build-image:
				source $(VIRTUALENV)/bin/activate; \
                                python src/freeze_graph.py; \
                                docker build --tag duckietown/il-baseline-tensorflow:$(AG_VERSION) .;

evaluate-sim-local:
				AGENT_VERSION=$(AG_VERSION); \
                                docker-compose up;

evaluate-real-local:
				ls # still todo

submit:
				ls # to be determined
