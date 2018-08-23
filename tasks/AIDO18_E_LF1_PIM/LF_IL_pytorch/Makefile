VIRTUALENV = LF_IL_pytorch_virtualenv
DUCKIETOWN = gym-duckietown
AG_VERSION = 2018.8.20

# Makefile to structure work on the AI-DO baselines.
#
# - install: Installs dependencies need to perform reinforcement learning
# - learn: Performs reinforcement learning and saves the learned agent

install:
				virtualenv -p python3 $(VIRTUALENV); \
				. $(VIRTUALENV)/bin/activate; \
				pip3 install -e .; \
				cd ..; \
				git clone -b julian https://github.com/duckietown/gym-duckietown.git; \
				cd $(DUCKIETOWN); \
				pip3 install -e .;

download:
				. $(VIRTUALENV)/bin/activate; \
				python src/dataset.py;

preprocess:
				. $(VIRTUALENV)/bin/activate; \
				python src/extract_data.py -s logdir -t datadir;

learn:
				. $(VIRTUALENV)/bin/activate; \
				python src/train.py -s datadir -t modeldir;

build-image:
				. $(VIRTUALENV)/bin/activate; \
				python src/freeze_graph.py; \
				docker build --tag duckietown/il-baseline-pytorch:gpu-$(AG_VERSION) .;

evaluate-sim-local:
				AGENT_VERSION=$(AG_VERSION); \
				docker-compose up;

evaluate-real-local:
				ls # still todo

submit:
				ls # to be determined
