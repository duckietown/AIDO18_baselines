VIRTUALENV = LF_IL_pytorch_virtualenv
# Makefile to structure work on the AI-DO baselines.
#
# - install: Installs dependencies need to perform reinforcement learning
# - learn: Performs reinforcement learning and saves the learned agent

install:
				virtualenv -p python3 $(VIRTUALENV); \
				source $(VIRTUALENV)/bin/activate; \
				pip install -e .;

download:
				source $(VIRTUALENV)/bin/activate; \
				python src/dataset.py;

preprocess:
				source $(VIRTUALENV)/bin/activate; \
				python src/extract_data.py -s logdir -t datadir;

learn:
				source $(VIRTUALENV)/bin/activate; \
				python src/train.py -s datadir -t modeldir;
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
