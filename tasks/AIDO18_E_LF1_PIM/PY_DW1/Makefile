AGENT=gym-duckietown-agent

# Makefile to structure work on the AI-DO baselines.
#
# - install: Installs dependencies need to perform reinforcement learning
# - learn: Performs reinforcement learning and saves the learned agent

install:
        if [ ! -d "$(AGENT)" ]; then \
                git clone https://github.com/duckietown/$(AGENT).git; \
        fi; \
        cd $(AGENT); \
        pwd; \
        docker build -t duckietown/$(AGENT) .

learn:
        cd $(AGENT); \
        docker-compose pull && docker-compose up
        #TODO: make sure learned agent is saved

container:
        #TODO: command to create new container

evaluate-sim-local:
        git clone https://github.com/duckietown/gym-duckietown
        cd $(AGENT) \
        docker run duckietown/$(AGENT)

evaluate-real-local:
        ls # still todo

submit:
        ls # to be determined
