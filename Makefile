
PORT_OPTS ?= -p 127.0.0.1:8888:8888
IMAGENAME:=image-attrib

#Builds the docker image with a non-root user, with a user id and group
#id matching the user who runs `make docker-image`.  
HOST_USER ?= $(USER)
HOST_UID ?= $(shell id -u)
HOST_GID ?= $(shell id -g)

.PHONY: help
help:
	@echo "Makefile help:"
	@echo "  A Makefile and Dockerfile are included to simplify running the code."
	@echo "  To build the docker image:"
	@echo "    make docker-image"
	@echo "  To start jupyter lab to view the experiment notebooks:"
	@echo "    make run-jupyterlab"
	@echo "  To get a bash shell into the running container:"
	@echo "    make shell"
	@echo "  You can override makefile args as needed. E.g. to change the port to access the notebook:"
	@echo "     make run-jupyterlab PORT=8889"

docker-image:
	docker build --build-arg HOST_UID=$(HOST_UID) \
		--build-arg HOST_GID=$(HOST_GID) \
		--build-arg HOST_USER=$(HOST_USER) \
		-t $(IMAGENAME) -f ./Dockerfile .

run-jupyter: docker-image
	docker run -it --rm $(PORT_OPTS) -v ${PWD}/:/app \
	  $(IMAGENAME) 

shell: docker-image
	docker run -it --rm $(PORT_OPTS) -v ${PWD}/:/app \
	  $(IMAGENAME)  /bin/bash
	