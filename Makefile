GIT_USER := $(shell git config user.name)
GIT_EMAIL := $(shell git config user.email)

.PHONY: style
style: black isort lint

.PHONY: black
black:
	poetry run black .

.PHONY: isort
isort:
	poetry run isort .

.PHONY: lint
lint:
	@find . -type f -name "*.py" ! -path "./notebooks/*" ! -path "*/.ipynb_checkpoints/*" | xargs poetry run pylint

.PHONY: mypy
mypy:
	@find . -type f -name "*.py" ! -path "./notebooks/*" ! -path "*/.ipynb_checkpoints/*" | xargs poetry run mypy

.PHONY: test
test: test-unit

.PHONY: test-unit
test-unit:
	poetry run pytest -vv tests/unit

.PHONY: build-docker-dev-image
build-docker-dev-image:
	docker build \
	-t synth-data-gen-from-text-image \
	-f Dockerfile.dev \
	--build-arg USER="${USER}" \
	--build-arg GIT_USER="${GIT_USER}" \
	--build-arg GIT_EMAIL="${GIT_EMAIL}" \
	.

.PHONY: create-docker-dev-container
create-docker-dev-container:
	docker run -dit \
	--name synth-data-gen-from-text-container \
	--user root \
	-e CHOWN_HOME=yes -e CHOWN_HOME_OPTS="-R" \
	-e RESTARTABLE=yes \
	-e GRANT_SUDO=yes \
	-v ${PWD}:/root/${USER}/synth-data-gen-from-text \
	-w /root/${USER}/synth-data-gen-from-text \
	synth-data-gen-from-text-image

.PHONY: start-docker-dev-container
start-docker-dev-container:
	docker start synth-data-gen-from-text-container

.PHONY: exec-docker-dev-container
exec-docker-dev-container:
	docker exec -it synth-data-gen-from-text-container /bin/bash
