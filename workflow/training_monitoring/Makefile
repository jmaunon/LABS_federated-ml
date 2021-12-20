PWD := $(CURDIR)
JOB_ID := $(shell bash -c 'echo $$RANDOM')

install:
	-cd $(PWD)/build && docker-compose rm -f && rm -f .env
	cd $(PWD)/build && echo "JOB_ID=${JOB_ID}" > .env
	docker-compose -f $(PWD)/build/docker-compose.yml up --build --force-recreate --no-deps