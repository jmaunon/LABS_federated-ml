PWD := $(CURDIR)

install:
	-cd $(PWD)/build && docker-compose rm -f
	docker-compose -f $(PWD)/build/docker-compose.yml up --build --force-recreate --no-deps