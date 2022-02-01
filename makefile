build:
	@docker build -t haiku .

run:
	@docker run -dp 3000:3000 --name haiku haiku

stop:
	@docker stop haiku

clear:
	@docker rm haiku

up: build run

down: stop clear

all: down up