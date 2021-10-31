CC = g++
CLAGS=-Wall
LINK = -lpthread

all: anns_client_non_blocking

anns_client_non_blocking: anns_client_non_blocking.cpp
	${CC} ${CLAGS} anns_client_non_blocking.cpp ${LINK} -o anns_client_non_blocking

.PHONY: clean, cleanall

cleanall: clean

clean:
	rm anns_client_non_blocking
