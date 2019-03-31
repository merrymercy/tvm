HALIDE_HOME = /home/mercy/project/gradient-halide

CC = g++
LIB = ../lib

CFLAGS = -std=c++11 -Wall -I$(HALIDE_HOME)/include -I$(HALIDE_HOME)/tools
LDFLAGS = -lpthread -ldl -L$(HALIDE_HOME)/lib -lHalide

all: halide_gen halide_eval

halide_gen: test_halide.cpp
	$(CC) $? $(CFLAGS) $(LDFLAGS) -o $@

halide_code: halide_gen
	HL_DEBUG_CODEGEN=1 ./halide_gen

halide_eval: test_halide.cpp
	$(CC) $? *.a $(CFLAGS) $(LDFLAGS) -D EVAL_MODE -o $@

test: halide_eval
	./halide_eval

.PHONY: clean
clean:
	rm -f *.a halide_gen halide_eval func_*.h

