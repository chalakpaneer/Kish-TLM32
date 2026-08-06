CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c99 \
	-I../components/tlm_core/include

SRCS = test_generate.c \
	../components/tlm_core/src/tlm_math.c \
	../components/tlm_core/src/tlm_model.c \
	../components/tlm_core/src/tlm_weights_data.c

all: tlm32_host

tlm32_host: $(SRCS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) -lm

clean:
	rm -f tlm32_host

.PHONY: all clean
