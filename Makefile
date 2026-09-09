CC      = clang
CFLAGS  = -O2 -Wall -Wextra -Icore
CORE    = core/psk.c core/forward.c core/sketch.c
MODEL   = models/plus_sketch_q4.psk

host: platform/host.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $^ -lm

run: host
	./host $(MODEL)

clean:
	rm -f host *.svg

.PHONY: run clean
