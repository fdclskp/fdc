

.PHONY: main.cpp
all : a.out 
	./a.out
a.out : main.cpp
	g++ -std=c++11 main.cpp -lm


.PHONY: cmp
cmp: main.cpp
	g++ -std=c++11 main.cpp -lm

.PHONY: seg
seg:a.out
	catchsegv ./a.out

.PHONY: clean
clean :
	rm a.out
