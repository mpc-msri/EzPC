OCAMLC=ocamlc
GEN_FOLDER=gen
FCFILE=empty
FLAGS=-cflags '-w -11@8 -g' -use-menhir -use-ocamlfind -package str -package stdint # -yaccflags -v,--unused-tokens,--explain,--list-errors

all: clean main.byte
default: main
debug: FLAGS += -tag 'debug'
debug: all

main.byte:
	ocamlbuild $(FLAGS) $@
	cp main.byte ezpc

clean:
	ocamlbuild -clean -quiet
	rm -f main.byte ezpc

runtest: main.byte
	@mkdir -p $(GEN_FOLDER)
	./runtests.sh -genDir ./$(GEN_FOLDER) -genDiff

forceCopyTestAll: main.byte
	@mkdir -p $(GEN_FOLDER)
	./runtests.sh -genDir ./$(GEN_FOLDER) -genForceCopyAll

forceCopyTestFile: main.byte
	@mkdir -p $(GEN_FOLDER)
#Note: the following line should remain non-indented.
ifeq ($(FCFILE),empty)
	$(error FCFILE is not set)
endif
	./runtests.sh -genDir ./$(GEN_FOLDER) -forceCopyFile $(FCFILE)
