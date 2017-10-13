CXX           = clang++
CXXFLAGS        = -std=c++1z -stdlib=libc++ -O3 -ferror-limit=2 -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-sign-conversion -Wno-exit-time-destructors -Wno-float-equal -Wno-global-constructors -Wno-missing-declarations -Wno-unused-parameter -Wno-padded -Wno-shadow -Wno-weak-vtables -Wno-missing-prototypes -Wno-unused-variable -ferror-limit=1 -Wno-deprecated -Wno-conversion -Wno-double-promotion -fPIC -Wno-documentation -Wno-old-style-cast -Wno-reserved-id-macro -Wno-documentation-unknown-command -Wno-undef
INCPATH       = -Iinclude -Igame
LINK          = $(CXX)
LFLAGS        = -lc++ -lc++abi -O3
DEL_FILE      = rm -f

####### Output directory
OBJECTS_DIR   = ./obj
BIN_DIR       = ./bin

all: lspe

clean:
	rm -rf $(OBJECTS_DIR)/*.o
	rm -rf $(BIN_DIR)/*

lspe.o: src/lspe.cpp
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o $(OBJECTS_DIR)/lspe.o src/lspe.cpp

lspe: lspe.o
	$(LINK) -o $(BIN_DIR)/lspe $(OBJECTS_DIR)/lspe.o $(LFLAGS)

