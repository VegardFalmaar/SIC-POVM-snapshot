GD_BIN = gradient_descent
DE_BIN = differential_evolution
DEVELOPMENT_BIN = development
TEST_BIN = test_all
BINs = $(TEST_BIN) $(GD_BIN) $(DE_BIN)

CXX = g++
CXXFLAGS = --std=c++17 -Wall
CXXOPTIM = -O3 -flto
DIVER_LIB = -ldiver -lgfortran
OBJ_DIR = obj
BIN_DIR = bin

# create directories for object files, executables and output if they do not
# already exist
$(info $(shell mkdir -p $(OBJ_DIR) $(BIN_DIR) output/loss output_diver))

SRCs = $(wildcard src/*.cpp)
OBJs = $(patsubst src/%.cpp, $(OBJ_DIR)/src_%.o, $(SRCs))

TESTSRCs = $(wildcard test/*.cpp)
TESTOBJs = $(patsubst test/%.cpp, $(OBJ_DIR)/test_%.o, $(TESTSRCs))


### targets to be used from command line

all: $(BINs)

execute: $(GD_BIN)
	$(BIN_DIR)/$^

test: $(TEST_BIN)
	$(BIN_DIR)/$^

clean:
	rm $(OBJ_DIR)/* $(BIN_DIR)/*

documentation:
	doxygen Doxyfile


### executables

$(GD_BIN): $(OBJ_DIR)/main_$(GD_BIN).o $(OBJs)
	$(CXX) -o $(BIN_DIR)/$@ $(CXXFLAGS) -fopenmp $(CXXOPTIM) $^

$(DE_BIN): $(OBJ_DIR)/main_$(DE_BIN).o $(OBJs)
	$(CXX) -o $(BIN_DIR)/$@ $(CXXFLAGS) $(CXXOPTIM) $^ $(DIVER_LIB)

$(TEST_BIN): $(OBJs) $(TESTOBJs)
	$(CXX) -o $(BIN_DIR)/$@ $(CXXFLAGS) $^

$(DEVELOPMENT_BIN): $(OBJ_DIR)/main_$(DEVELOPMENT_BIN).o $(OBJs)
	$(CXX) -o $(BIN_DIR)/$@ $(CXXFLAGS) $^


### compilation to object files

$(OBJ_DIR)/src_%.o: src/%.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CXXOPTIM) $^

$(OBJ_DIR)/test_%.o: test/%.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $^

$(OBJ_DIR)/main_$(GD_BIN).o: main/$(GD_BIN).cpp
	$(CXX) -c -o $@ $(CXXFLAGS) -fopenmp $(CXXOPTIM) $^

$(OBJ_DIR)/main_$(DE_BIN).o: main/$(DE_BIN).cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CXXOPTIM) $^

$(OBJ_DIR)/main_$(DEVELOPMENT_BIN).o: main/$(DEVELOPMENT_BIN).cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $^
