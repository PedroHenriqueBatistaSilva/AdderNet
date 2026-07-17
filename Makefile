# AdderNet — portable CPU build. Use `make NATIVE=1` for local AVX2/native flags.
CC ?= cc
ARCH := $(shell uname -m)
BUILD_DIR := build
SRC_DIR := src
NATIVE ?= 0
OPENMP ?= 1

BASE_FLAGS := -O3 -fPIC -Wall -Wextra
SIMD_FLAGS :=
SIMD_DEF :=
ifeq ($(NATIVE),1)
  ifeq ($(ARCH),x86_64)
    SIMD_FLAGS += -march=native -mpopcnt
    SIMD_DEF += -DHAVE_AVX2 -D__AVX2__
  else ifeq ($(ARCH),aarch64)
    SIMD_FLAGS += -march=native
    SIMD_DEF += -DHAVE_NEON -D__ARM_NEON
  endif
endif
ifeq ($(OPENMP),1)
  OMP_FLAGS := -fopenmp
else
  OMP_FLAGS :=
endif
CFLAGS := $(BASE_FLAGS) $(SIMD_FLAGS) $(SIMD_DEF) $(OMP_FLAGS)
LDFLAGS := -lm -lpthread $(OMP_FLAGS)

LIB_SO := $(BUILD_DIR)/libaddernet.so
HDC_SO := $(BUILD_DIR)/libaddernet_hdc.so

.PHONY: all clean install-libs test native
all: $(LIB_SO) $(HDC_SO)
native:
	$(MAKE) NATIVE=1 all

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_SO): $(SRC_DIR)/addernet.c $(SRC_DIR)/addernet.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared $(SRC_DIR)/addernet.c -o $@ $(LDFLAGS)

$(HDC_SO): $(SRC_DIR)/addernet.c $(SRC_DIR)/hdc_core.c $(SRC_DIR)/hdc_lsh.c $(SRC_DIR)/addernet_hdc.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

install-libs: all
	cp -f $(LIB_SO) addernet/
	cp -f $(HDC_SO) addernet/

test: install-libs
	python -m pytest -q

clean:
	rm -rf $(BUILD_DIR) addernet/libaddernet.so addernet/libaddernet_hdc.so
