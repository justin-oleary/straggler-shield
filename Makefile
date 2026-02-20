NVCC      := nvcc
GO        := go
CUDA_DIR  := cuda
BUILD_DIR := build

# sm_80 = Ampere (A100/A30). Adjust for your target:
#   sm_70  = Volta (V100)
#   sm_86  = Ampere consumer (RTX 3090)
#   sm_90  = Hopper (H100)
NVCC_FLAGS := -O2 -arch=sm_80

SO := $(CUDA_DIR)/libgpupulse.so

.PHONY: all cuda go go-stub test vet clean docker

all: cuda go

cuda: $(SO)

$(SO): $(CUDA_DIR)/gpu_pulse.cu $(CUDA_DIR)/gpu_pulse.h
	$(NVCC) $(NVCC_FLAGS) -shared -Xcompiler -fPIC \
		$(CUDA_DIR)/gpu_pulse.cu \
		-o $(SO) \
		-lcudart

go: cuda
	mkdir -p $(BUILD_DIR)
	LD_LIBRARY_PATH=$(CURDIR)/$(CUDA_DIR) \
	CGO_LDFLAGS="-L$(CURDIR)/$(CUDA_DIR) -lgpupulse -lcudart -lstdc++" \
	CGO_CFLAGS="-I$(CURDIR)/$(CUDA_DIR)" \
	$(GO) build -tags cuda -o $(BUILD_DIR)/straggler-shield ./cmd/agent

# non-CUDA build for CI lint/vet on machines without GPUs
go-stub:
	mkdir -p $(BUILD_DIR)
	$(GO) build -o $(BUILD_DIR)/straggler-shield-stub ./cmd/agent

test:
	$(GO) test ./...

vet:
	$(GO) vet ./...

clean:
	rm -f $(SO)
	rm -rf $(BUILD_DIR)

docker:
	docker build -t straggler-shield:dev .
