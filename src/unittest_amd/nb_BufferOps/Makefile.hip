HIPCC=/opt/rocm/bin/hipcc

EXECUTABLE=bufferOpsCheck_hip

all: $(EXECUTABLE)

CXXFLAGS := -g -std=c++11  
LDFLAGS := 

bufferOpsCheck_hip: bufferOpsCheck_hip.o
	$(HIPCC) -o $@ $^

%.o: %.cpp
	$(HIPCC) $(CXXFLAGS) -c -o $@ $^ 

clean:
	rm -f $(EXECUTABLE) *.o 
