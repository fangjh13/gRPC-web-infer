python3 -m grpc_tools.protoc \
        -I ./protos \
        --python_out=./protos  \
        --grpc_python_out=./protos \
        ./protos/infer.proto
echo 'success generate python protobuf files in ./protos'
