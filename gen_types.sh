#!/bin/bash

cd `dirname $0`
mkdir -p voxelproto

export PATH="$PATH:$PWD/lrpc"

protoc -I=proto/ --python_out=. --lrpc_out=. --descriptor_set_out=./voxelproto/voxelproto.desc proto/voxelproto/*.proto
