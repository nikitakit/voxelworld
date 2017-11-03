#!/bin/bash

cd `dirname $0` >/dev/null
pushd .. >/dev/null
export PATH="$PATH:$PWD"
popd >/dev/null

protoc -I=. --python_out=. --descriptor_set_out=./types.desc --lrpc_out=. ./*.proto
