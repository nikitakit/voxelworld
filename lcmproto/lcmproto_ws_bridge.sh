#!/bin/bash
cd `dirname $0`
python lcmproto_ws_bridge.py --descriptor=../voxelproto/voxelproto.desc
