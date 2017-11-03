#!/bin/bash
set -e

if command -v tmux; then
    :
else
    echo "ERROR: tmux is not installed"
    exit 1
fi

if command -v gem; then
    :
else
    echo "ERROR: Ruby 'gem' command is not installed"
    exit 1
fi

if gem list teamocil -i | grep false; then
    echo "ERROR: teamocil not installed"
    echo "Please run 'sudo gem install teamocil'"
    exit 1
fi

./gen_types.sh

export TEAMOCIL_PATH="$PWD"

teamocil --layout $PWD/nodes.yml
