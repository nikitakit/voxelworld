#!/bin/bash
set -e

if [ $(id -u) != 0 ]; then
    echo "This script must be run as root"
    exit 1
fi

# For explanation, see: http://lcm-proj.github.io/multicast_setup.html

# Set up link-local multicast IP 228.6.7.8
ifconfig lo multicast
route add -net 228.6.7.8 netmask 255.255.255.255 dev lo

# Increase UDP receive buffer size
sysctl -w net.core.rmem_default=2097152
sysctl -w net.core.rmem_max=2097152
