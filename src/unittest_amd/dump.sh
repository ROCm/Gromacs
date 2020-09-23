#!/bin/bash


if [ "$1" == "gfx906" ]; then
    echo "gfx906"
    cp gfx906-pmc.txt input.txt
else 
    if [ "$1" == "gfx908" ]; then
        echo "gfx908"
        cp gfx908-pmc.txt input.txt
    else
	echo "Please enter sh dump.sh gfx908 or sh dump.sh gfx906"
        exit 1
    fi
fi

mkdir result
mkdir result/adh;

cd bonded && make -f Makefile.$1
cd ../nb && make -f Makefile.$1
cd ../pme && make -f Makefile.$1

cd ../../nb && sh dump-nb.sh && cd ../pme && sh dump-pme.sh && cd ../bonded && sh dump-bonded.sh
