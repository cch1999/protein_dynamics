#!/bin/bash

# CUDA SDK setup for bash & zsh.  Source this via ". /path/to/cuda/setup.bash"

# First, establish location of this script; assume CUDA install is there.
# If you are copying this script, remember to maintain its relative location.
if [ -n "$BASH_VERSION" ]; then
        export CUDADIR=$( dirname $( readlink -f -n $BASH_SOURCE ) )
elif [ -n "$ZSH_VERSION" ]; then
        export CUDADIR=$( dirname $( readlink -f -n $0 ) )
fi
export CUDA_PATH=${CUDADIR}


ARCH=`/usr/bin/arch`
# Need to quote == for zsh string equality test
# http://www.zsh.org/mla/users/2011/msg00161.html
if [ $ARCH '==' x86_64 ]
  then
        export CUDAPATH="${CUDADIR}/lib64:${CUDADIR}/lib"
        export CUDA_SEARCH_PATH="${CUDADIR}/x86_64-linux-gnu"
  else
        export CUDAPATH="${CUDADIR}/lib"
        export CUDA_SEARCH_PATH="/usr/lib/i386-linux-gnu"
fi

# Put ${CUDADIR}/bin at front of user path just in case we need a particular
# version of gcc.  Nvidia CUDA kit does not alway support default Ubuntu gcc.
# CSG dist a sym-link to the 'correct' version under ${CUDADIR}/bin.
export PATH=${CUDADIR}/bin:${PATH}

# Add location of relevant CUDA libraries to path for dynamic link loader.
if  [ -z ${LD_LIBRARY_PATH+x} ]
  then
        export LD_LIBRARY_PATH="$CUDAPATH:$CUDA_SEARCH_PATH"
  else
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDAPATH}:${CUDA_SEARCH_PATH}"
fi

unset CUDADIR NVIDIAVER
return 0