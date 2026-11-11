#!/bin/sh
# Downloads DiskANN at the commit the paper used, applies diskann.patch and checks the patched files.
#   sh third_party/fetch_diskann.sh [directory]        (default: third_party/DiskANN)
set -eu

here=$(cd "$(dirname "$0")" && pwd)
target=${1:-$here/DiskANN}
commit=a26f824be5a597b49a7bf33f51ee75121e7c03af

if [ ! -d "$target" ]; then
    git clone --quiet https://github.com/microsoft/DiskANN.git "$target"
    git -C "$target" checkout --quiet "$commit"
    git -C "$target" apply "$here/diskann.patch"
fi

cd "$target"
if ! md5sum --check --quiet <<EOF
10f80c50685548fb0fb5e9161e1d90e8  include/index.h
c9a68202082800b0617e895676d2f60c  include/in_mem_data_store.h
8b589c01c068ddc857c650e08eb2f140  src/index.cpp
EOF
then
    echo "$target is not DiskANN $commit with diskann.patch: delete it and run this script again" >&2
    exit 1
fi
echo "DiskANN $commit with diskann.patch: $target"
