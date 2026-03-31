#!/bin/bash
set -eu
set -o pipefail

deploy_path=$1
deploy_list=$2

mkdir -p "${deploy_path}"

rm -rf "${deploy_path}"/*
cat "$deploy_list" | xargs -n1 -I{} cp --parents -pr {} "${deploy_path}"

exit 0
