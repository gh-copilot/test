#!/bin/bash

SCRIPT_DIR=$(dirname -- "$0")

rm .SRCINFO

sed -i "s/(free edition)/(minimal free edition)/g" PKGBUILD
sed -i 's/noextract=("${pkgname}-${pkgver}.jar")/noextract=("${pkgname}-${pkgver}.sh")/g' PKGBUILD
sed -i "s/&type=Jar/&type=Linux/g" PKGBUILD
sed -i 's/source=("${pkgname}-${pkgver}.jar::/source=("${pkgname}-${pkgver}.sh::/g' PKGBUILD
sed -i -E "s/sha256sums=\(.*/sha256sums=\('SKIP'/g" PKGBUILD

git apply $SCRIPT_DIR/burpsuite.patch

cat PKGBUILD

