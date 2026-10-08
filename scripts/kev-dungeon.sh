#!/bin/sh
# Build and run the kev-dungeon example. MLX's Metal library is only bundled by xcodebuild (not `swift build`), so
# a plain `swift run` aborts with "Failed to load the default metallib".
#
#   scripts/kev-dungeon.sh [--release] [kev-dungeon arguments...]
set -e
cd "$(dirname "$0")/.."
config=Debug
if [ "$1" = "--release" ]; then config=Release; shift; fi
mkdir -p build
xcodebuild build -scheme kev-dungeon -configuration "$config" -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath build/DerivedData -skipPackagePluginValidation -skipMacroValidation -quiet \
  > build/xcodebuild-kev-dungeon.log 2>&1 || { cat build/xcodebuild-kev-dungeon.log; exit 1; }
exec "build/DerivedData/Build/Products/$config/kev-dungeon" "$@"
