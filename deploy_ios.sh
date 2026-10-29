#!/usr/bin/env bash
set -euo pipefail

# Configuration
SSH_HOST="${SSH_HOST:-root@ks-iphone.ts.krr.cl}"
SSH_PORT="${SSH_PORT:-36598}"
BUNDLE_ID="com.example.wgpu-triangle"
APP_NAME="WgpuTriangleApp"
IPA_PATH="ios/xtool/${APP_NAME}.ipa"

BUILD_CONFIG="release"
ACTION="all"
STREAM_LOGS=false

show_help() {
    cat <<EOF
Usage: $(basename "$0") [options]

Builds and deploys the wgpu triangle app to an iOS device via TrollStore over SSH.

Options:
  --release          Build in release mode (default, optimized)
  --debug            Build in debug mode
  --build-only       Only build the Rust lib and IPA, do not deploy
  --deploy-only      Skip build, only upload and install existing IPA
  --log, -l          Stream app logs (oslog) after launching
  -h, --help         Show this help message

Environment Variables:
  SSH_HOST           SSH target (default: root@ks-iphone.ts.krr.cl)
  SSH_PORT           SSH port (default: 36598)
EOF
}

# Parse CLI options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --release)
            BUILD_CONFIG="release"
            shift
            ;;
        --debug)
            BUILD_CONFIG="debug"
            shift
            ;;
        --build-only)
            ACTION="build"
            shift
            ;;
        --deploy-only)
            ACTION="deploy"
            shift
            ;;
        --log|-l)
            STREAM_LOGS=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'" >&2
            show_help
            exit 1
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- 1. BUILD PIPELINE ---
if [[ "$ACTION" != "deploy" ]]; then
    echo "============================================================"
    echo "==> 1/2. Building Rust static library ($BUILD_CONFIG, aarch64-apple-ios)..."
    echo "============================================================"
    CARGO_FLAGS=(--lib --target aarch64-apple-ios)
    if [[ "$BUILD_CONFIG" == "release" ]]; then
        CARGO_FLAGS+=(--release)
    fi
    cargo build "${CARGO_FLAGS[@]}"

    echo "============================================================"
    echo "==> 2/2. Building unsigned .ipa with xtool ($BUILD_CONFIG)..."
    echo "============================================================"
    (cd ios && xtool dev build -c "$BUILD_CONFIG" --ipa)

    if [[ ! -f "$IPA_PATH" ]]; then
        echo "Error: IPA was not generated at $IPA_PATH" >&2
        exit 1
    fi
    echo "==> Generated IPA: $IPA_PATH ($(du -h "$IPA_PATH" | cut -f1))"
fi

if [[ "$ACTION" == "build" ]]; then
    echo "Build complete. Skipping deployment (--build-only specified)."
    exit 0
fi

# --- 2. DEPLOYMENT PIPELINE ---
echo "============================================================"
echo "==> Connecting to device ($SSH_HOST:$SSH_PORT)..."
echo "============================================================"

# Terminate running instance so TrollStore can cleanly overwrite and relaunch
echo "==> Stopping any running instance of $APP_NAME..."
ssh -p "$SSH_PORT" -o ConnectTimeout=5 "$SSH_HOST" "killall -9 $APP_NAME 2>/dev/null || true"

echo "==> Uploading $IPA_PATH to device..."
scp -P "$SSH_PORT" "$IPA_PATH" "$SSH_HOST":/tmp/${APP_NAME}.ipa

echo "==> Installing via TrollStore helper & launching..."
ssh -p "$SSH_PORT" "$SSH_HOST" bash <<EOF
set -e
# Fast path check for trollstorehelper
TS_HELPER=""
if [ -x /var/jb/usr/bin/trollstorehelper ]; then
    TS_HELPER="/var/jb/usr/bin/trollstorehelper"
elif [ -x /Applications/TrollStore.app/trollstorehelper ]; then
    TS_HELPER="/Applications/TrollStore.app/trollstorehelper"
else
    TS_HELPER=\$(find /private/var/containers/Bundle/Application -name trollstorehelper 2>/dev/null | head -n 1)
fi

if [ -z "\$TS_HELPER" ]; then
    echo "Error: trollstorehelper not found on device!" >&2
    exit 1
fi

echo "Installing with: \$TS_HELPER"
"\$TS_HELPER" install /tmp/${APP_NAME}.ipa
rm -f /tmp/${APP_NAME}.ipa

echo "Launching \$BUNDLE_ID..."
if [ -x /var/jb/usr/bin/uiopen ]; then
    /var/jb/usr/bin/uiopen --bundleid "$BUNDLE_ID"
elif which uiopen >/dev/null 2>&1; then
    uiopen --bundleid "$BUNDLE_ID"
else
    echo "uiopen not found; please launch $APP_NAME manually from the home screen."
fi
EOF

echo "============================================================"
echo "==> Successfully deployed and launched $BUNDLE_ID!"
echo "============================================================"

if [[ "$STREAM_LOGS" == true ]]; then
    echo "==> Attaching to device log stream (Ctrl+C to stop)..."
    ssh -t -p "$SSH_PORT" "$SSH_HOST" "/var/jb/usr/bin/oslog -p $APP_NAME 2>/dev/null || log stream --predicate 'process == \"$APP_NAME\"'"
fi
