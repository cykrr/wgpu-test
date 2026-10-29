#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-root@ks-iphone.ts.krr.cl}"
SSH_PORT="${SSH_PORT:-36598}"
BUNDLE_ID="com.example.wgpu-triangle"

echo "==> 1. Building Rust static library for iOS (aarch64-apple-ios)..."
cargo build --lib --release --target aarch64-apple-ios

echo "==> 2. Building unsigned .ipa with xtool..."
(cd ios && xtool dev build -c release --ipa)

echo "==> 3. Uploading IPA to device ($SSH_HOST:$SSH_PORT)..."
scp -P "$SSH_PORT" ios/xtool/WgpuTriangleApp.ipa "$SSH_HOST":/tmp/WgpuTriangleApp.ipa

echo "==> 4. Installing IPA via TrollStore and launching app..."
ssh -p "$SSH_PORT" "$SSH_HOST" bash <<EOF
set -e
TS_HELPER=\$(find /private/var/containers/Bundle/Application -name trollstorehelper 2>/dev/null | head -n 1)
if [ -z "\$TS_HELPER" ]; then
    echo "trollstorehelper not found!"
    exit 1
fi

echo "Installing with \$TS_HELPER..."
"\$TS_HELPER" install /tmp/WgpuTriangleApp.ipa
rm -f /tmp/WgpuTriangleApp.ipa

echo "Launching \$BUNDLE_ID..."
if [ -x /var/jb/usr/bin/uiopen ]; then
    /var/jb/usr/bin/uiopen --bundleid "$BUNDLE_ID"
elif which uiopen >/dev/null 2>&1; then
    uiopen --bundleid "$BUNDLE_ID"
fi
EOF

echo "==> Successfully deployed and launched $BUNDLE_ID on device!"
