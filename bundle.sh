#!/bin/sh
# Builds "Movie Shrinker.app" into ./dist (drag it to /Applications).
# Optional: BUNDLE_FFMPEG=1 ./bundle.sh copies ffmpeg/ffprobe into the app.
set -e
cd "$(dirname "$0")"
cargo build --release
APP="dist/Movie Shrinker.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp target/release/movie-shrinker "$APP/Contents/MacOS/"
if [ -n "$BUNDLE_FFMPEG" ]; then
  cp "$(command -v ffmpeg)" "$(command -v ffprobe)" "$APP/Contents/MacOS/"
fi
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Movie Shrinker</string>
  <key>CFBundleDisplayName</key><string>Movie Shrinker</string>
  <key>CFBundleIdentifier</key><string>local.movie-shrinker</string>
  <key>CFBundleExecutable</key><string>movie-shrinker</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
codesign --force -s - "$APP" >/dev/null 2>&1 || true
echo "Built: $APP"
