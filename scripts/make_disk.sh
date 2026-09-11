#!/bin/sh
# Build an 800K floppy with everything needed to run PlusSketch.
# Mount it in Mini vMac alongside a System 6 boot disk.
# Note: this is not bootable and has no SCSI driver partition, so it is for
# emulators and floppy transfer, not BlueSCSI. See README.
set -e

OUT=dist/PlusSketch.dsk
mkdir -p dist
rm -f "$OUT"

dd if=/dev/zero of="$OUT" bs=1024 count=800 2>/dev/null
hformat -l "PlusSketch" "$OUT"
hmount "$OUT"
hcopy -m build-68k/PlusSketch.bin   :PlusSketch
hcopy -r models/plus_sketch_q4.psk  :plus_sketch_q4.psk
hcopy -t data/categories.txt        :categories.txt
hdir
humount

echo "wrote $OUT"
