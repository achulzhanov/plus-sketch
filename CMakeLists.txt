cmake_minimum_required(VERSION 3.10)
project(plussketch C)

# -Os plus section GC: the linker drops everything unreferenced, which
# matters a lot when newlib is in the picture.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Os -ffunction-sections -fdata-sections")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

add_application(PlusSketch
    platform/mac68k.c
    core/psk.c
    core/forward.c
    core/sketch.c
)
target_include_directories(PlusSketch PRIVATE core)

# No stdio on the Mac: the File Manager reads the model, and the console is
# QuickDraw. Keeps newlib's printf (and its float support) out of the link.
target_compile_definitions(PlusSketch PRIVATE PSK_NO_STDIO)