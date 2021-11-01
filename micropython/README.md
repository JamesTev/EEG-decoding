## MicroPython setup
### Overview
The instructions below detail the steps necessary to get a working version of the **ESP32 port of MicroPython** on your local machine. This incldues the uLab extension for Micropython which is required for this project. 

For more generic information related to setting up MicroPython for other ports, check out (INSERT REPO LINK).

### Setup
Install script tries to install `virutalenv` with `--user` flag which causes issues if you are installing from some virtual environments. Install it first manually to prevent this issue with `pip install virtualenv` (provided you are within an activated virtual environment if applicable).

Pretty much all setup is contained within the script called `esp32-cmake.sh`. You may need to make it executable first using `chmod u+x esp32-cmake.sh`. Then, setup the build directory:

```bash
mkdir ~/mpy-esp32
export BUILD_DIR=~/mpy-esp32
```
You're free to perform the setup wherever. Just update `BUILD_DIR` accordingly. Now, we run the setup scrip using:

```bash
./esp32-cmake.sh
```
>  Note that if you're using macOS with Apple Silicon, you may need to run the above prefixed with arch -x86_64 to ensure compatibility with later steps.

Once complete, the following may be run for convenience:
```bash
# export mpy cross compiler to path to use `mpy-cross` command 
export PATH=$PATH:~/mpy-esp32/micropython/mpy-cross/
```
This will allow you to cross compile ordinary MicroPython scrips (`*.py` files) into binary `*.mpy` versions which are more efficient in terms of memory and speed. These can be built into the firmware by copying your MicroPython modules into `$BUILD_DIR/micropython/ports/esp32/modules`.

Finally, to deploy to your board, run `make erase && make deploy`.

### Troubleshooting
If you see complaints about not finding `idf.py`, it may be that the ESP-IDF has not been exported properly. To rectify this, navigate to `$BUILD_DIR/micropython/esp-idf`. Make sure the correct python environment has been activated and then run 
```bash
./install.sh
. ./export.sh
```
to install the IDF dependencies and export necessary variables to your shell environment.

```bash
BOARD = GENERIC
USER_C_MODULES = $(BUILD_DIR)/ulab/code/micropython.cmake
# Makefile for MicroPython on ESP32.
#
# This is a simple, convenience wrapper around idf.py (which uses cmake).
#...
```