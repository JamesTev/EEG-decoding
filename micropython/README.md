Install script tries to install `virutalenv` with `--user` flag which causes issues if you are installing from some virtual environments. Install it first manually to prevent this issue with `pip install virtualenv` (provided you are within an activated virtual environment if applicable).

Make executable:
`chmod u+x esp32-cmake.sh`

```bash
mkdir ~/mpy-esp32
export BUILD_DIR=~/mpy-esp32

arch -x86_64 ./esp32-cmake.sh

# export mpy cross compiler to path to use `mpy-cross` command 
export PATH=$PATH:~/mpy-esp32/micropython/mpy-cross/
```
Finally, to deploy to your board, run `make erase && make deploy`

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