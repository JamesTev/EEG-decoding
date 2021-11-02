## MicroPython setup
### Overview
The instructions below detail the steps necessary to get a working version of the [ESP32 port of MicroPython](https://docs.micropython.org/en/latest/esp32/quickref.html) on your local machine. This incldues the [`ulab`](https://github.com/v923z/micropython-ulab) extension for Micropython which is required for this project. It is offers very convenient and efficient `numpy`-like array manipulation.

For more generic information related to setting up MicroPython for other ports, check out the [repo](https://github.com/micropython/micropython).

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

#### Deployment
Set your serial port. It will look something like this (but probably not the same): `export PORT=/dev/tty.usbserial-02U1W54L`. At least for Unix-based systems, you can list available serial ports using `ls /dev/tty.*`.

Finally, to deploy to your board, run `make erase && make deploy`.

### Aditional notes
#### Precision considerations
An important consideration is the precision needed for your application. If your application requires numerous matrix multiplications or other operations where precision errors are at risk of propagating to an unacceptable degree, you may want to enable __double precision__ in the firmware build. This can be done by updating `MICROPY_FLOAT_IMPL` in `$BUILD_DIR/micropython/ports/esp32/mpconfigport.h` as follows:
```c
// #define MICROPY_FLOAT_IMPL                  (MICROPY_FLOAT_IMPL_FLOAT)
#define MICROPY_FLOAT_IMPL                  (MICROPY_FLOAT_IMPL_DOUBLE)
```
__NB__: In order for this to take effect, you'll have to rebuild the firmware image as detailed below. 
#### Rebuilding firmware
A new firmware binary can be compiled by running the following within `$BUILD_DIR/micropython/ports/esp32/`:
```bash
make clean && make all
```
Then, follow the deployment steps as mentioned above to flash the new image onto your target board. Note that this requires the ESP-IDF to be exported in your shell environment. If you get an error concerning this, you'll likely need to __export the IDF variables again__. See the section on the ESP-IDF in the troubleshooting section to rectify this.
### Uploading code

Install the [`ampy`](https://learn.adafruit.com/micropython-basics-load-files-and-run-code/install-ampy) Python package in your same virtual environment. This allows you to upload, read, manipulate and delete files in non-volatile storage (flash) on the ESP32 over serial. 

Test the installation with `ampy -p /dev/tty.usbserial-02U1W54L ls` to list the files in NVS on the board (remember to replace your port accordingly). You should see something like `/boot.py`. You can see a list of other useful commands by typing `ampy --help`.  

### Development
An extremely useful feature of the MicroPython development platform is that it is compatible with Jupyter notebooks over serial. This has been made possible by projects like [this one](https://github.com/goatchurchprime/jupyter_micropython_kernel/) which contains all the details in getting setup. 

In summary, from within your virtual env, run:

```bash
pip install jupyterlab 
pip install jupyter_micropython_kernel
python -m jupyter_micropython_kernel.install
```
You can run `jupyter kernelspec list` to see where your Jupyter kernels are installed. You should see the `micropython` kernel listed there. To start a notebook, run
```bash
jupyter lab 
```
You can also run `jupyter notebook` for a more lightweight version. This will start the Jupyter server and should open a window in your default browser. Then, make sure the selected kernel is the `micropython` kernel you created earlier. In order to connect with the ESP32 over serial, run
```ipython
%serialconnect to --port=/dev/tty.usbserial-02U1W54L --baud=115200
```
replacing your specific serial port as required. After a few moments, should see a response like 
```text
Connecting to --port=/dev/tty.usbserial-02U1W54L --baud=115200
Ready.
```
Then, you're free to use your MicroPython board in an interactive notebook environment! You can test it with pretty much any standard python commands, including very convenient structures such as dictionary and list comprehensions. In order to test MicroPython-specific functionality, try running something like 
```python
from machine import Pin
import time

LED_PIN = 5 # replace as necessary

led = Pin(LED_PIN, Pin.OUT) # define pin 0 as output

led.value(1) # set LED on
time.sleep(2) # wait 2 seconds
led.value(0) # set LED off
```
Test that `ulab` was built into the compiled firmware image correctly by importing it: 
```python
import ulab
```
If no errors are shown, it worked! Here are some example of basic linear algebra functionality offered by `ulab` that give an idea of just how convenient and useful it is
```python
from ulab import numpy as np

# create an arbitrary positive definite, symmetric 3x3 matrix 
# A can be sliced like A[i0:i1, j0:j1] as with regular numpy
A = np.array([[25, 15, -5], [15, 18, 0], [-5, 0, 11]])

A_sqrt = np.linalg.cholesky(A) # compute lower triangular square root or A using Cholesky decomp

det_A = np.linalg.det(A) # compute determinant of A

A_inv = np.linalg.inv(A) # compute determinant of A

# compute Moore-Penrose pseudoinverse of A = (A^T.A)^-1.A^T
A_pinv = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A)) 

# many other utility functions such as argmax(), argsort(), convolve()
```


### Troubleshooting
If you see complaints about not finding `idf.py`, it may be that the ESP-IDF has not been exported properly. To rectify this, navigate to `$BUILD_DIR/micropython/esp-idf`. Make sure the correct python environment has been activated and then run 
```bash
./install.sh
. ./export.sh
```
to install the IDF dependencies and export necessary variables to your shell environment. 

Your `Makefile` under `$BUILD_DIR/micropython/ports/esp32/` should begin with something like this:
```bash
BOARD = GENERIC
USER_C_MODULES = $(BUILD_DIR)/ulab/code/micropython.cmake
# Makefile for MicroPython on ESP32.
#
# This is a simple, convenience wrapper around idf.py (which uses cmake).
#...
```
