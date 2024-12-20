# xArm Controller
A simple single-file controller for the XArm7 with minimal dependencies
## Installation
Clone and install the [xArm-Python-SDK](https://github.com/xArm-Developer/xArm-Python-SDK)
```
git clone https://github.com/xArm-Developer/xArm-Python-SDK.git
cd xArm-Python-SDK
python setup.py install
```
Install requirements
```
git clone https://github.com/MVerghese/xArm_Controller.git
cd xArm_Controller
pip install -r requirements.txt
```
## Usage
This library supports easy position, force, and hybrid position force control. For examples of these control modes, see the freespace_move, move_to_contact, and move_along_contact primitives.
