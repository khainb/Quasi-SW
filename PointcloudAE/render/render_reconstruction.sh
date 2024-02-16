#!/bin/bash
FILE=reconstruct_random_50_shapenetcore55.npy

FOLDER=raw
COLOR=tab:gray
python3 render_mitsuba2_pc.py images/${FOLDER}/${FILE} ${COLOR}

FOLDER=swd
COLOR=tab:red
python3 render_mitsuba2_pc.py images/${FOLDER}/${FILE} ${COLOR}
