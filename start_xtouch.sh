#!/bin/bash
/Applications/Pd-0.49-1.app/Contents/Resources/bin/pd midi_controllers/xtouch-mini.pd &
python predictive_music_model.py -d=9 --modelsize=s --log -v -c