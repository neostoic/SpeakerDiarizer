#!/bin/env python
import sys, string

rttmlines = sys.stdin.readlines()
rttmlines = map(string.strip,rttmlines)
rttmlines = filter(lambda x: x.find("SPEAKER") == 0,rttmlines)
rttmlines = filter(lambda x: x.find("SPEECH") > 0,rttmlines)

for l in rttmlines:
    parts = l.split()
    et = float(parts[3]) + float(parts[4])
    print parts[1],parts[2],parts[3],"%0.3f"%et

sys.exit(0)
