import os

import configparser
config = configparser.ConfigParser()
config.read("config.cfg")

ROOT = config["SAVE"]["ROOT"]

ROOT = ROOT + "/result"

if not os.path.exists(ROOT):
    os.makedirs(ROOT)

dirs = ["output", "figures", "features"]

for d in dirs:
    p = os.path.join(ROOT, d)
    if not os.path.exists(p):
        os.makedirs(p)
        