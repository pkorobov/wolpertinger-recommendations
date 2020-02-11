import json
import os
import logging

SEED = 42

with(open(os.path.join(os.environ["config_path"], "config.json"))) as config_file:
    logging.info("loading config from {}".format(os.environ["config_path"]))
    config = json.load(config_file)
