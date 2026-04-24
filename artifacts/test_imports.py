import sys, traceback

def log(msg):
    print(msg, flush=True)
    with open("debug.log", "a") as f:
        f.write(msg + "\n")

open("debug.log", "w").close()
log("start")
import numpy as np
log("numpy ok")
import pandas as pd
log("pandas ok")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
log("sklearn ok")
log("about to import keras...")
import keras
log(f"keras ok: {keras.__version__}")
df = pd.read_csv("asl_landmarks.csv", header=None)
log(f"csv ok: {df.shape}")
