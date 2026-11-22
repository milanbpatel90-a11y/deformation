"""Perceptual-hash duplicate filtering without optional third-party dependencies."""
import cv2, numpy as np

def perceptual_hash(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); small=cv2.resize(gray,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    dct=cv2.dct(small); block=dct[:8,:8]; vals=block.flatten()[1:]; median=float(np.median(vals))
    bits=vals>median; value=0
    for bit in bits: value=(value<<1)|int(bit)
    return value

def hash_distance(a,b): return int((int(a)^int(b)).bit_count())
