"""360-degree orbit video support for the Defirmation deformation pipeline.

The orbit path is deliberately separate from the still-image path. A phone
orbit video is large, high frame rate and mostly redundant, so frames are
decoded once, downscaled and sampled on a fixed time grid before any model
runs. Everything downstream then operates on ordinary BGR arrays, exactly like
the still-image pipeline.
"""
