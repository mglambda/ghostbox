#!/usr/bin/env python
# This example demonstrates how to load and interact with images.
# When you try it, make sure you have a backend and a language model that supports multimodality.
# This is the case with OpenAI.
# It is (as of this writing in march 2025) *not* the case with llama.cpp.
# However it is tested with llama-box, which also supports the generic OAI api.
import argparse, os, sys
import ghostbox


p = argparse.ArgumentParser(description="Compare two images and describe which one is more asthetically pleasing.")
p.add_argument("image1", type=str, help="Path to the first image to compare.")
p.add_argument("image2", type=str, help="Path to the second image to compare.")
args = p.parse_args()

# get a ghostbox
box = ghostbox.from_generic(character_folder="art_critic",
                            stderr=False,
                            quiet=True)

# let's make sure the images exist
# ghostbox will not raise an error if they don't
if not(os.path.isfile(args.image1)):
    print("error: First image could not be found.")
    sys.exit()

if not(os.path.isfile(args.image2)):
    print("error: Second image could not be found.")
    sys.exit()

# we need both images in context before asking the art critic
with box.images([args.image1, args.image2]):
    criticism = box.text("Which of these images is more aesthetically pleasing, and why?")

print(criticism)
