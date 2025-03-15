#!/usr/bin/env python
# This example shows how to use pydantic classes to get structured output from an LLM
# You could also use json schemas, but I recommend against it.
# Pydantic is quite wonderful.
from pydantic import BaseModel
from typing import *
import ghostbox, json

box = ghostbox.from_generic(character_folder="ghost-writer")


# this is the type for the object that we will let the LLM create
# how we name things here really matters
class BlogPost(BaseModel):
    title: str
    content: str
    tags: List[str]


post = box.new(
    BlogPost,  # this tells ghostbox and the backend what the structure should be
    "Write an extremely argumentative post about how an overabundance of busking is ruining berlin.",
)  # the prompt will provide context for filling in the python object
print(json.dumps(post.model_dump(), indent=4))
