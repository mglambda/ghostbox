#!/usr/bin/env python
# This is an example of a very basic interaction loop
# it probably won't feel good, since it blocks and doesn't use streaming
# but we want to keep it simple
import ghostbox, time, random

# the generic adapter will work with anything that supports the OAI API
box = ghostbox.from_generic(
    character_folder="game_master",  # see below
    stderr=False,  # since this is a CLI program, we don't want clutter
    quiet=True,  # we do printing and tts ourselves
    tts=True,  # this means responses will be spoken automatically
    tts_model="kokoro",  # kokoro is nice because it's small and good
    tts_voice="bm_daniel",  # daniel is real GM material
)

if name := input("What is your cool adventurer name?\nName: "):
    print(f"Welcome, {name}! A game master will be with you shortly...")
else:
    name = "Drizzt Do'Urden"
    print("Better sharpen your scimitars...")

# this will make {{chat_user}} expand to whatever the user just typed
box.set_vars({"chat_user": name})

print(
    box.text(
        "Come up with an adventure scenario and give an introduction to the player."
    )
)

# we start conservative, but the adventure will get wilder as we go on
current_temperature, escalation_factor = 0.3, 0.05
while True:
    user_msg = input("Your response (q to quit): ")
    box.tts_stop()  # users usually like it when the tts shuts up after they hit enter

    if user_msg == "q":
        print(
            box.text(
                "{{chat_user}} will quit the game now. Please conclude the adventure and write a proper goodbye."
            )
        )
        break

    with box.options(
        temperature=current_temperature,  # this changes every loop iteration
        max_length=100
        + 10
        * random.randint(
            -3, 3
        ),  # keep it from talking for too long, but give some variety
    ):
        print(box.text(user_msg))

    current_temperature = min(current_temperature + escalation_factor, 1.3)

time.sleep(10)  # give time to finish the speech
