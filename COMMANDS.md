# Full list of commands for the ghostbox CLI

The below commands can be entered at the ghostbox command line interface prompt.

You can generate the list yourself with

```bash
ghostbox
/help commands
```

## !
Records using the microphone until you hit enter. The recording is then transcribed using openai's whisper, and inserted into the current line at the CLI.
THe precise model to be used for transcribing is determined using the 'whisper_model' option. Larger models transcribe more accurately, and may handle more languages, but also consume more resources. See https://huggingface.co/openai/whisper-large for a list of model names. The default is 'base.en'.
The model will be automatically downloaded the first time you transcribe with it. This may take a moment, but will only happen once for each model.
## /audio
Enable/disable audio input. This means the program will automatically record audio and transcribe it using the openai whisper model. A query is send to the backend with transcribed speech whenever a longish pause is detected in the input audio.
See whisper_model for the model used for transcribing.
## /clone 
    Create a new branch in the story folder, copying the contents entirely from the current story.
## /cont 
Continue generating without new input. Use this whenever you want the AI to 'just keep talking'.
Specifically, this will send the story up to this point verbatim to the LLM backend. You can check where you are with /log.
If there are templated tokens that are normally inserted as part of the prompt format when a user message is received, they are not inserted. Existing end tokens, like <|im_end|> will be removed before sending the prompt back.
This command is also executed when you hit enter without any text.
## /continue 
Continue generating without new input. Use this whenever you want the AI to 'just keep talking'.
Specifically, this will send the story up to this point verbatim to the LLM backend. You can check where you are with /log.
If there are templated tokens that are normally inserted as part of the prompt format when a user message is received, they are not inserted. Existing end tokens, like <|im_end|> will be removed before sending the prompt back.
This command is also executed when you hit enter without any text.
## /detokenize [-n]
Turn a list of tokens into strings.
Tokens can be supplied like this /detokenize 23 20001 1
    If -n is supplied, reads one token per line until an empty line is found.
## /drop 
Drops the last entry from the current story branch, regardless of wether it was user provided or generated by the LLM backend.
## /help [TOPIC] [-v|--verbose]
    List help on various topics. Use -v to see even more information.
## /helpcommands 
    List commands, their arguments, and a short description.
## /hide 
    Hide various program outputs and change some variables for a less verbose display.
    This does nothing that you cannot achieve by manually setting several options, it just bundles an eclectic mix of them in one command.
    I like to use this with TTS for a more immersive experience.
## /image [--id=IMAGE_ID] IMAGE_PATH
Add images for multimodal models that can handle them. You can refer to images by their id in the form of `[img-ID]`. If --id is omitted, id= 1 is assumed. Examples:
```
    /image ~/Pictures.test.png
Please describe [img-1].
    ```

    Alternatively, with multiple images:
    ```
    /image --id=1 ~/Pictures.paris.png
    /image --id=2 ~/Pictures/berlin.png
Can you compare [img-1] and [img-2]?
    ```
## /image_watch [DIR]
Enable / disable automatic watching for images in a specified folder.
    When new images are created / modified in image_watch_dir, a message (image_watch_msg)is automatically sent to the backend.
    If DIR is provided to this command, image_watch_dir will be set to DIR. Otherwise, the preexisting image_watch_dir is used, which defaults to the user's standard screenshot folder.
    This allows you to e.g. take screenshots and have the TTS automatically describe them without having to switch back to this program.
    Check status with /status image_watch.
## /lastrequest 
    Dumps a bunch of information about the last request send. Note that this won't do anything if you haven't sent a request to a working backend server that answered you.
## /lastresult 
    Dumps a bunch of information about the last result received. Note that this won't do anything if you haven't sent a request to a working backend server that answered you.
## /load STORY_FOLDER_NAME
    Loads a previously saved story folder from file STORY_FOLDER_NAME. See the /save command on how to save story folders.
    A story folder is a json file containing the entire chat history.
## /loadconfig CONFIG_FILE
Loads a json config file at location CONFIG_FILE. A config file contains a dictionary of program options. 
You can create an example config with /saveconfig example.conf.json.
You can also load a config file at startup with the --config_file command line argument.
If it exists, the ~/.ghostbox.conf.json will also be loaded at startup.
The order of config file loading is as follows .ghostconf.conf.json > --config_file > conf.json (from character folder). Config files that are loaded later will override settings from earlier files.
## /log 
    Prints the raw log of the current story branch.
    This includes prompt-format tokens and other stuff that is normally filtered out. For a prettier display, see /print.
    Also, /log prints to stderr, while /print will output to stdout.
## /lschars 
    Lists all available character folders. These can be used with the /start command or the -c command line argument, and are valid values for the character_folder parameter in the Ghostbox API.
    To see what places are searched for characters, see the value of the 'include' option.
## /lsoptions [OPTION_NAME]
Displays the list of program options, along with their values. Provide OPTION_NAME to see just its value.
The options displayed can all be set using /set OPTION_NAME.
Almost all of them may also be provided as command line arguments with preceding dashes, e.g. include as --include=/some/path.
Finally, they may be set in various config files, such as in character folders, or in a config file loaded with /load.
## /lstemplates 
Lists all available templates for prompt formats.
To see the places searched, check the value of the template_include option.
## /lsvars [VARIABLE_NAME]
    Shows all session variables and their respective values. Show a single variable if VARIABLE_NAME is provided.
    Variables are automatically expanded within text that is provided by the user or generated by the LLM backend.
    E.g. if variable favorite_fruit is set to 'Tomatoe', any occurrence of {{favorite_fruit}} in the text will be replaced by 'Tomatoe'.
    There are three ways to set variables:
      1. Loading a character folder (/start). All files in a character folder automatically become variables, with the respective filename becoming the name of the variable, and the file's content becoming the value of the variable.
      2. The -x or --varfile command line option. This argument provides additional files that may serve as variables, similar to (1). The argument may be repeated multiple times.
      3. API only: Using the set_vars method on a Ghostbox object.

    You can also do {[FILENAME]} to ad-hoc splice the contents of FILENAME into a prompt. However, due to security reasons, this only works at the CLI.
    
## /lsvoices 
List all available voices for the TTS program. These can be used with /set tts_voice.
To see the places searched for voices, see the value of tts_voice_dir.
Depending on the value of tts_program, that location may be meaningless, as the voice won't be file based, like with amazon polly voices. In this case /lsvoices tries to give a helpful answer if it can.
## /mode [MODE_NAME}
Put the program into the specified mode, or show the current mode if MODE_NAME is omitted.
Possible values are 'default', or 'chat'.
Setting a certain mode has various effects on many aspects of program execution. Currently, most of this is undocumented :)
## /new 
    Create a new, empty branch in the story folder. You can always go back with /prev or /story.
## /next 
    Go to next branch in story folder.
## /prev 
    Go to previous branch in story folder.
## /print [FILENAME]
Print the current story.
If FILENAME is provided, save the story to that file.
## /quit 
    Quit the program. Chat history (story folder) is discarded. All options are lost.
    See also /save, /saveoptions, /saveconfig
## /raw 
    Displays the raw output for the last prompt that was sent to the backend.
## /rephrase 
This will rewind the story to just before your last input, allowing you to rephrase your query.
    Note that /rephrase is not destructive. It will always create a new story branch before rewinding.
    See also: /retry
## /restart 
    Restarts the current character folder. Note that this will wipe the current story folder, i.e. your chat history, so you may want to /save.
    /restart is equivalent to /start CURRENT_CHAR_FOLDER.
## /retry 
    Retry generation of the LLM's response.
This will drop the last generated response from the current story and generate it again. Use this in most cases where you want to regenerate. If you extended the LLM's repsone (with /cont or hitting enter), the entire repsonse will be regenerated, not just the last part.
Note that /retry is not destructive. It always creates a new story branch before regenerating a repsonse.
    See also: /rephrase.
## /save [STORY_FOLDER_NAME]
    Save the entire story folder in the file STORY_FOLDER_NAME. If no argument is provided, creates a file with the current timestampe as name.
    The file created is your accumulated chat history in its entirety, including the current story and all other ones (accessible with /prev and /next, etc). It is saved in the json format.
    A saved story folder may be loaded with /load.
## /saveconfig CONFIG_FILE
    Save the current program options and their values to the file at location CONFIG_FILE. This will either create or overwrite the CONFIG_FILE, deleting all its previous contents.
## /saveoptions CONFIG_FILE
    Save the current program options and their values to the file at location CONFIG_FILE. This will either create or overwrite the CONFIG_FILE, deleting all its previous contents.
## /set OPTION_NAME [OPTION_VALUE]
    Set options during program execution. To see a list of all possible values for OPTION_NAME, do /lsoptions. OPTION_VALUE must be a valid python expression, so if you want to set e.g. chat_user to Bob, do /set chat_user "Bob".
    When OPTION_VALUE is omitted, the value is set to "". This is equivalent to /unset OPTION_NAME.
## /start CHARACTER_FOLDER
Start a new session with the character or template defined in CHARACTER_FOLDER. You can specify a full path, or just the folder name, in which case the program will look for it in all folders specified in the 'include' paths. See /lsoptions include.
## /status 
Give an overall report about the program and some subprocesses.
## /story [BRANCH_NUMBER]
    Moves to a different branch in the story folder. If BRANCH_NUMBER is omitted, shows the current branch instead.
## /switch  CHARACTER_FOLDER
Switch to another character folder, retaining the current story.
As opposed to /start or /restart, this will hot-switch the AI without wiping your story so far, or adding the initial message. This can e.g. allow the used of specialized AI's in certain situations.
Specifically, '/switch bob' will do the following:
 - Set your system prompt to bob/system_msg
 - Set all session variables defined in bob/, possibly overriding existing ones
 - Load bob/config.json, if present, not overriding command line arguments.
See also: /start, /restart, /lschars
## /template [PROMPT_TEMPLATE]
Shows the current prompt template. If PROMPT_TEMPLATE is supplied, tries to load that template.
LLMs usually work best when supplied with input that is formatted similar to their training data. Prompt templates apply some changes to your inputs in the background to get them into a shape that the LLM expects.
    To find out the right template to use, consult the model card of the LLM you are using. When in doubt, 'chat-ml' is a very common prompt format.
To disable this, set the template to 'raw' and you won't have anything done to your inputs. This can be useful for experimentation.
Templates are searched for in the directories specified by the template_include option, which can be supplied at the command line.
To get a full list of available templates, try /lstemplates .
## /test 
Send a random test question to the backend.
The question is pulled from a random set of example questions. These are sometimes funny, but also turn out to be quite useful to get a general sense of a model.
## /time 
Show some performance stats for the last request.
## /tokenize [-c] MSG
Send a tokenize request to the server. Will print raw tokens to standard output, one per line. This is mostly used to debug prompts. With -c, it will print the number of tokens instead.
## /transcribe Records using the microphone until you hit enter. The recording is then transcribed using openai's whisper, and inserted into the current line at the CLI.
THe precise model to be used for transcribing is determined using the 'whisper_model' option. Larger models transcribe more accurately, and may handle more languages, but also consume more resources. See https://huggingface.co/openai/whisper-large for a list of model names. The default is 'base.en'.
The model will be automatically downloaded the first time you transcribe with it. This may take a moment, but will only happen once for each model.
## /tts 
This turns the TTS (text-to-speech) module on or off. When TTS is on, text that is generated by the LLM backend will be spoken out loud.
The TTS service that will be used depends on the value of tts_program. The tts_program can be any executable or shell script that reads lines from standard input. It may also support additional functionality.
An example tts program for amazon polly voices is provided with 'ghostbox-tts-polly'. Note that this requires you to have credentials with amazon web services.
On linux distributions with speech-dispatcher, you can set the value of tts_program to 'spd-say', or 'espeak-ng' if it's installed. This works, but it isn't very nice.
The voice used depends on the value of tts_voice, which you can either /set or provide with the -V or --tts_voice command line option. It can also be set in character folder's config.json.
ghostbox will attempt to provide the tts_program with the voice using a -V command line argument.
To see the list of supported voices, try /lsvoices.
Enabling TTS will automatically set stream_flush to 'sentence', as this works best with most TTS engines. You can manually reset it to 'token' if you want, though.
## /ttsdebug 
Get stdout and stderr from the underlying tts_program when TTS is enabled.
## /unset 
## /varfile 

