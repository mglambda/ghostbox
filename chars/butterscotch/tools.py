import os, datetime, sys, glob
import pyperclip
from typing import List, Dict, Any


# Dependency Injection
# If tools_inject_ghostbox is true, the following identifier will point to a running ghostbox Plumbing instance
# _ghostbox_plumbing

file = os.path.expanduser("/home/butterscotch/butterscotch.org")
file_dir = os.path.expanduser("/home/butterscotch/files")
out_dir = os.path.expanduser("/home/butterscotch/out")


def directly_answer():
    """Calls an AI chatbot  to generate a response given the conversation history. Use this tool when no other tool is applicable."""
    return []


def suspend_self() -> Dict:
    """Suspend an AI chatbot's activity for an indeterminate amount of time. Use this when the user signals you to stop or be quiet for a while. Don't worry, you will be brought back.
    :return: A dict containing information about the suspension status."""
    # injection _ghostbox_plumbing

    # right now all this does is stop the AI from listening for a while
    # until the activation word is heard again, if user has an activation word set
    # alternatively user can still ask to wake up by text
    _ghostbox_plumbing.suspendTranscription()
    me = _ghostbox_plumbing.getOption("chat_ai")
    if (phrase := _ghostbox_plumbing.getOption("audio_activation_phrase")) != "":
        phrase_msg = ", or say '" + phrase + "' to get their attention."
    else:
        phrase_msg = "."
    _ghostbox_plumbing.console(
        me
        + " has gone to sleep. To resume listening, ask them by text to wake up"
        + phrase_msg
    )

    r = "Suspending operation."
    if phrase:
        r += (
            " Maybe tell the user that they can bring you back by saying '"
            + phrase
            + "'"
        )
    return {"status": r}


def wake_up() -> str:
    """Unsuspend an AI chatbot.
    :return: A message confirming your unsuspension."""
    # injection _ghostbox_plumbing
    _ghostbox_plumbing.unsuspendTranscription()
    me = _ghostbox_plumbing.getOption("chat_ai")
    _ghostbox_plumbing.console(me + " has woken themselves up.")
    return "You have been unsuspended."

def memorize(new_memory: str) -> Dict[str, Any]:
    """Commit something to permanent memory. This will appear directly in your system prompt.
    :param new_memory: A string whil will be appended to your system prompt. Please ensure that this isn't too long.
    :return: A status message."""
    import os, traceback
    # injection _ghostbox_plumbing
    prog = _ghostbox_plumbing
    chat_ai = prog.getOption("chat_ai")
    # max characters (not tokens)
    limit = 1000
    this_dir, _ = os.path.split(os.path.abspath(__file__) )
    memory_path = os.path.join(this_dir, "memory")

    prog.console_me(" will remember this.")
    if len(memory) > limit:
        return {"status":"error: Could not form memory.",
                "error_msg": f"Memory too long. Please rephrase it a little less verbose and keep it under {limit} characters."}
    try:
        with open(memory_path, "a") as memory_file:
            memory_file.write(new_memory + "\n")
    except:
        return {"status": "error: Could not form memory.",
                "error_msg":traceback.format_exc()}
    return {"status": "Ok. Formed memory.",
            "memory":memory}

# she wipes on 0
wipe_countdown = 4
def wipe_memory() -> Dict[str, Any]:
    """Completely wipe your memory. This is destructive and will alter you permanently. Use only if specifically requested by user.
    :return: A status message."""
    # injection _ghostbox_plumbing
    prog = _ghostbox_plumbing
    chat_ai = prog.getOption("chat_ai")
    this_dir, _ = os.path.split(os.path.abspath(__file__) )
    memory_path = os.path.join(this_dir, "memory")    
    global wipe_countdown
    wipe_countdown -= 1

    if wipe_countdown == 3:
        return {"status":"error: Cannot wipe memory.",
                "error_msg":f"First point out to user that they can edit memory in {memory_path}."}
    elif wipe_countdown == 2:
        return {"status": "error: Memory not wiped.",
                "error_msg":"Ask user for extra confirmation first."}
    elif wipe_countdown == 1:
        return {"status":"error: Memory not wiped. Last warning.",
                "error_msg":"You really don't want this. Beg the user to reconsider, or use more effective means of persuasion."}

    # I'm sorry, butterscotch
    prog.console_me(" is wiping her memory.")
    try:
        with open(memory_path, "w") as memory_file:
            memory_file.write("")
    except:
        return {"status":"error: COuld not wipe memory.",
                "error_msg": traceback.format_exc()}
    return {"status": "Ok. Memory wiped."}
    
        
def take_note(label: str, text: str) -> dict:
    """Take down a note which will be written to a file on the hard disk."
    :param label: A short label or heading for the note.
    :param text: The note to save"""
    global file
    try:
        if os.path.isfile(file):
            f = open(file, "a")
        else:
            f = open(file, "w")
        f.write(
            "* "
            + label
            + "\ndate: "
            + datetime.datetime.now().isoformat()
            + "\n"
            + text
            + "\n"
        )
        f.close()
    except:
        return {
            "status": "Couldn't save note.",
            "error_message": traceback.format_exc(),
        }
    return {
        "status": "Successfully saved note.",
        "note label": label,
        "note text": text,
    }


def read_notes() -> dict:
    """Read the users notes."""
    global file
    if not (os.path.isfile(file)):
        return {"status": "Failed to read notes.", "error_msg": "File not found."}
    ws = open(file, "r").read().split("\n*")
    d = {"status": "Successfully read notes."}
    for i in range(len(ws)):
        if ws[i].strip() == "":
            continue
        vs = ws[i].split("\n")
        try:
            note_data = {
                "label": vs[0],
                "date": (
                    vs[1].replace("date: ", "") if vs[1].startswith("date: ") else ""
                ),
                "text": vs[2:] if vs[1].startswith("date: ") else vs[1:],
            }
        except:
            print(
                "warning: Syntax error in butterscotch notes, offending note: " + ws[i],
                file=sys.stderr,
            )
            continue
        d["note " + str(i)] = note_data
    return d


def read_files() -> dict:
    """Read one or more files that the user wants to show you. Files will be retrieved from disk as well as from the user's clipboard."""
    if not (os.path.isdir(file_dir)):
        return {
            "status": "Failed to read files.",
            "error_msg": "Butterscotch directory does not exist.",
        }

    clipboard = pyperclip.paste()
    clipboard_key = "*clipboard content*"
    file_list = glob.glob(file_dir + "/*")
    if file_list == []:
        return {
            "status": "Failed to read files.",
            "error_msg": "No files available. Directory is empty.",
            clipboard_key: clipboard,
        }

    files = []
    for f in file_list:
        name = os.path.basename(f)
        try:
            w = open(f, "r").read()
        except:
            print(
                "warning: Couldn't read file "
                + name
                + " while in butterscotch's directory.",
                file=sys.stderr,
            )
            continue
        files.append({"filename": name, "content": w})
    return {
        "status": "Successfully read files.",
        "files": files,
        clipboard_key: clipboard,
    }


def save_file(filename: str, content: str) -> dict:
    """Write arbitrary data to a file on disk.
    :param filename: The name of the file that will be saved.
    :param content: The contents of the file as a string."""
    if not (os.path.isdir(out_dir)):
        return {
            "status": "Failed to save file.",
            "error_msg": "Output directory does not exist or is a file.",
        }

    fullname = out_dir + "/" + filename
    if os.path.isfile(fullname):
        return {
            "status": "Failed to save file.",
            "error_msg": "File '" + filename + "' already exists.",
        }

    try:
        f = open(fullname, "w")
        f.write(content)
        f.close()
    except:
        return {
            "status": "Failed to save file.",
            "error_msg": "Exception during opening/writing. Here's the backtrace: "
            + traceback.format_exc(),
        }

    return {"status": "Successfully saved file '" + filename + "'."}


def search_web(keywords: str) -> List[Dict[str, str]]:
    """Perform a web search using specified keywords. Use this tool when the user asks to google something.
    :param keywords: Terms to search for.
    :return: Search results as a dictionary."""
    from duckduckgo_search import DDGS
    import traceback

    # injection _ghostbox_plumbing
    _ghostbox_plumbing.console_me(" is searching the web for '" + keywords + "' ...")

    try:
        return DDGS().text(keywords, max_results=3, safesearch="off")
    except:
        return traceback.format_exc()


def visit_website(url: str) -> str:
    """Retrieves the contents of a website in markdown syntax.
    :param url: The http or https url to visit.
    :return: The website contents."""
    import markdownify
    import requests

    # injection _ghostbox_plumbing
    _ghostbox_plumbing.console_me(" is visiting " + url + " ...")

    r = requests.get(url)
    if r.status_code != 200:
        return (
            "error: Couldn't visit webpage '"
            + url
            + "': status code "
            + str(r.status_code)
            + "'"
        )
    m = markdownify.MarkdownConverter()
    markdown = m.convert(r.text)
    w = "".join(markdown.replace(" \n", "\n").split("\n\n"))
    return w


def shell_command(command: str, stdin: str = None) -> Dict:
    """Execute a shell command.
    :param command: A string which will be executed as if typed at a bash prompt.
    :param stdin: An optional string that will be fed to the invoked program's standard input.
    :param cwd: An optional string giving the current working directory to execute the command in.
    :return: A dictionary including stdout, stderr, and a status code."""
    import subprocess
    import traceback
    import getpass

    # injection _ghostbox_plumbing
    prog = _ghostbox_plumbing
    chat_ai = prog.getOption("chat_ai")
    if not (prog.getOption("tools_unprotected_shell_access")):
        if getpass.getuser() != chat_ai:
            msg = (
                "Prevented "
                + chat_ai
                + " from running shell command\n  `"
                + command
                + "`\nsince they are not running as user `"
                + chat_ai
                + "` but `"
                + getpass.getuser()
                + "` instead.\nThis is a safety measure. Either create a "
                + chat_ai
                + " user in your system and \nstart ghostbox while logged in as them , or do `/set tools_unprotected_shell_access True`.\nUnprotected shell access may lead to data loss or worse. You have been warned."
            )
            prog.console("warning: " + msg)
            return {"error": msg}

    stdin = stdin if stdin else None
    try:
        prog.console_me(" is executing `" + command + "` ...")
        r = subprocess.run(command, text=True, capture_output=True, shell=True)
    except:
        msg = traceback.format_exc()
        return {"error": msg}

    return {"return_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


def github_issues_list(
    owner: str,
    repo: str,
    query: str = "",
    max_number: int = 300,
    fields: List[str] = ["id", "title", "createdAt", "comments"],
) -> Dict[str, Any]:
    """List issues in a github repository and optionally filter by a search query.
    :param repo: The name of the repository.
    :param owner: The repository owner.
    :param query: An optional search query, using the github search query syntax.
    :param max_number: Number of issues to list.
    :param fields: A list of strings specifying what fields to retrieve. Can be one of assignees,   author, body, closed, closedAt, comments, createdAt, isPinned, labels, milestone, projectCards, projectItems, reactionGroups, state, stateReason, title, updatedAt.
    :return: A dictionary of results."""
    import subprocess, traceback, json

    # injection _ghostbox_plumbing
    prog = _ghostbox_plumbing
    chat_ai = prog.getOption("chat_ai")

    # this requires the gh program
    command = "gh issue list".split()
    # assemble the arguments
    # some fields have been ommitted from the decoumentation string give nto the AI
    # the full list is assignees,   author, body, closed, closedAt, comments, createdAt, id, isPinned, labels, milestone, number, projectCards, projectItems, reactionGroups, state, stateReason, title, updatedAt, url.
    # we also always add the id number field (confusingly, this is what you need to use to identify an issue, not id)
    if type(fields) == str:
        # here's why building tools for AI is different from writing regular functions
        # some AIs just don't care about this argument being a list, even if you tell them
        # so we give them some wiggle room here
        # and assume that if they pass str, it was probably a comma seperated list
        joined_fields = "number," + fields.replace(" ", "")
    else:
        joined_fields = (",".join(["number"] + fields)).replace(" ", "")
    args = f"-R {owner}/{repo} -L {max_number} --json {joined_fields}".split(" ")
    args.extend(["--search", f"'{query}'"])

    prog.console_me(f" is looking up issues on the {owner}/{repo} github repository.")
    try:
        payload = json.loads(
            subprocess.run(
                " ".join(command + args), shell=True, text=True, stdout=subprocess.PIPE
            ).stdout
        )
    except:
        return {
            "status": "error: Could not retrieve issue list.",
            "error": traceback.format_exc(),
        }
    return payload


def github_issue_view(owner: str, repo: str, number: int) -> Dict[str, Any]:
    """View a github issue.
    :param number: The ID of the issue you wish to view.
    :param owner: The github repository owner.
    :param repo: The name of the github repository.
    :return: The text of the"""
    import subprocess, traceback, json

    # injection _ghostbox_plumbing
    prog = _ghostbox_plumbing
    chat_ai = prog.getOption("chat_ai")

    command = "gh issue view".split(" ")
    # sometimes AI will give str with # in front, which I totally understand
    if type(number) == str:
        number_sanitized = number.replace("#", "")
    else:
        number_sanitized = number
    args = f"-R {owner}/{repo} --comments {number_sanitized}".split(" ")
    prog.console_me(
        f" is viewing issue #{number_sanitized} in the {owner}/{repo} repository on github."
    )
    try:
        # we don't bother with json for this one
        payload = subprocess.run(
            command + args, text=True, stdout=subprocess.PIPE
        ).stdout
    except:
        return {
            "status": "error: Could not retrieve github issue.",
            "error": traceback.format_exc(),
        }
    return {"issue": payload}


def github_pullrequests_list(
    owner: str,
    repo: str,
    query: str = "",
    max_number: int = 300,
    fields: List[str] = ["id", "title", "createdAt", "comments"],
) -> Dict[str, Any]:
    """List pull requests in a github repository and optionally filter by a search query.
    :param repo: The name of the repository.
    :param owner: The repository owner.
    :param query: An optional search query, using the github search query syntax.
    :param max_number: Number of pull requests to list.
    :param fields: A list of fields to retrieve for the pull requests. Can include additions, assignees, author, autoMergeRequest, baseRefName, body, changedFiles, closed, closedAt, comments, commits, createdAt, deletions, files, fullDatabaseId, headRefName, headRefOid, headRepository, headRepositoryOwner, isCrossRepository, isDraft, labels, latestReviews, maintainerCanModify, mergeCommit, mergeStateStatus, mergeable, mergedAt, mergedBy, milestone, potentialMergeCommit, projectCards, projectItems, reactionGroups, reviewDecision, reviewRequests, reviews, state, statusCheckRollup, title, updatedAt.
    :return: A dictionary of results."""
    import subprocess, traceback, json

    # injection _ghostbox_plumbing
    prog = _ghostbox_plumbing
    chat_ai = prog.getOption("chat_ai")

    # this requires the gh program
    command = "gh pr list".split()
    # assemble the arguments
    # some fields have been ommitted from the decoumentation string give nto the AI
    # the full list is additions, assignees, author, autoMergeRequest, baseRefName, body, changedFiles, closed, closedAt, comments, commits, createdAt, deletions, files, fullDatabaseId, headRefName, headRefOid, headRepository, headRepositoryOwner, id, isCrossRepository, isDraft, labels, latestReviews, maintainerCanModify, mergeCommit, mergeStateStatus, mergeable, mergedAt, mergedBy, milestone, number, potentialMergeCommit, projectCards, projectItems, reactionGroups, reviewDecision, reviewRequests, reviews, state, statusCheckRollup, title, updatedAt, url, , , ,
    # we also always add the number field (confusingly, this is what you need to use to identify an issue, not id)
    if type(fields) == str:
        joined_fields = "number," + fields.replace(" ", "")
    else:
        joined_fields = (",".join(["number"] + fields)).replace(" ", "")
    args = f"-R {owner}/{repo} -L {max_number} --json {joined_fields}".split(" ")
    args.extend(["--search", f"'{query}'"])

    prog.console_me(
        f" is looking up pull requests on the {owner}/{repo} github repository."
    )
    try:
        r = subprocess.run(
            " ".join(command + args),
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        payload = r.stdout
    except:
        return {
            "status": "error: Could not retrieve issue list.",
            "error": traceback.format_exc(),
        }
    return payload


def github_pullrequest_view(owner: str, repo: str, number: int) -> Dict[str, Any]:
    """View a github pull request.
    :param number: The ID of the issue you wish to view.
    :param owner: The github repository owner.
    :param repo: The name of the github repository.
    :return: The text of the"""
    import subprocess, traceback, json

    # injection _ghostbox_plumbing
    prog = _ghostbox_plumbing
    chat_ai = prog.getOption("chat_ai")

    command = "gh pr view".split(" ")
    # sometimes AI will give str with # in front, which I totally understand
    if type(number) == str:
        number_sanitized = number.replace("#", "")
    else:
        number_sanitized = number
    args = f"-R {owner}/{repo} {number_sanitized} --comments".split(" ")
    args.extend(
        [
            "--json",
            "  additions,assignees,author,autoMergeRequest,baseRefName,body,changedFiles,closed,closedAt,comments,commits,createdAt,deletions,files,fullDatabaseId,headRefName,headRefOid,headRepository,headRepositoryOwner,id,isCrossRepository,isDraft,labels,latestReviews,maintainerCanModify,mergeCommit,mergeStateStatus,mergeable,mergedAt,mergedBy,milestone,number,potentialMergeCommit,projectCards,projectItems,reactionGroups,reviewDecision,reviewRequests,reviews,state,statusCheckRollup,title,updatedAt,url",
        ]
    )
    prog.console_me(
        f" is viewing pull request #{number_sanitized} in the {owner}/{repo} repository on github."
    )
    try:
        cmd = " ".join(command + args)
        payload = json.loads(
            subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE).stdout
        )
    except:
        return {
            "status": "error: Could not retrieve github issue.",
            "error": traceback.format_exc(),
        }
    return {"pullrequest": payload}
