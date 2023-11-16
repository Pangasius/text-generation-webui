"""Reads all the files under examples/f_embed_raw
If they are under jira_f, rename json to jjson
if they are under confluence_f, rename json to cjson"""

from glob import glob

import os

from tqdm import tqdm

DEBUG = True


def rename_file(file, extension):
    file = file.replace(".json", extension)
    return file


def rename_files():
    counters = {"jjson": 0, "cjson": 0, "skipped": 0, "skipped_json": 0}
    for file in tqdm(glob("examples/f_embed_raw/**/*", recursive=True)):
        if not file.endswith(".json"):
            counters["skipped"] += 1
            continue

        if file.__contains__("/jira_f/"):
            print("Renaming file", file, "to", rename_file(file, ".jjson")) if DEBUG else os.rename(file, rename_file(file, ".jjson"))
            counters["jjson"] += 1
        elif file.__contains__("/conf_f_embed/"):
            print("Renaming file", file, "to", rename_file(file, ".cjson")) if DEBUG else os.rename(file, rename_file(file, ".cjson"))
            counters["cjson"] += 1
        else:
            counters["skipped_json"] += 1
            continue

    print(counters)


if __name__ == "__main__":
    rename_files()
