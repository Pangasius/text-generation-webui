"""The goal of this module is to pre-parse the
automatically generated questions in order to
have a better training set for the model."""

import regex as re
import json

import argparse


def question_cleaner(name: str):
    """
    Clean the document that resembles this :

    {
        "queries": {
            "96e2941f-d1e5-4603-a4da-caf3f82348ad": "not a question",
            "96e2941f-d1e5-4603-a4da-caf3f82348ad": "Actual question?"
            },
        "relevant_docs": {
            "96e2941f-d1e5-4603-a4da-caf3f82348ad": [
                "96e2941f-d1e5-4603-a4da-caf3f82348ad"
            ]
        }
    }

    into this :

        {
        "queries": {
            "96e2941f-d1e5-4603-a4da-caf3f82348ad": "Actual question?"
            },
        "relevant_docs": {
            "96e2941f-d1e5-4603-a4da-caf3f82348ad": [
                "96e2941f-d1e5-4603-a4da-caf3f82348ad"
            ]
        }
    }

    Args:
        path (str): _description_
    """
    # path = "models/embedder/name_of_the_model"
    with open("models/embedder/" + name + ".json") as f:
        doc = json.load(f)

    # inside are "queries" and "relevant_docs"
    queries = doc["queries"]
    relevant_docs = doc["relevant_docs"]

    # we want to keep only the questions that are actual questions
    # we will start at a [A-Z] and end at a ?
    keys_to_delete = []
    for key in queries:
        if re.search(r"[A-Z][^:]*\?", queries[key]) is None:
            keys_to_delete.append(key)
        else:
            queries[key] = re.findall(r"[A-Z][^:]*\?", queries[key])[0]

    # delete the keys that are not questions
    for key in keys_to_delete:
        del queries[key]
        del relevant_docs[key]

    # finally write the new json to the file with _cleaned appended to the name
    with open("models/embedder/" + name + "_cleaned.json", "w") as f:
        json.dump(doc, f, indent=4)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean the questions.json file')
    parser.add_argument('name', type=str, help='name of the model')
    args = parser.parse_args()

    question_cleaner(args.name)
