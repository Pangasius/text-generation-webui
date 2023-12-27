import json
from pathlib import PosixPath
from llama_index.readers.base import BaseReader
from llama_index import Document
from typing import Any, List

from bs4 import BeautifulSoup

class ConfluenceReader(BaseReader):
    """This is a reader to read the structured from custom json files"""

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """
        Load data from a Json file.
        The JSON should at least contain "content" and "metadata" keys.
        The "content" key should contain the text of the document.
        The "metadata" key should contain a dictionary of metadata.
        If attachments are present in the metadata they will be thrown away.

        Args:
            file (Path): Path to the file to load

        Raises:
            TypeError: If the file argument is not a string

        Returns:
            List[Document]: The list of LlamaIndex Documents containing
            the data or an empty list if the file is not valid
        """

        # Extract the file path from the args
        file = args[0]

        with open(file, "r", encoding="utf_8") as f:
            text_dict = f.read()

        j = json.loads(text_dict)

        content = j["content"]
        metadata = j["metadata"]

        # Remove attachments for now
        if "attachments" in metadata:
            metadata.pop("attachments")

        # Remove html tags because they take a lot of space
        soup = BeautifulSoup(content,features="lxml")
        content = soup.get_text("\n")

        doc = Document(text=content, metadata=metadata)

        return [doc]


class JiraReader(BaseReader):
    """This is a reader to read the structured from custom json files"""

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """
        Load data from a Json file.
        The JSON should at least contain "content" and "metadata" keys.
        The "content" key should contain the text of the document.
        The "metadata" key should contain a dictionary of metadata.
        If attachments are present in the metadata they will be thrown away.

        Args:
            file (Path): Path to the file to load

        Raises:
            TypeError: If the file argument is not a string

        Returns:
            List[Document]: The list of LlamaIndex Documents containing
            the data or an empty list if the file is not valid
        """

        # Extract the file path from the args
        file = PosixPath(args[0])

        if file.match("*project.jjson"):
            return []

        with open(file, "r", encoding="utf_8") as f:
            text_dict = f.read()

        j = json.loads(text_dict)

        # add to metadata some fields like
        # id, key, project, date, type, summary, resolution, status
        metadata = {}

        for key in ("project", "id", "key", "date", "type", "resolution", "status", "versions"):
            if key == "resolution" and "resolution" in j and j["resolution"] and "name" in j["resolution"]:
                metadata["resolution"] = j["resolution"]["name"]
            elif key == "resolution":
                continue

            metadata[key] = j[key]

        content = {}

        for key in ("summary", "description", "comments", "fixVersion"):
            if key == "comments":
                if j[key]:
                    content[key] = "\n\n".join([comment["content"] for comment in j[key]])
                else:
                    content[key] = ""
            else:
                content[key] = j[key]

        # only keep the values for the content
        content = "\n\n".join(map(str, [value for value in content.values() if value]))

        doc = Document(text=content, metadata=metadata)

        return [doc]


class JiraReaderComments(BaseReader):
    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        # Extract the file path from the args
        file = PosixPath(args[0])

        if file.match("*project.jjson"):
            return []

        with open(file, "r", encoding="utf_8") as f:
            text_dict = f.read()

        j = json.loads(text_dict)

        content = {key: j[key] for key in ("summary", "description", "comments")}
        doc = Document(text=json.dumps(content))

        return [doc]


class CombinedReader(BaseReader):
    """This is a reader to read the structured from custom json files"""

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """
        Load data from a Json file.
        The JSON should at least contain "content" and "metadata" keys.
        The "content" key should contain the text of the document.
        The "metadata" key should contain a dictionary of metadata.
        If attachments are present in the metadata they will be thrown away.

        Args:
            file (Path): Path to the file to load

        Raises:
            TypeError: If the file argument is not a string

        Returns:
            List[Document]: The list of LlamaIndex Documents containing
            the data or an empty list if the file is not valid
        """

        # Extract the file path from the args
        file = args[0]

        with open(file, "r", encoding="utf_8") as f:
            text_dict = f.read()

        try:
            if file.endswith(".cjson"):
                j = json.loads(text_dict)

                content = j["content"]
                metadata = j["metadata"]

                # Remove attachments for now
                if "attachments" in metadata:
                    metadata.pop("attachments")

                doc = Document(text=content, metadata=metadata)
            elif file.endswith(".jjson"):
                j = json.loads(text_dict)

                doc = Document(text=j)

            return [doc]
        except (KeyError, json.decoder.JSONDecodeError):
            print("Invalid JSON file")
            return []
