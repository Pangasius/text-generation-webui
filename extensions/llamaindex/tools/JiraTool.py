"""This module is used to dynamically load Jira issues when they are requested"""

from typing import Dict, Optional
from attr import dataclass
import requests
import json

from llama_index.tools import BaseTool

URL_BASE = "https://jira.haulogy.net/jira/rest/api/2/search"

HEADERS = {
    "Accept": "application/json"
}


@dataclass
class JiraQuery:
    """This class represents a query to Jira"""
    jql: str
    startAt: int = 0
    maxResults: int = 4
    validateQuery: bool = True
    fields: list[str] = ["summary", "description", "comment"]
    expand: str = ""

    @staticmethod
    def default_keyword_query(keyword: list[str], categories: list[str]):
        # category in (...) AND text ~ "keyword"
        query = "category in ("
        for i in range(len(categories)):
            query += '"' + categories[i] + '"'
            if i < len(categories) - 1:
                query += ","
        query += ") AND text ~ "
        for i in range(len(keyword)):
            query += '"' + keyword[i] + '"'
            if i < len(keyword) - 1:
                query += " OR text ~ "
        query += " AND status in (\"Closed\", \"Done\")"
        query += " AND created > \"2018-01-01\""
        query += " ORDER BY created DESC"

        return JiraQuery(jql=query)


class JiraToolSpec(BaseTool):
    """
    This class contains different functions to interact with Jira.

    Each of them returns a string.
    If an error occurs, the string starts with "FAILED: " and the error message follows.
    If the error is unrecoverable, an exception is raised instead.
    """

    name = "JiraTool"
    description = "This tool is used to dynamically load Jira issues when they are requested"
    spec_functions = ["jira_query", "detail_issue"]

    def __init__(self, app_id: Optional[str] = None) -> None:
        with open("extensions/llamaindex/tools/details.json", "r") as infile:
            config = json.load(infile)

        self.categories = config["categories"]
        self.login_data = config["login_data"]
        self.last_results = None
        self.last_details = None
        self.last_summary = None
        self.cookies = self._get_cookies()

    def _get_cookies(self) -> Dict[str, str]:
        """Opens a session with Jira and returns the cookies"""

        r_auth = requests.post('https://jira.haulogy.net/jira/rest/auth/1/session', json=self.login_data)

        if r_auth.status_code != 200:
            raise Exception("Login failed with status code " + str(r_auth.status_code) + ".")

        r_auth = r_auth.json()["session"]
        cookies = {'JSESSIONID': r_auth["value"]}

        return cookies

    def jira_query(self, query: str) -> str:
        """
        Make a query to Jira which contains diverse Haulogy information. Returns a list of issues to investigate.

        Example inputs:
        - rectification billing
        - haugazel, invoice

        Args:
            query (str): a comma separated list of keywords
        """

        # Parse query
        keywords = query.split(",")
        jira_query = JiraQuery.default_keyword_query(keywords, self.categories)

        # Make query
        response = requests.request("GET", URL_BASE, headers=HEADERS, params=jira_query.__dict__, cookies=self.cookies)
        if response.status_code != 200:
            raise Exception("Query failed with status code " + str(response.status_code) + ".")

        # Parse results
        self.last_results = response.json()["issues"]

        if len(self.last_results) == 0:
            return "FAILED: No issue found."

        # Only print the 20 first characters of the summary
        self.last_summary = ""
        for issue in self.last_results:
            summary = issue["fields"]["summary"]
            self.last_summary += issue["key"] + ": " + summary + "\n"

        # Here we have to print or the variable is not saved
        print(self.last_summary)

        return self.last_summary

    def detail_issue(self, issue_requested: str):
        """
        Reads the description and comments about a specific issue.

        Example inputs:
        - HAUGAZEL-123

        Args:
            issue (str): The issue key exactly.
        """
        if self.last_results is None:
            return "FAILED: No query has been made yet, please use jira_query(keywords)."

        # Get the description and comments

        if isinstance(issue_requested, str):
            description = ""
            comments = ""
            for issue in self.last_results:
                if issue["key"] == issue_requested:
                    description = issue["fields"]["description"]
                    comments = issue["fields"]["comment"]["comments"]
                    break
        else:
            return f"FAILED: The issue requested must be a string, not {type(issue_requested)}"

        if description == "":
            return f"FAILED: Issue {issue_requested} not found. Available issues are: {', '.join([issue['key'] for issue in self.last_results])}"

        # Format the result
        self.last_results = "Description: " + description + "\n"
        self.last_results += "Comments:\n"
        for comment in comments:
            self.last_results += comment["body"] + "\n"

        # Here we have to print or the variable is not saved
        print(self.last_results)

        return self.last_results
