"""This module is used to dynamically load Jira issues when they are requested"""

from typing import Dict, Optional
from attr import dataclass
from llama_index.tools import BaseTool
import regex
import requests
import json

URL_BASE = "https://jira.haulogy.net/jira/rest/api/2/search"

HEADERS = {
    "Accept": "application/json"
}


@dataclass
class JiraQuery:
    """This class represents a query to Jira"""
    jql: str
    startAt: int = 0
    maxResults: int = 3
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

    name = "JiraTool"
    description = "This tool is used to dynamically load Jira issues when they are requested"
    spec_functions = ["jira_query", "detail_issue"]

    def __init__(self, app_id: Optional[str] = None) -> None:
        with open("extensions/llamaindex/tools/details.json", "r") as infile:
            config = json.load(infile)

        self.categories = config["categories"]
        self.login_data = config["login_data"]
        self.last_results = None

    def get_cookies(self) -> Dict[str, str]:
        """Opens a session with Jira and returns the cookies"""
        print("Logging in Jira ...")
        r_auth = requests.post('https://jira.haulogy.net/jira/rest/auth/1/session', json=self.login_data)
        if r_auth.status_code != 200:
            print("Login failed with status code " + str(r_auth.status_code) + ".")
            raise Exception("Login failed.")
        else:
            print("Login successful.")
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

        # Connect and get cookies
        cookies = self.get_cookies()

        # Parse query
        keywords = query.split(",")
        jira_query = JiraQuery.default_keyword_query(keywords, self.categories)

        # Make query
        response = requests.request("GET", URL_BASE, headers=HEADERS, params=jira_query.__dict__, cookies=cookies)
        if response.status_code != 200:
            print("Query failed with status code " + str(response.status_code) + ".")
            raise Exception("Query failed.")
        else:
            print("Query successful.")

        # Parse results
        r = response.json()
        issues = r["issues"]

        self.last_results = issues

        # Only print the 20 first characters of the summary
        summaries = ""
        for issue in issues:
            summary = issue["fields"]["summary"]
            summaries += issue["key"] + ": " + summary[:100] + ("...\n" if len(summary) > 100 else "\n")

        return summaries

    def detail_issue(self, issue_requested: str):
        """
        Read the description and comments about a specific issue.

        Example inputs:
        - HAUGAZEL-123
        - 1

        Args:
            issue (str): The issue key exactly or the cardinal number indicating the position in the last results from the top = 0.
        """
        if self.last_results is None:
            return "Make a query before detailing its issues."

        # Get the description and comments

        if regex.match(r"^\d+$", issue_requested):
            issue_requested = int(issue_requested)
            if issue_requested < len(self.last_results):
                description = self.last_results[issue_requested]["fields"]["description"]
                comments = self.last_results[issue_requested]["fields"]["comment"]["comments"]
            else:
                return "The issue requested must be a string or an integer."

        elif isinstance(issue_requested, str):
            description = ""
            comments = ""
            for issue in self.last_results:
                if issue["key"] == issue_requested:
                    description = issue["fields"]["description"]
                    comments = issue["fields"]["comment"]["comments"]
                    break
        else:
            return "The issue requested must be a string or an integer."

        if description == "":
            return "Issue not found."

        # Format the result
        result = "Description: " + description + "\n"
        result += "Comments:\n"
        for comment in comments:
            result += comment["body"] + "\n"

        # Clear the last results
        self.last_results = None

        return result
