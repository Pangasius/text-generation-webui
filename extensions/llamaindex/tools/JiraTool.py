"""This module is used to dynamically load Jira issues when they are requested"""

from typing import Dict, Optional
from attr import dataclass
from llama_index.tools import BaseTool
import requests
import json

URL_BASE = "https://jira.haulogy.net/jira/rest/api/2/search/"


@dataclass
class JiraQuery:
    """This class represents a query to Jira"""
    jql: str
    startsAt: int = 0
    maxResults: int = 10
    validateQuery: bool = True
    fields: str = "summary,description,issuetype,project,creator,reporter,created,updated,status,labels,attachment,comment"
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
        query += " ORDER BY created DESC"

        return JiraQuery(jql=query)


class JiraToolSpec(BaseTool):

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
            exit(1)
        else:
            print("Login successful.")
        r_auth = r_auth.json()["session"]
        cookies = {'JSESSIONID': r_auth["value"]}

        return cookies

    def jira_query(self, query: str) -> str:
        """
        Make a query to Jira which contains diverse Haulogy information. Returns a list of issues to investigate.

        Example inputs:
        - \"rectification billing\"
        - \"haugazel, invoice\"

        Args:
            query (str): a comma separated list of keywords
        """

        # Connect and get cookies
        cookies = self.get_cookies()

        # Parse query
        keywords = query.split(",")
        jira_query = JiraQuery.default_keyword_query(keywords, self.categories)

        # Make query
        r = requests.post(URL_BASE, json=jira_query.__dict__, cookies=cookies)
        if r.status_code != 200:
            print("Query failed with status code " + str(r.status_code) + ".")
            exit(1)
        else:
            print("Query successful.")

        # Parse results
        r = r.json()
        issues = r["issues"]

        self.last_results = issues

        # Only print the 20 first characters of the summary
        summary = ""
        for issue in issues:
            summary += issue["key"] + ": " + issue["fields"]["summary"][:20] + "\n"

        return summary

    def detail_issue(self, issue: "str"):
        """
        Read the description and comments about a specific issue.

        Example inputs:
        - \"HAUGAZEL-123\"

        Args:
            issue (str): The issue key
        """
        if self.last_results is None:
            return "Make a query before detailing its issues."

        # Get the description and comments
        description = ""
        comments = ""
        for issue in self.last_results:
            if issue["key"] == issue:
                description = issue["fields"]["description"]
                comments = issue["fields"]["comment"]["comments"]
                break

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
