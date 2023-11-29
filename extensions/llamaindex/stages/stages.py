from dataclasses import dataclass
from typing import Callable, Generator, List

from llama_index.schema import (
    MetadataMode,
    NodeWithScore
)

import regex as re

from extensions.llamaindex.llama_index_extension import BaseRetriever
from extensions.llamaindex.tools.JiraTool import JiraToolSpec


@dataclass
class LlamaIndexVars():
    index: BaseRetriever
    jira_tool: JiraToolSpec
    generate_func: Callable[[str, str, int, dict, List[str]], Generator]


def annotate_response(response, title, header_size=3, sep_line=False) -> str:
    """
    Wraps the response in a markdown block-quote and adds
    the title with a Header of size header-size.
    """
    header = "#" * header_size
    separator = "\n---\n" if sep_line else "\n"

    return f"{header} {title}\n> {response}{separator}"


def get_meta_if_possible(nodes: List[NodeWithScore]) -> str:
    """
    Try to get the title and url of the nodes, if possible.
    Also gives the similarity score if it exists.

    Args:
        nodes (List[NodeWithScore]): List of nodes to get the metadata from
    Returns:
        info (str): String containing the metadata
    """
    info = ""

    for node in nodes:
        info += ">"

        if node.metadata is not None:
            # For confluence reader
            if ("url" in node.metadata.keys()
                    and "title" in node.metadata.keys()):
                info += 'Confluence: [' + node.metadata['title'] + '](' + node.metadata['url'] + ')'

            # For jira reader
            if ("key" in node.metadata.keys()
                    and "summary" in node.metadata.keys()):
                info += 'Jira: [' + node.metadata['summary'] + '](' + node.metadata['key'] + ')'
        else:
            # put a small extract instead
            info += 'No metadata found'
        if node.get_score() is not None:
            info += ' Similarity: ' + str(node.get_score())

        info += '\n'

    return info


def desc_and_template(question, function, additional=""):
    """
    Generates a prompt in which a python function is called with the given question.
    """

    args = function.__annotations__  # Get the arguments of the function

    reformed = [f"{name}: {type_.__name__}" if name != "return" else "" for name, type_ in list(zip(args.keys(), args.values()))]

    reformed = [x for x in reformed if x != ""]

    function_description = (
        function.__name__ + str(reformed).replace("[", "(").replace("]", ")") + ":" + function.__doc__
    )

    prompt_template = (
        "System:\n"
        "You (HaulogyBot) are a helpful chatbot that answers questions about Haulogy. You can access with one line commands special tools such as the Jira search tool.\n"
        "Task:\n"
        "Given the following function description:\n"
        "```\n"
        "{doc}\n"
        "```\n"
        "And the following request:\n"
        "```\n"
        "{question}\n"
        "```\n"
        "{additional}"
        "Call the function with the correct arguments in one line of Python code.\n"
        "It is imperative your answer must only contain the described function and its arguments without any text.\n"
        "HaulogyBot:\n"
    )

    return prompt_template.format(doc=function_description, question=question, additional=additional)


def query_jira_issue(question: str, seed: int, state: dict, stopping_strings: List[str], function, gen: Callable[[str, str, int, dict, List[str]], Generator]):
    """
    In this first stage, we will let the LLM use the first functionality of the Jira tool:
    - Query Jira for issues

    This function return a string containing a modified question to fulfill that purpose.
    """

    prompt = desc_and_template(question, function)

    return gen(prompt, prompt, seed, state, stopping_strings)


def query_jira_detail(question: str, issues: str, seed: int, state: dict, stopping_strings: List[str], function, gen: Callable[[str, str, int, dict, List[str]], Generator]):
    """
    In ths stage, we will let the LLM use the second functionality of the Jira tool:
    - Detail an issue

    This function return a string containing a modified question to fulfill that purpose.
    """

    additional_prompt = (
        "This is the list of issues returned by the Jira tool:\n"
        "```\n"
        "{issues}\n"
        "```\n"
    ).format(issues=issues)

    prompt = desc_and_template(question, function, additional=additional_prompt)

    return gen(prompt, prompt, seed, state, stopping_strings)


def tool_parse(response: str, function) -> str:
    """
    In the best case, the LLM will have responded by calling the Jira tool.
    This function parses the answer and calls the tool for real.
    """

    function_name = function.__name__

    # Parse the response by extracting only the keyword(s)
    arguments = re.findall(rf"{function_name}\((.+)\)", response)

    if len(arguments) == 0:
        raise Exception("No arguments found in response for function " + function.__name__)

    # Remove if necessary the "argument = " part

    if arguments.__contains__("="):
        arguments[0] = arguments[0].split("=")[1]

    if arguments.__contains__(":"):
        arguments[0] = arguments[0].split(":")[1]

    # Remove the quotes if necessary

    arguments[0] = arguments[0].replace("'", "")
    arguments[0] = arguments[0].replace('"', "")

    return function(arguments[0])


def chunk_detail(context: str, question, seed: int, state: dict, stopping_strings: List[str], start: int, end: int, gen: Callable[[str, str, int, dict, List[str]], Generator]):
    """
    In this stage, we will post-process the retrieved context so that it can enter in the LLM.
    To do this, we will pass a rolling window of 4096 characters to create summaries of the context.

    All the summaries will be concatenated together to provide a final context.
    """

    prompt = (
        "Context:\n"
        "```\n"
        "{context}\n"
        "```\n"
        "Above is a portion of text you must summarize so that you keep all the information needed to answer the query.\n"
        "Query: {question}\n"
        "HaulogyBot:\n"
    )

    prompt = prompt.format(question=question, context=context[start:end])

    return gen(prompt, prompt, seed, state, stopping_strings)


def jira_pipeline(question: str, seed: int, state: dict, stopping_strings: List[str], llama_index_vars: LlamaIndexVars):
    """
    This function is the main pipeline for the Jira tool.
    It will call all the functions above in order.
    """

    all_responses = []

    ans = ""
    gen = llama_index_vars.generate_func

    # First, query Jira for issues
    query_function = llama_index_vars.jira_tool.jira_query
    response = query_jira_issue(question, seed, state, stopping_strings, query_function, gen)

    # Unroll the generator
    for ans in response:
        yield ans
    response = ans
    all_responses.append(annotate_response(response, "Jira Query"))

    # Parse the response and call the tool
    issues = tool_parse(response, query_function)

    # Then, detail an issue
    query_function_detail = llama_index_vars.jira_tool.detail_issue
    response = query_jira_detail(question, issues, seed, state, stopping_strings, query_function_detail, gen)

    # Unroll the generator
    for ans in response:
        yield "\n".join(all_responses) + "\n" + ans
    response = ans
    state["jira_metadata"] = "Issue queried:\n> " + ans + "\nCandidate issues:\n> " + issues.replace("\n", "\n>") + "\n"
    all_responses.append(annotate_response(response, "Jira Detail"))

    # Parse the response and call the tool
    context = tool_parse(response, query_function_detail)

    # Chunkify the context using a rolling window of 4096 characters
    # Do this until the result fits in 4096 characters
    last_pass = False
    length_skip = 10000
    while not last_pass:
        if len(context) <= length_skip:
            last_pass = True

        answers = []
        for start in range(0, len(context), length_skip):
            end = min(start + length_skip, len(context))
            response = chunk_detail(context, question, seed, state, stopping_strings, start, end, gen)

            # Unroll the generator
            for ans in response:
                yield "\n".join(all_responses) + "\n" + ans
            answers.append(ans)
            all_responses.append(annotate_response(ans, "Jira Chunk"))
        context = "\n".join(answers)
        all_responses = all_responses[:-len(answers)]

    # Return the final context, which is supposed to be the final summary
    yield "\n".join(all_responses) + "\n" + annotate_response(context, "Jira Summary")


def query_confluence_index(question: str, seed: int, state: dict, stopping_strings: List[str], llama_index_vars: LlamaIndexVars):
    """
    In this stage, we will use standard retrieval to retrieve the most relevant documents and get a response from the LLM.
    """

    resp = llama_index_vars.index.retrieve(question)

    context = "\n\n".join([x.get_content(metadata_mode=MetadataMode.NONE)
                            for x in resp])

    state['index_metadata'] = get_meta_if_possible(resp)

    prompt = (
        "System:\n"
        "You (HaulogyBot) are a helpful chatbot that answers questions about Haulogy.\n"
        "The retrieved documents are below:\n"
        "```\n"
        "{context}\n"
        "```\n"
        "Given the information from the retrieved documents, answer the query.\n"
        "Query: {question}\n"
        "HaulogyBot:\n"
    )

    prompt = prompt.format(context=context, question=question)

    return llama_index_vars.generate_func(prompt, prompt, seed, state, stopping_strings)


def compose_response(question: str, first_response: str, second_response: str, seed: int, state: dict, stopping_strings: List[str], llama_index_vars: LlamaIndexVars):
    """
    In this last stage, we will present both answers from the Jira tool and the LLM and let the LLM improve them into a final response.
    """

    prompt = (
        "System:\n"
        "You (HaulogyBot) are a helpful chatbot that answers questions about Haulogy.\n"
        "The first response using the Jira tool is below:\n"
        "```\n"
        "{first_response}\n"
        "```\n"
        "The second response using the search tool is below:\n"
        "```\n"
        "{second_response}\n"
        "```\n"
        "Given the information from the two responses, answer the query.\n"
        "Query: {question}\n"
        "HaulogyBot:\n"
    )

    prompt = prompt.format(first_response=first_response, second_response=second_response, question=question)

    return llama_index_vars.generate_func(prompt, prompt, seed, state, stopping_strings)


def conf_jira_pipeline(question: str, seed: int, state: dict, stopping_strings: List[str], llama_index_vars: LlamaIndexVars):
    """
    This function is the main pipeline for the Confluence and Jira tools.
    It will call all the functions above in order.
    """

    ans = ""
    all_responses = []

    # Use the jira pipeline to get the first response
    response = jira_pipeline(question, seed, state, stopping_strings, llama_index_vars)

    # Unroll the generator
    for ans in response:
        yield ans
    all_responses.append(annotate_response(ans, "Jira Pipeline", header_size=2, sep_line=True))

    first_response = ans

    # Then, use the search tool to get the second response
    response = query_confluence_index(question, seed, state, stopping_strings, llama_index_vars)

    # Unroll the generator
    for ans in response:
        yield "\n".join(all_responses) + "\n" + ans
    all_responses.append(annotate_response(ans, "Confluence search Tool", header_size=2, sep_line=True))

    second_response = ans

    # Finally, compose the two responses
    response = compose_response(question, first_response, second_response, seed, state, stopping_strings, llama_index_vars)

    # Unroll the generator
    for ans in response:
        yield "\n".join(all_responses) + "\n" + annotate_response(ans, "Final answer", header_size=1)
