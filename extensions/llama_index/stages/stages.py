"""
This modules contains all the different functions used
in the pipeline for answering user queries.

In particular, it contains :
- Offline Confluence querying tool
- Online Jira querying tool
- Summarization tool
"""

from typing import Callable, Generator, List, Optional
import regex as re

from attr import dataclass

from wandb.sdk.data_types.trace_tree import Trace

from llama_index.schema import (
    MetadataMode,
    NodeWithScore
)

from extensions.llama_index.llama_index_extension import BaseRetriever
from extensions.llama_index.tools.jira_tool import JiraToolSpec, JiraQueryError


@dataclass
class LlamaIndexVars():
    """Dataclasss containing the variables needed for the pipeline"""
    index: BaseRetriever
    jira_tool: JiraToolSpec
    generate_func: Callable[[str, str, int, dict, List[str]], Generator]
    current_span: Optional[Trace] = None


def annotate_response(response, title, header_size=3, sep_line=False, quote=True) -> str:
    """
    Wraps the response in a markdown block-quote and adds
    the title with a Header of size header-size.
    """
    header = "#" * header_size
    separator = "\n---\n" if sep_line else "\n"
    quotation = "> " if quote else ""

    return f"{header} {title}\n{quotation}{response}{separator}"


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

    reformed = [f"{name}: {type_.__name__}"
                if name != "return"
                else "" for name, type_ in list(zip(args.keys(), args.values()))]

    reformed = [x for x in reformed if x != ""]

    function_description = function.__name__ +\
        str(reformed).replace("[", "(").replace("]", ")") +\
            ":" + function.__doc__

    prompt_template = (
        "System:\n"
        "You (HaulogyBot) are a helpful chatbot that answers questions about Haulogy. \
            You can access with one line commands special tools such as the Jira search tool.\n"
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
        "It is imperative your answer must only contain the described \
            function and its arguments without any text.\n"
        "HaulogyBot:\n"
        "{function_name}"
    )

    return prompt_template.format(doc=function_description, question=question,
                                  additional=additional, function_name=function.__name__)


def query_jira_issue(question: str, seed: int, state: dict,
                     stopping_strings: List[str], function,
                     gen: Callable[[str, str, int, dict, List[str]], Generator]):
    """
    In this first stage, we will let the LLM use the first functionality of the Jira tool:
    - Query Jira for issues

    This function return a string containing a modified question to fulfill that purpose.
    """

    prompt = desc_and_template(question, function)

    return gen(prompt, prompt, seed, state, stopping_strings)


def query_jira_detail(question: str, issues: str, seed: int,
                      state: dict, stopping_strings: List[str],
                      function, gen: Callable[[str, str, int, dict, List[str]], Generator]):
    """
    In ths stage, we will let the LLM use the second functionality of the Jira tool:
    - Detail an issue

    This function return a string containing a modified question to fulfill that purpose.
    """

    additional_prompt = (
        "This is the list of issues returned by the Jira tool:\n"
        "```\n"
        f"{issues}\n"
        "```\n"
    )

    prompt = desc_and_template(question, function, additional=additional_prompt)

    return gen(prompt, prompt, seed, state, stopping_strings)


def tool_parse(response: str, function) -> str:
    """
    In the best case, the LLM will have responded by calling the Jira tool.
    This function parses the answer and calls the tool for real.
    """

    function_name = function.__name__

    # Parse the response by extracting only the keyword(s)
    arguments = re.findall(r"\(([^\(\)]+)\)", response)

    if len(arguments) == 0:
        return f"FAILED: The function {function_name} was not called.\
            Make sure to include {function_name}, parentheses and the keywords in between."

    arguments = arguments[0]

    # Remove if necessary the "argument = " part

    if "=" in arguments:
        arguments = arguments.split("=")[1]

    if ":" in arguments:
        arguments = arguments.split(":")[1]

    # Remove the quotes if necessary

    arguments = arguments.replace("'", "")
    arguments = arguments.replace('"', "")

    # Finally, trim the arguments
    arguments = arguments.strip()

    try:
        return function(arguments)
    except JiraQueryError as e:
        return f"ERROR: The function {function_name} was called with\
            arguments {arguments} but failed with the following error: {e}"


def chunk_detail(context: str, question, seed: int,
                 state: dict, stopping_strings: List[str],
                 gen: Callable[[str, str, int, dict, List[str]], Generator]):
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
        "Above is a portion of text you must summarize so that\
            you keep all the information needed to answer the query.\n"
        "Query: {question}\n"
        "Do not include any outside information in your answer.\n"
        "HaulogyBot:\n"
    )

    prompt = prompt.format(question=question, context=context)

    return gen(prompt, prompt, seed, state, stopping_strings)


def jira_pipeline(question: str, seed: int, state: dict,
                  stopping_strings: List[str],
                  llama_index_vars: LlamaIndexVars):
    """
    This function is the main pipeline for the Jira tool.
    It will call all the functions above in order.
    """

    # Initialize the variables
    all_responses = []

    ans = ""
    gen = llama_index_vars.generate_func

    current_span = llama_index_vars.current_span

    # First, query Jira for issues
    query_function = llama_index_vars.jira_tool.jira_query
    response = query_jira_issue(question, seed, state,
                                stopping_strings, query_function, gen)

    # Unroll the generator
    for ans in response:
        yield ans
    response = ans
    all_responses.append(annotate_response(response, "Jira Query"))

    wandb_trace("HaulogyBot_Jira_Query", question, response, current_span)

    # Parse the response and call the tool
    issues = tool_parse(response, query_function)

    wandb_trace("HaulogyBot_Jira_Query_Parse", response, issues, current_span)

    if issues.startswith("FAILED") or issues.startswith("ERROR"):
        yield "\n".join(all_responses) + "\n" + issues
        return

    # Then, detail an issue
    query_function_detail = llama_index_vars.jira_tool.detail_issue
    response = query_jira_detail(question, issues, seed, state,
                                 stopping_strings, query_function_detail, gen)

    # Unroll the generator
    for ans in response:
        yield "\n".join(all_responses) + "\n" + ans
    response = ans
    state["jira_metadata"] = "Issue queried:\n> " +\
        ans + "\nCandidate issues:\n> " +\
            issues.replace("\n", "\n>") + "\n"
    all_responses.append(annotate_response(response, "Jira Detail"))

    wandb_trace("HaulogyBot_Jira_Detail", question, response, current_span)

    # Parse the response and call the tool
    context = tool_parse(response, query_function_detail)

    wandb_trace("HaulogyBot_Jira_Detail_Parse", response, context, current_span)

    if context.startswith("FAILED") or context.startswith("ERROR"):
        yield "\n".join(all_responses) + "\n" + context
        return

    # create a span for the summarize stage
    summarize_span = Trace(
        name="Summarize",
    )

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
            response = chunk_detail(context[start:end], question,
                                    seed, state, stopping_strings, gen)

            # Unroll the generator
            for ans in response:
                yield "\n".join(all_responses) + "\n" + ans
            answers.append(ans)
            all_responses.append(annotate_response(ans, "Jira Chunk"))

            wandb_trace(f"HaulogyBot_Jira_Chunk_{start}_{end}", context[start:end],
                        ans, summarize_span)

        context = "\n".join(answers)
        all_responses = all_responses[:-len(answers)]

    if current_span is not None:
        current_span.add_child(summarize_span)

    # Return the final context, which is supposed to be the final summary
    yield "\n".join(all_responses) + "\n" + annotate_response(context, "Jira Summary")


def query_confluence_index(question: str, seed: int, state: dict,
                           stopping_strings: List[str],
                           llama_index_vars: LlamaIndexVars):
    """
    In this stage, we will use standard retrieval to retrieve the most
    relevant documents and get a response from the LLM.
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


def compose_response(question: str, first_response: str,
                     second_response: str, seed: int, state: dict,
                     stopping_strings: List[str],
                     llama_index_vars: LlamaIndexVars):
    """
    In this last stage, we will present both answers from
    the Jira tool and the LLM and let the LLM improve them into a final response.
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

    prompt = prompt.format(first_response=first_response,
                           second_response=second_response, question=question)

    return llama_index_vars.generate_func(prompt, prompt,
                                          seed, state, stopping_strings)


def restate_question(question: str, seed: int, state: dict,
                     stopping_strings: List[str],
                     llama_index_vars: LlamaIndexVars):
    """
    Given the chat history and the question,
    restate the question in one sentence.
    """

    prompt = (
        "{question}\n"
        "Given the chat history above, \
            restate the last user's question in a single sentence.'\n"
        "Keep the question brief. It should not be longer than 50 words.\n"
        "HaulogyBot:\n"
    )

    prompt = prompt.format(question=question)

    return llama_index_vars.generate_func(prompt, prompt, seed, state, stopping_strings)


def wandb_trace(name: str, inputs: str, outputs: str, parent: Trace | None):
    """Creates a child trace for the current span"""
    conf_jira_span = Trace(
        name=name,
        inputs={"query": inputs},
        outputs={"response": outputs},
    )

    if parent is not None:
        parent.add_child(conf_jira_span)


def conf_jira_pipeline(question: str, seed: int, state: dict,
                       stopping_strings: List[str],
                       llama_index_vars: LlamaIndexVars):
    """
    This function is the main pipeline for the Confluence and Jira tools.
    It will call all the functions above in order.
    """

    # Initialize the variables
    all_responses = []
    current_span = llama_index_vars.current_span

    # First, restate the question
    response_gen_restate = restate_question(question, seed,
                                            state, stopping_strings, llama_index_vars)

    # Unroll the generator
    restated_question = ""
    for restated_question in response_gen_restate:
        yield restated_question
    all_responses.append(annotate_response(restated_question, "Restated question", header_size=2,
                                           sep_line=True, quote=True))

    wandb_trace("HaulogyBot_Restate", question, restated_question, current_span)

    # Use the jira pipeline to get the first response
    response_gen_jira = jira_pipeline(restated_question, seed,
                                      state, stopping_strings, llama_index_vars)

    # Unroll the generator
    first_response = ""
    for first_response in response_gen_jira:
        yield first_response
    all_responses.append(annotate_response(first_response, "Jira Pipeline", header_size=2,
                                           sep_line=True, quote=False))

    wandb_trace("HaulogyBot_Jira", restated_question, first_response, current_span)

    # Then, use the search tool to get the second response
    response_gen_conf = query_confluence_index(restated_question, seed,
                                               state, stopping_strings, llama_index_vars)

    # Unroll the generator
    second_response = ""
    for second_response in response_gen_conf:
        yield "\n".join(all_responses) + "\n" + second_response
    all_responses.append(annotate_response(second_response, "Confluence search Tool",
                                           header_size=2, sep_line=True))

    wandb_trace("HaulogyBot_Confluence", restated_question, second_response, current_span)

    # Finally, compose the two responses
    response = compose_response(restated_question, first_response,
                                second_response, seed, state,
                                stopping_strings, llama_index_vars)

    # Unroll the generator
    final_response = ""
    for final_response in response:
        yield "\n".join(all_responses) + "\n" + final_response

    wandb_trace("HaulogyBot_Compose", restated_question, final_response, current_span)

    yield "\n".join(all_responses) + "\n" +\
        annotate_response(final_response, "Final answer", header_size=1)
