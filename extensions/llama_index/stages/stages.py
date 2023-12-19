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

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from modules import shared

from llama_index.node_parser import TokenTextSplitter

from modules.text_generation import get_encoded_length

from modules.logits import get_next_logits

PIPELINE_FAILURE = "...FAILURE"


class Summarizer:
    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum", use_fast=True)

        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")

    def summarize_all(self, texts: List[str]) -> List[str]:
        print([len(x) for x in texts])

        # summarize by batch
        summaries = []
        last_pass = False
        for i in range(0, len(texts), 2):
            input_ids = self.tokenizer(texts[i:i + 2], return_tensors="pt", padding=True).input_ids.to("cuda:0")

            summary_ids = self.model.generate(input_ids, max_length=256, early_stopping=True, min_length=min(len(texts[i]), 64))

            if i + 4 >= len(texts):
                last_pass = True

            summaries.extend([self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=last_pass) for g in summary_ids])

        return summaries


@dataclass
class LlamaIndexVars():
    """Dataclass containing the variables needed for the pipeline"""
    conf_index: BaseRetriever
    jira_index: BaseRetriever
    jira_tool: JiraToolSpec
    generate_func: Callable[[str, str, int, dict, List[str]], Generator]
    summarizer: Summarizer
    current_span: Optional[Trace] = None


def annotate_response(response, title, header_size=3, sep_line=False, quote=True) -> str:
    """
    Wraps the response in a markdown block-quote and adds
    the title with a Header of size header-size.
    """
    header = "#" * header_size
    separator = "\n---\n" if sep_line else "\n"
    quotation = "> " if quote else ""

    if quote:
        response = response.replace("\n", "\n> ")

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


def query_jira_issue(question: str, seed: int, state: dict,
                     stopping_strings: List[str], function,
                     gen: Callable[[str, str, int, dict, List[str]], Generator]):
    """
    In this first stage, we will let the LLM use the first functionality of the Jira tool:
    - Query Jira for issues

    This function return a string containing a modified question to fulfill that purpose.
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
        "{doc}\n"
        "And the following request:\n"
        "{question}\n"
        "Call the function with the correct arguments in one line of Python code.\n"
        "HaulogyBot:\n"
        "{function_name}"
    )

    prompt = prompt_template.format(doc=function_description, question=question, function_name=function.__name__)

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

    arguments = arguments[0].replace(" OR ", ",")
    arguments = arguments.replace(" AND ", ",")

    # Remove if necessary the "argument = " part

    if "=" in arguments:
        arguments = arguments.split("=")[1]

    if ":" in arguments:
        arguments = arguments.split(":")[1]

    if " ~ " in arguments:
        arguments = arguments.split("~")[1]

    # Remove the quotes if necessary

    arguments = arguments.replace("'", "")
    arguments = arguments.replace('"', "")

    # start at the first letter
    first_letter = re.search(r"[a-zA-Z0-9]", arguments)
    if first_letter is not None:
        arguments = arguments[first_letter.start():]
    else:
        return f"FAILED: The function {function_name} was called with\
            arguments {arguments} but no keyword was found."

    # Finally, trim the arguments
    arguments = arguments.strip()

    try:
        return function(arguments)
    except JiraQueryError as e:
        return f"ERROR: The function {function_name} was called with\
            arguments {arguments} but failed with the following error: {e}"


def summarize_contexts(contexts: List[str], summarizer: Summarizer):
    """
    Summarizes the contexts using the summarizer.

    Args:
        contexts (List[str]): A list of contexts to summarize
        summarizer (Summarizer): The summarizer to use
        all_responses (List[str]): The incremental list of responses
    """

    # cause hf tokenizer not directly compatible with llama_index
    def new_tokenizer(x: str):
        return summarizer.tokenizer.encode(x)

    summarizer.model.to("cuda:0")

    for context, index in zip(contexts, range(len(contexts))):
        # Use the summarizer to get the final context
        last_pass = False
        splitter = TokenTextSplitter(tokenizer=new_tokenizer, chunk_size=1024, chunk_overlap=64)

        while not last_pass:
            if get_encoded_length(context) < shared.args.n_ctx:
                last_pass = True

            summary = "\n".join(summarizer.summarize_all(splitter.split_text(context)))

            context = summary

        contexts[index] = context

    summarizer.model.to("cpu")


def determine_usefulness(context: str, question: str, state: dict, all_responses: List[str]):
    """
    Determines if the context is useful for the question.

    Args:
        context (str): The context to check
        question (str): The question to check

    Returns:
        useful (bool): True if the context is useful, False otherwise
    """

    prompt = (
        "Here follows a summary:\n"
        "{context}\n"
        "And a question, that may or may not be related:\n"
        "{question}\n"
        "Can the query be answered with the summary? (Yes/No) \n"
        "Answer:\n"
    )

    # here we will determine the answer based on the logits of the LLM
    logits = get_next_logits(prompt.format(context=context, question=question), state, use_samplers=False, previous=None, return_dict=True)

    # checks to remove uncertainty about the type of the output
    assert isinstance(logits, dict)
    assert "Yes" in logits.keys()
    assert "No" in logits.keys()

    # get the logits for yes and no
    yes_logits = logits["Yes"]
    no_logits = logits["No"]

    emoji = "👍" if yes_logits > no_logits else "👎"

    log_text = f"The logits for **yes** are {yes_logits} and the logits for **no** are {no_logits} concerning the relevance of the context:\n {context}.\
        \nAre you sure the context is relevant? {emoji}"

    all_responses.append(annotate_response(log_text, "Logits", header_size=3,
                                           sep_line=False, quote=True))

    # if the logits for yes are higher than the logits for no, then the answer is yes
    return yes_logits > no_logits


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

    parse_trace = wandb_trace("HaulogyBot_Jira_Query_Parse", response, issues, current_span)

    if issues.startswith("FAILED"):
        # attempt to rescue the search by changing the query
        # for this we broaden the search by relaxing keywords with or
        all_responses.append("[DEBUG] The Jira search tool failed to find an issue.\
            Attempting to rescue the search by relaxing the keywords.\n")
        issues = tool_parse(response.replace(" ", ","), query_function)

        wandb_trace("HaulogyBot_Jira_Query_Parse_Rescue", response, issues, parse_trace)

    # considered unrecoverable at this point
    if issues.startswith("FAILED") or issues.startswith("ERROR"):
        yield "\n".join(all_responses) + "\n" + issues + PIPELINE_FAILURE
        return

    all_responses.append(annotate_response(issues, "Jira Query Result", header_size=3,
                                           sep_line=False, quote=True))

    # Make a quick filtering of issues based on the question
    issues = llama_index_vars.jira_tool.filter_out_issues(filter=lambda x: determine_usefulness(x, question, state, all_responses))

    # In case there is no issue left, we stop the pipeline
    if len(issues) == 0:
        yield "\n".join(all_responses) + "\n" + "No relevant issue found." + PIPELINE_FAILURE
        return

    # Then, we detail the issues
    yield "\n".join(all_responses) + "Summarizing issues..."

    # here the issues are concatenated and passed to be summarized
    contexts = llama_index_vars.jira_tool.get_all_issues_details(issues)

    # summarize the contexts
    summarize_contexts(contexts, llama_index_vars.summarizer)

    # filter out irrelevant contexts
    contexts = [context for context in contexts if determine_usefulness(context, question, state, all_responses)]

    if len(contexts) == 0:
        yield "\n".join(all_responses) + "\n" + "No relevant context found." + PIPELINE_FAILURE
        return

    for context, index in zip(contexts, range(len(contexts))):
        response_gen = in_context_response(context, question, seed, state, stopping_strings, llama_index_vars)

        # Unroll the generator
        response = ""
        for response in response_gen:
            yield "\n".join(all_responses) + "\n" + response

        all_responses.append(annotate_response(response, f"Jira Context {index}", header_size=3,
                                               sep_line=False, quote=True))

        yield "\n".join(all_responses)

        contexts[index] = context

    # Finally, compose the answer by summarizing all responses
    summarize_contexts(contexts=["\n".join(contexts)], summarizer=llama_index_vars.summarizer)

    all_responses.append(annotate_response(contexts[0], "Jira Final Answer", header_size=3,
                                           sep_line=False, quote=True))

    yield "\n".join(all_responses) + "\n" + contexts[0]


def in_context_response(context: str, question: str, seed: int, state: dict,
                 stopping_strings: List[str],
                 llama_index_vars: LlamaIndexVars):
    """A simple prompt to use retrieved documents to answer a question."""

    prompt = (
        "System:\n"
        "You (HaulogyBot) are a helpful chatbot that can answer Haulogy related questions thanks to summaries.\n"
        "The retrieved documents are below:\n"
        "{context}\n"
        "Given the information from the retrieved documents, answer the query below:\n"
        "{question}\n"
        "If no answer can be found, answer with 'UNRELATED'.\n"
        "HaulogyBot:\n"
    )

    prompt = prompt.format(context=context, question=question)

    return llama_index_vars.generate_func(prompt, prompt, seed, state, stopping_strings)


def query_index(question: str, seed: int, state: dict,
                           stopping_strings: List[str],
                           index: BaseRetriever,
                           llama_index_vars: LlamaIndexVars):
    """
    In this stage, we will use standard retrieval to retrieve the most
    relevant documents and get a response from the LLM.
    """

    resp = index.retrieve(question)

    context = "\n\n".join([x.get_content(metadata_mode=MetadataMode.NONE)
                            for x in resp])

    if "index_metadata" not in state.keys():
        state['index_metadata'] = []

    state['index_metadata'].append(get_meta_if_possible(resp))

    print("Querying index...")

    return in_context_response(context, question, seed, state, stopping_strings, llama_index_vars)


def compose_response(question: str, responses: List[str], seed: int, state: dict,
                     stopping_strings: List[str],
                     llama_index_vars: LlamaIndexVars):
    """
    In this last stage, we will present both answers from
    the Jira tool and the LLM and let the LLM improve them into a final response.
    """

    prompt = (
        "System:\n"
        "You (HaulogyBot) are a helpful chatbot that answers questions about Haulogy.\n"
        "Here follows a list of potential responses to the query from different sources:\n"
        "{responses}\n"
        "Given the information above, answer the query.\n"
        "Query: {question}\n"
        "HaulogyBot:\n"
    )

    prompt = prompt.format(responses=responses, question=question)

    return llama_index_vars.generate_func(prompt, prompt,
                                          seed, state, stopping_strings)


def wandb_trace(name: str, inputs: str, outputs: str, parent: Trace | None):
    """Creates a child trace for the current span"""
    conf_jira_span = Trace(
        name=name,
        inputs={"query": inputs},
        outputs={"response": outputs},
    )

    if parent is not None:
        parent.add_child(conf_jira_span)

    return conf_jira_span


def conf_jira_pipeline(question: str, seed: int, state: dict,
                       stopping_strings: List[str],
                       llama_index_vars: LlamaIndexVars):
    """
    This function is the main pipeline for the Confluence and Jira tools.
    It will call all the functions above in order.
    """

    # Initialize the variables
    all_responses = []
    to_summarize = []
    current_span = llama_index_vars.current_span

    # Use the jira pipeline to get the first response
    print("Querying Jira...")
    response_gen_jira = jira_pipeline(question, seed,
                                      state, stopping_strings, llama_index_vars)

    # Unroll the generator
    first_response = ""
    for first_response in response_gen_jira:
        yield first_response

    all_responses.append(annotate_response(first_response, "Jira Pipeline", header_size=2,
                                           sep_line=False, quote=False))

    to_summarize.append(first_response)

    wandb_trace("HaulogyBot_Jira", question, first_response, current_span)

    # Use the search tool to get the second response
    print("Querying Jira index...")
    jira_index = llama_index_vars.jira_index
    response_gen_search = query_index(question, seed,
                                    state, stopping_strings,
                                    jira_index,
                                    llama_index_vars)

    # Unroll the generator
    jira_sim_response = ""
    for jira_sim_response in response_gen_search:
        yield "\n".join(all_responses) + "\n" + jira_sim_response

    all_responses.append(annotate_response(jira_sim_response, "Jira Similarity Search Tool",
                                        header_size=3, sep_line=True))

    to_summarize.append(jira_sim_response)

    wandb_trace("HaulogyBot_Jira_Sim_Search", question, jira_sim_response, current_span)

    # Then, use the confluence search tool to have the third response
    print("Querying Confluence index...")
    conf_index = llama_index_vars.conf_index
    response_gen_conf = query_index(question, seed,
                                               state, stopping_strings,
                                               conf_index,
                                               llama_index_vars)

    # Unroll the generator
    conf_sim_response = ""
    for conf_sim_response in response_gen_conf:
        yield "\n".join(all_responses) + "\n" + conf_sim_response

    all_responses.append(annotate_response(conf_sim_response, "Confluence search Tool",
                                           header_size=2, sep_line=True))

    to_summarize.append(conf_sim_response)

    wandb_trace("HaulogyBot_Confluence", question, conf_sim_response, current_span)

    # Finally, compose the two responses
    response = compose_response(question, to_summarize, seed, state,
                                stopping_strings, llama_index_vars)

    # Unroll the generator
    final_response = ""
    for final_response in response:
        yield "\n".join(all_responses) + "\n" + final_response

    wandb_trace("HaulogyBot_Compose", question, final_response, current_span)

    yield "\n".join(all_responses) + "\n" +\
        annotate_response(final_response, "Final answer", header_size=1)
