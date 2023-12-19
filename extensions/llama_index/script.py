"""Module that define the extensions basic functions and calls the llama_index_extension."""
import datetime
import traceback

import nest_asyncio

import wandb
from wandb.sdk.data_types.trace_tree import Trace
from extensions.llama_index.tools.jira_tool import JiraToolSpec

from modules import shared

from modules.text_generation import (
    generate_reply_HF,
    generate_reply_custom,
)

from extensions.llama_index.llama_index_extension import IndexEngine

from extensions.llama_index.stages.stages import (
    LlamaIndexVars,
    conf_jira_pipeline,
    Summarizer
)

global LLAMA_INDEX_VARS
LLAMA_INDEX_VARS = None

CONF_DATASET = "conf_3_12_2023"
CONF_INDEX_NAME = "conf_entity"

JIRA_DATASET = "jira_3_12_2023"
JIRA_INDEX_NAME = "jira_entity"

WANDB_LOG = False


def setup():
    """
    This function is called once when the extension is loaded.
    After the model has been loaded.
    """
    nest_asyncio.apply()

    # TODO: Change when go to production
    if WANDB_LOG:
        wandb.init(project="Haulogy-First-Test")

    # basic checks to avoid infinite loops
    if shared.args.n_ctx < 512:
        raise Exception("The context length is too short. Please increase it.")

    # Initialize the index engine so that a maximum of models are shared
    index = IndexEngine()

    confluence_retriever = index.as_retriever(
                                              index_name=CONF_INDEX_NAME,
                                              dataset=CONF_DATASET,
                                              kg=False,
                                              fine_tune=False,
                                              build_index=False
                                            )

    jira_retriever = index.as_retriever(
                                              index_name=JIRA_INDEX_NAME,
                                              dataset=JIRA_DATASET,
                                              kg=False,
                                              fine_tune=False,
                                              build_index=False
                                            )

    jira_tool = JiraToolSpec()

    generate_func = get_generate_func()

    summarizer = Summarizer()

    global LLAMA_INDEX_VARS
    LLAMA_INDEX_VARS = LlamaIndexVars(conf_index=confluence_retriever,
                                      jira_index=jira_retriever,
                                      jira_tool=jira_tool,
                                      generate_func=generate_func,
                                      summarizer=summarizer)


def output_modifier(output: str, state: dict, is_chat=False):
    """
    This function is called after the model has generated a reply.
    Adds the sources corresponding to the title, url and similarity score.
    In case the metadata is not available, it skips this step.

    Args:
        string (str): the original reply
        state (dict): the state of the chatbot
        is_chat (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    if "index_metadata" in state.keys():
        output = output + "\nSources : \n\n"
        for metadata in state["index_metadata"]:
            output = output + metadata
        state["index_metadata"] = []

    return output


def get_generate_func():
    """
    Chooses the right generate function depending on the model.
    """

    if shared.model.__class__.__name__ in ['LlamaCppModel',
                                           'RWKVModel',
                                           'ExllamaModel',
                                           'Exllamav2Model',
                                           'CtransformersModel']:
        generate_func = generate_reply_custom
    else:
        generate_func = generate_reply_HF

    return generate_func


def custom_generate_reply(*args, **kwargs):
    """
    This function is called when the user presses the "Generate Reply" button.
    Here it acts as a wrapper to log the inputs and outputs of the model using wandb.

    Args:
        question (str): the question
        original_question (str): the original question,
                                 is not the same as question only
                                 for non-chat mode
        seed (int): the generation seed
        state (dict): the state of the chatbot
        stopping_strings (List, optional): List containing the strings to
                                           put an early stop the production.
                                           Defaults to None.
        is_chat (bool, optional): Wether the chatbot is in chat mode.
                                  Defaults to False.

    Yields:
        str: The generated reply
    """

    if LLAMA_INDEX_VARS is None:
        raise Exception("The extension has not been initialized.")

    start_time_ms = round(datetime.datetime.now().timestamp() * 1000)

    llm_span = Trace(
            name=shared.model_name,
            kind="llm",
            start_time_ms=start_time_ms,
            metadata={"error": "",
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": shared.model_name,
                    "llama_index_vars": LLAMA_INDEX_VARS,
                    "config": shared.model_config})

    LLAMA_INDEX_VARS.current_span = llm_span

    question, original_question, seed, state, stopping_strings = args[:5]

    original_question = question[question.rfind("You:") + 5:question.rfind("HaulogyBot:")]

    state["truncation_length"] = shared.args.n_ctx

    ans = ""
    try:
        for ans in conf_jira_pipeline(question, seed, state, stopping_strings, LLAMA_INDEX_VARS):
            yield ans

    except RuntimeError as e:
        traceback.print_exc()
        if llm_span.metadata:
            llm_span.metadata["error"] = str(e)
    finally:

        end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        llm_span.inputs = {"query": original_question,
                           "history": question}
        llm_span.end_time_ms = end_time_ms
        llm_span.outputs = {"response": ans}

        if WANDB_LOG:
            llm_span.log(name="HaulogyBot_Jira_Conf")
