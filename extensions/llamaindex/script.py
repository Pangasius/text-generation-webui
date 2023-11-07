import datetime
import traceback
from typing import List
import torch
from extensions.llamaindex.LlamaIndex import IndexEngine
from llama_index.prompts.default_prompts import DEFAULT_TREE_SUMMARIZE_PROMPT
from llama_index.schema import MetadataMode
import modules.shared as shared

from llama_index.schema import NodeWithScore

from modules.text_generation import generate_reply_HF, generate_reply_custom
from modules.logging_colors import logger

import wandb
from wandb.sdk.data_types.trace_tree import Trace

wandb.init(project="Haulogy-First-Test")

DATASET = "conf_custo_embed"


def setup():
    if shared.index is not None:
        print("Index already loaded!")

    shared.index = IndexEngine(index_name=DATASET).as_retriever(kg=False,
                                            fine_tune=False,
                                            build_index=False)


def get_meta_if_possible(nodes: List[NodeWithScore]):
    info = ""

    for node in nodes:

        if node.metadata is not None:
            if node.metadata.keys().__contains__("url") and node.metadata.keys().__contains__("title"):
                info += '[' + node.metadata['title'] + '](' + node.metadata['url'] + ')'
        else:
            # put a small extract instead
            info += 'No metadata found'
        if node.get_score() is not None:
            info += ' Similarity: ' + str(node.get_score())

        info += '\n'

    return info


def input_modifier(question: str, state: dict, is_chat: bool = False) -> str:
    state['last_question'] = question
    try:
        with torch.no_grad():
            resp = shared.index.retrieve(question)

            context = "\n\n".join([x.get_content(metadata_mode=MetadataMode.NONE)
                                    for x in resp])

            state['last_metadata'] = get_meta_if_possible(resp)

            question = DEFAULT_TREE_SUMMARIZE_PROMPT.format(context_str=context,
                                                            query_str=question)

    except Exception:
        traceback.print_exc()
    finally:
        return question


def output_modifier(string, state, is_chat=False):
    if state.keys().__contains__("last_metadata"):
        output = string + "\nSources : \n\n" + state["last_metadata"] + "\n\n"
        return output

    return string


def custom_generate_reply(question, original_question, seed, state, stopping_strings=None, is_chat=False):

    generate_func = None

    if shared.model_name == 'None' or shared.model is None:
        logger.error("No model is loaded! Select one in the Model tab.")
        yield ''
        return

    if shared.model.__class__.__name__ in ['LlamaCppModel', 'RWKVModel', 'ExllamaModel', 'Exllamav2Model', 'CtransformersModel']:
        generate_func = generate_reply_custom
    else:
        generate_func = generate_reply_HF

    start_time_ms = round(datetime.datetime.now().timestamp() * 1000)

    llm_span = Trace(
            name=shared.model_name,
            kind="llm",
            start_time_ms=start_time_ms,
            metadata={"error": "",
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": shared.model_name,
                    "dataset": DATASET,
                    "config": shared.model_config})
    ans = ""

    try:
        for ans in generate_func(question, original_question, seed, state, stopping_strings, is_chat):
            yield ans
    except Exception as e:
        traceback.print_exc()
        llm_span.metadata["error"] = str(e)
    finally:
        last_metadata = state['last_metadata'] if state.keys().__contains__("last_metadata") else ""

        last_question = state['last_question'] if state.keys().__contains__("last_question") else ""

        end_time_ms = round(datetime.datetime.now().timestamp() * 1000)
        llm_span.inputs = {"query": last_question, 
                           "input_documents": last_metadata,
                           "context" : question}
        llm_span.end_time_ms = end_time_ms
        llm_span.outputs = {"response": ans}
        llm_span.log(name="HauBot-test-CEO")

        #print("\n\033[91m" + "Start Of Produced Answer" + "\033[0m\n")

        #print("Question : " + question)
        #print("Answer : " + ans + "\n")

        #print("\033[91m" + "End Of Produced Answer" + "\033[0m\n")

        return
