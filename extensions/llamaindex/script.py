import traceback
from typing import List
import torch
from extensions.llamaindex.LlamaIndex import IndexEngine
from llama_index.prompts.default_prompts import DEFAULT_TREE_SUMMARIZE_PROMPT
from llama_index.schema import MetadataMode
import modules.shared as shared

from llama_index.schema import NodeWithScore


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
    try:
        if shared.index is None:
            shared.index = IndexEngine().as_retriever(kg=False,
                                                      fine_tune=False,
                                                      build_index=False,
                                                      index_name="conf_attach")

        with torch.no_grad():
            print("Llama Index is enabled")
            resp = shared.index.retrieve(question)

            context = "\n\n".join([x.get_content(metadata_mode=MetadataMode.NONE)
                                    for x in resp])

            state['last_metadata'] = get_meta_if_possible(resp)

            question = DEFAULT_TREE_SUMMARIZE_PROMPT.format(context_str=context,
                                                            query_str=question)

        print(question)

    except Exception:
        traceback.print_exc()
    finally:
        return question


def output_modifier(string, state, is_chat=False):
    if state.keys().__contains__("last_metadata"):
        output = string + "\nSources : \n\n" + state["last_metadata"] + "\n\n"
        return output
    else:
        return string
