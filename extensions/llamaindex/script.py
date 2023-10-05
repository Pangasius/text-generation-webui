import traceback
import torch
from extensions.llamaindex.LlamaIndex import IndexEngine
from llama_index.prompts.default_prompts import DEFAULT_TREE_SUMMARIZE_PROMPT
from llama_index.schema import MetadataMode
import modules.shared as shared

def input_modifier(question: str, state: dict, is_chat: bool = False) -> str:
    try:
        if shared.settings["use_llama_index"] is False:
            print("Llama Index is disabled")
            return question

        if shared.index is None:
            shared.index = IndexEngine().as_retriever(kg=False, fine_tune=False)

        with torch.no_grad():
            print("Llama Index is enabled")
            resp = shared.index.retrieve(question)

            context = "\n\n".join([x.get_content(metadata_mode=MetadataMode.ALL)
                       for x in resp])

            state["last_context"] = context

            question = DEFAULT_TREE_SUMMARIZE_PROMPT.format(context_str=context,
                                                            query_str=question)

        print(question)

    except Exception:
        traceback.print_exc()
    finally:
        return question


def output_modifier(string, state, is_chat=False):
    if state["last_context"] is not None:
        output = "Using Llama Index\n\n" + state["last_context"] + "\n\n" + string
        return output
    else:
        return string
