import traceback
import torch
from extensions.llamaindex.LlamaIndex import IndexEngine
from llama_index.prompts.default_prompts import DEFAULT_TREE_SUMMARIZE_PROMPT
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

            question = DEFAULT_TREE_SUMMARIZE_PROMPT.format(context_str=resp,
                                                            query_str=question)

        print(question)

    except Exception:
        traceback.print_exc()
    finally:
        return question
