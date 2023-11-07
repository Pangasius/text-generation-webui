# package: code/backend

from pathlib import Path
import glob
import regex as re

import torch

import os
from typing import Dict, List, Optional

import json

# from pdf2docx import Converter
from zipfile import ZipFile
import pytesseract

from pdf2docx import Converter

from llama_index import (
    Document,
    download_loader,
    SimpleDirectoryReader,
    ServiceContext,
    PromptHelper,
    QueryBundle,
)

from llama_index.llms import HuggingFaceLLM

from llama_index.schema import NodeWithScore, BaseNode

from llama_index.readers.base import BaseReader

from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KnowledgeGraphRAGRetriever
)

from llama_index.node_parser import SimpleNodeParser

from llama_index.finetuning import EmbeddingAdapterFinetuneEngine

from llama_index.embeddings import (
    resolve_embed_model,
    LinearAdapterEmbeddingModel
)

from llama_index.finetuning.embeddings.common import (
    generate_qa_embedding_pairs,
    EmbeddingQAFinetuneDataset
)

import modules.shared as shared

from extensions.llamaindex.connections import connect_NebulaGraph, connect_PostgreSQL

from tqdm import tqdm

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


class CustomRetriever(BaseRetriever):
    """
    Custom retriever that performs
    both Vector search and Knowledge Graph search.
    """

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KnowledgeGraphRAGRetriever,
        mode: str = "OR",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        kg_ids = {n.node.node_id for n in kg_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in kg_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


class CustomReader(BaseReader):
    def load_data(self, file: Path, extra_info: Optional[Dict]) -> List[Document]:
        # The documents to be read are python dictionaries in string format
        # The keys are "content", "attachment", "url", the url will be the metadata

        with open(file, "r", encoding="utf_8") as f:
            text_dict = f.read()

        try:
            j = json.loads(text_dict)

            content = j["content"]
            metadata = j["metadata"]

            # Remove attachments for now
            if "attachments" in metadata:
                metadata.pop("attachments")

            doc = Document(text=content, metadata=metadata)

            return [doc]
        except Exception as e:
            print("JSON not accepted", e)
            return []


class IndexEngine():
    def __init__(self, index_name: str):
        print("Loading model...")
        self.retriever = None

        print("Model loaded")

        llm = HuggingFaceLLM(model=shared.model,
                             tokenizer=shared.tokenizer,
                             max_new_tokens=shared.settings['max_new_tokens'],
                             context_window=shared.args.n_ctx)

        print("Max new tokens:", shared.settings['max_new_tokens'])
        print("Context window:", shared.args.n_ctx)

        self.llm = llm
        self.index_name = index_name

    def get_service_context(self):
        prompt_helper = PromptHelper(tokenizer=shared.tokenizer)

        self.embed_model = resolve_embed_model(self.embed_model)

        self.embed_model.embed_batch_size = 1

        service_context = ServiceContext.\
            from_defaults(llm=self.llm,
                          embed_model=self.embed_model,
                          prompt_helper=prompt_helper,
                          chunk_size=1024,
                          context_window=min(512, shared.args.n_ctx))

        return service_context

    @staticmethod
    def unzip_files():
        print("Unzipping and converting files...")
        # first unzip all zip files and turn all pdf files into docx files
        for paths in glob.glob("./examples/**/*", recursive=True):
            try:
                if re.match(r".*\.zip$", paths):
                    print("Found zip file", paths)
                    with ZipFile(paths, 'r') as zipObj:
                        zipObj.extractall("/".join(paths.split("/")[:-1]))
                    os.remove(paths)
                    print("Extracted zip file", paths)
                elif re.match(r".*\.pdf$", paths):
                    print("Found pdf file", paths)
                    cv = Converter(paths)
                    cv.convert(paths.replace(".pdf", ".docx"))
                    os.remove(paths)
                    print("Converted pdf file", paths)
                else:
                    continue
            except Exception as e:
                print("Error extracting zip file", paths, ":", e)
                continue

    def read_documents(self, documents: List[Document]):
        print("Unstructured reading files...")

        unstructuredReader = download_loader("UnstructuredReader")
        unstructuredLoader = unstructuredReader()

        toBeProcessed = []

        for paths in tqdm(glob.glob("./examples/" + self.index_name + "/**/*.*", recursive=True)):
            # Skip the produced document and all unsupported files
            if re.match(r".*\.((sql)|(json)|(xsd)|(css)|(xml)|(csv)|(png)|(jpg)|(xlsx))", paths):
                if re.match(r".*\.((json))", paths):
                    toBeProcessed.append(paths)
                continue

            try:
                documents += unstructuredLoader.load_data(file=Path(paths))
                documents[-1].metadata["url"] = paths
                documents[-1].metadata["title"] = paths.split("/")[-1]
            except Exception as e:
                continue

        reader = SimpleDirectoryReader(input_files=toBeProcessed,
                                       encoding="utf_8",
                                       file_extractor={".json": CustomReader()})
        documents += reader.load_data()

    def parse_documents(self):
        print("Indexing documents...")

        # self.unzip_files()

        documents = []
        self.read_documents(documents)

        print("Size:", sum(list(map(lambda x: len(x.text), documents))))

        # transform documents into nodes

        node_parser = SimpleNodeParser.from_defaults(chunk_size=1024,
                                                     chunk_overlap=128)

        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

        # print the average chunk size
        print("Average chunk size:", sum(list(map(lambda x: len(x.text), nodes))) / len(nodes))

        # print the maximum chunk size
        print("Maximum chunk size:", max(list(map(lambda x: len(x.text), nodes))))

        return nodes

    def save_nodes(self, nodes: List[BaseNode], kg=False):

        service_context = self.get_service_context()

        print("Indexing into a vector store...")
        vector_store = connect_PostgreSQL(self.index_name, service_context=service_context)
        vector_store.build_index_from_nodes(nodes=nodes)

        if kg:
            print("Indexing into a knowledge graph...")
            kg_store = connect_NebulaGraph(service_context=service_context)
            kg_store.build_index_from_nodes(nodes=nodes)

    def get_retrievers(self, kg=True):

        print("Indexing documents...")

        service_context = self.get_service_context()

        vector_index = connect_PostgreSQL(self.index_name, service_context=service_context)

        vec_retriever = vector_index.as_retriever(
            similarity_top_k=3,
            # vector_store_query_mode="mmr"
        )

        kg_retriever = None
        if kg:
            # Graph store params
            graph_store = connect_NebulaGraph(service_context=service_context)

            kg_retriever = graph_store.as_retriever()

        print("Indexing complete")
        return vec_retriever, kg_retriever

    def fine_tune(self, nodes: List[BaseNode] | None,
                  out_path: str):

        base_embed_model = resolve_embed_model("local:BAAI/bge-large-en-v1.5")
        base_embed_model.embed_batch_size = 1

        if os.path.exists(out_path):
            print("Loading already trained finetuned model...")
            return LinearAdapterEmbeddingModel(base_embed_model,
                                               out_path)

        dataset_path = out_path + ".json"

        print("Dataset path:", dataset_path)

        if not os.path.exists(dataset_path) and nodes is not None:
            print("Generating QA dataset...")
            dataset = generate_qa_embedding_pairs(nodes,
                                                  llm=self.llm,
                                                  num_questions_per_chunk=2)

            dataset.save_json(dataset_path)
        elif nodes is None:
            dataset = EmbeddingQAFinetuneDataset.from_json(dataset_path)
        else:
            raise Exception("Dataset not found but trying to load it")

        print("Finetuning...")
        finetune_engine = EmbeddingAdapterFinetuneEngine(
            dataset,
            base_embed_model,
            model_output_path=out_path,
            # bias=True,
            dim=1024,
            epochs=5,
            batch_size=1,
            verbose=True,
            optimizer_class=torch.optim.SGD,
            optimizer_params={"lr": 0.0001}
        )

        finetune_engine.finetune()

        self.embed_model = LinearAdapterEmbeddingModel(base_embed_model,
                                           out_path)

    def custom_retriever(self, embed_model):
        if embed_model.startswith("custom:"):
            self.embed_model = LinearAdapterEmbeddingModel(
                resolve_embed_model("local:BAAI/bge-large-en-v1.5"),
                embed_model[7:]
            )

    def as_retriever(self, embed_model="local:BAAI/bge-large-en-v1.5", kg=True, fine_tune=True, build_index=True, ):
        self.custom_retriever(embed_model)
        
        print("Loading engine...")
        if build_index:
            nodes = self.parse_documents()
            self.save_nodes(nodes, kg=kg)
        else:
            nodes = None

        if fine_tune:
            self.fine_tune(nodes=nodes, out_path="models/embedder/" + self.index_name)

        vec_retriever, kg_retriever = self.get_retrievers(kg=kg)

        self.retriever = vec_retriever

        if kg_retriever is not None:
            self.retriever = CustomRetriever(vec_retriever, kg_retriever)

        print("Engine loaded")
        return self.retriever
