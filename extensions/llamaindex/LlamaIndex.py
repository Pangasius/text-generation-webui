# package: code/backend

from pathlib import Path
import glob
import regex as re

import os
from typing import Dict, List, Optional

import json

# from pdf2docx import Converter
from zipfile import ZipFile
import pytesseract

from pdf2docx import Converter

from llama_index import (
    Document,
    StorageContext,
    VectorStoreIndex,
    download_loader,
    load_index_from_storage,
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    ServiceContext,
    PromptHelper,
    QueryBundle,
)

from llama_index.llms import HuggingFaceLLM
from llama_index.graph_stores import NebulaGraphStore

from llama_index.schema import NodeWithScore

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

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

os.environ["NEBULA_USER"] = "root"
os.environ["NEBULA_PASSWORD"] = "nebula"
os.environ[
    "NEBULA_ADDRESS"
] = "127.0.0.1:9669"  # assumed we have NebulaGraph 3.5.0 or newer installed locally

# Assume that the graph has already been created
# Create a NebulaGraph cluster with:
# Option 0: `curl -fsSL nebula-up.siwei.io/install.sh | bash`
# Option 1: NebulaGraph Docker Extension https://hub.docker.com/extensions/weygu/nebulagraph-dd-ext
# and that the graph space is called "llamaindex"
# If not, create it with the following commands from NebulaGraph's console:
# CREATE SPACE llamaindex(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);
# :sleep 10;
# USE llamaindex;
# CREATE TAG entity(name string);
# CREATE EDGE relationship(relationship string);
# CREATE TAG INDEX entity_index ON entity(name(256));

space_name = "llamaindex"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
]  # default, could be omit if create from an empty kg
tags = ["entity"]  # default, could be omit if create from an empty kg


def connect_PostgreSQL(index_name: str):
    from llama_index.vector_stores import PGVectorStore

    vector_store = PGVectorStore.from_params(
        database="llamaindex",
        host="localhost",
        password="index",
        port="5432",
        user="llama",
        table_name=index_name,
        embed_dim=1024,  # embedding dimension
    )

    return vector_store


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
    def __init__(self):
        print("Loading model...")
        self.retriever = None

        print("Model loaded")

    def get_service_context(self, embed_model="local:BAAI/bge-large-en-v1.5"):
        prompt_helper = PromptHelper(tokenizer=shared.tokenizer)

        llm = HuggingFaceLLM(model=shared.model,
                             tokenizer=shared.tokenizer,
                             max_new_tokens=shared.settings['max_new_tokens'],
                             context_window=shared.args.n_ctx)

        embed_model = resolve_embed_model("local:BAAI/bge-large-en-v1.5")

        embed_model.embed_batch_size = 1

        service_context = ServiceContext.\
            from_defaults(llm=llm,
                          embed_model=embed_model,
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

    @staticmethod
    def read_documents(documents: List[Document], index_name: str):
        print("Unstructured reading files...")

        unstructuredReader = download_loader("UnstructuredReader")
        unstructuredLoader = unstructuredReader()

        toBeProcessed = []

        for paths in glob.glob("./examples/" + index_name + "/**/*.*", recursive=True):
            # Skip the produced document and all unsupported files
            if re.match(r".*\.((sql)|(json)|(xsd)|(css)|(xml)|(csv)|(png)|(jpg)|(xlsx))", paths):
                if re.match(r".*\.((json))", paths):
                    print("+ ", paths)
                    toBeProcessed.append(paths)
                else:
                    print("- ", paths)
                continue

            try:
                documents += unstructuredLoader.load_data(file=Path(paths))
                documents[-1].metadata["url"] = paths
                documents[-1].metadata["title"] = paths.split("/")[-1]
                print("+ ", paths)
            except Exception as e:
                print("- ", paths)
                print("Error loading file ", paths, ":", e)
                continue

        reader = SimpleDirectoryReader(input_files=toBeProcessed,
                                       encoding="utf_8",
                                       file_extractor={".json": CustomReader()})
        documents += reader.load_data()

    @staticmethod
    def filter_out(documents: List[Document]):
        print("Filtering documents...")

        for document in documents:
            if (document.text.__contains__("Problem authenticating")
                    or document.text.__contains__("File Generator")
                    or document.text.__contains__("MIT P2P : Status")
                    or documents.__contains__("Defects – digest")):
                documents.remove(document)
                continue

            # remove all lines line breaks bigger than two \n
            document.text = re.sub(r"\n{2,}", "\n", document.text)

            if document.text == "":
                documents.remove(document)
                continue

    def parse_documents(self, index_name: str):
        print("Indexing documents...")

        # self.unzip_files()

        documents = []
        self.read_documents(documents, index_name=index_name)

        initial_size = sum(list(map(lambda x: len(x.text), documents)))

        self.filter_out(documents)

        print("Initial size:", initial_size)
        print("Final size:", sum(list(map(lambda x: len(x.text), documents))))

        # save all documents to a file
        with open("examples/private/documents.txt", "w") as f:
            for document in documents:
                f.write(document.text)
                f.write("\n\n\n")

        return documents

    def get_retrievers(self, documents,
                       index_name : str,
                       embed_model="local:BAAI/bge-large-en-v1.5", kg=True):

        print("Indexing documents...")
        service_context = self.get_service_context(embed_model=embed_model)

        # Indexing into a vector store
        if documents is not None:
            vector_index = VectorStoreIndex.from_documents(
                documents=documents,
                service_context=service_context,
                storage_context=StorageContext.from_defaults(vector_store=connect_PostgreSQL(index_name)),
                show_progress=True
            )
        else:
            vector_index = VectorStoreIndex.from_vector_store(vector_store=connect_PostgreSQL(index_name), service_context=service_context)

        vec_retriever = vector_index.as_retriever(
            similarity_top_k=3,
            # vector_store_query_mode="mmr"
        )
        kg_retriever = None

        if kg:
            # Graph store params
            graph_store = NebulaGraphStore(
                space_name=space_name,
                edge_types=edge_types,
                rel_prop_names=rel_prop_names,
                tags=tags,
            )
            storage_context = StorageContext.from_defaults(graph_store=graph_store)

            # Indexing into a knowledge graph
            if documents is not None:
                kg_index = KnowledgeGraphIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    max_triplets_per_chunk=30,
                    include_embeddings=True,
                    service_context=service_context,
                    show_progress=True,
                )

                kg_retriever = kg_index.as_retriever(
                    similarity_top_k=3,
                    graph_store_query_depth=4
                )

            else:
                kg_retriever = KnowledgeGraphRAGRetriever(
                    retriever_mode="keyword",
                    include_text=True,
                    service_context=service_context,
                    storage_context=storage_context,
                    verbose=True,
                    graph_traversal_depth=3
                )

        print("Indexing complete")
        return vec_retriever, kg_retriever

    def fine_tune(self, docs: List[Document],
                  out_path: str):
        # generate a finetuning qa dataset
        print("Generating finetuning dataset...")

        base_embed_model = resolve_embed_model("local:BAAI/bge-large-en-v1.5")
        base_embed_model.embed_batch_size = 1

        if os.path.exists(out_path):
            return LinearAdapterEmbeddingModel(base_embed_model,
                                               out_path)

        dataset_path = "examples/private/dataset.json"

        if not os.path.exists(dataset_path):
            parser = SimpleNodeParser.from_defaults()
            nodes = parser.get_nodes_from_documents(docs, show_progress=True)

            dataset = generate_qa_embedding_pairs(nodes,
                                                  llm=self.llm,
                                                  num_questions_per_chunk=5)

            dataset.save_json(dataset_path)
        else:
            dataset = EmbeddingQAFinetuneDataset.from_json(dataset_path)

        # [Optional] Load
        # dataset = EmbeddingQAFinetuneDataset.from_json("dataset.json")

        finetune_engine = EmbeddingAdapterFinetuneEngine(
            dataset,
            base_embed_model,
            model_output_path=out_path,
            # bias=True,
            dim=1024,
            epochs=4,
            batch_size=12,
            verbose=True,
            # optimizer_class=torch.optim.SGD,
            # optimizer_params={"lr": 0.01}
        )

        finetune_engine.finetune()

        return LinearAdapterEmbeddingModel(base_embed_model,
                                           out_path)

    def custom_retriever(self, embed_model="local:BAAI/bge-large-en-v1.5"):
        if embed_model.startswith("custom:"):
            embed_model = LinearAdapterEmbeddingModel(
                resolve_embed_model("local:BAAI/bge-large-en-v1.5"),
                embed_model[7:]
            )

        return embed_model

    def as_retriever(self, index_name: str, embed_model="local:BAAI/bge-large-en-v1.5", kg=True, fine_tune=True, build_index=True, ):
        print("Loading engine...")
        if build_index:
            documents = self.parse_documents(index_name=index_name)
            if fine_tune:
                embed_model = self.fine_tune(docs=documents, out_path="models/embedder/" + index_name)
        else:
            documents = None

        vec_retriever, kg_retriever = self.get_retrievers(documents=documents,
                                                   embed_model=embed_model, kg=kg, index_name=index_name)

        self.retriever = vec_retriever

        if kg_retriever is not None:
            self.retriever = CustomRetriever(vec_retriever, kg_retriever)

        print("Engine loaded")
        return self.retriever
