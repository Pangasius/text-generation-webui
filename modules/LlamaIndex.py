# package: code/backend

from pathlib import Path
import glob
import regex as re
import time
import os
from typing import List

# from pdf2docx import Converter
from zipfile import ZipFile
import pytesseract

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
    get_response_synthesizer
)

from llama_index.llms import HuggingFaceLLM
from llama_index.graph_stores import NebulaGraphStore

from llama_index.schema import NodeWithScore
from llama_index.query_engine import RetrieverQueryEngine
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

from llama_index.embeddings.adapter_utils import (
    TwoLayerNN
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

adapter_model = TwoLayerNN(
    1024,  # input dimension
    1024,  # hidden dimension
    1024,  # output dimension
    bias=True,
    add_residual=True,
)

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


class IndexEngine():
    def __init__(self):
        print("Loading model...")

        self.llm = HuggingFaceLLM(model=shared.model,
                             tokenizer=shared.tokenizer,
                             max_new_tokens=shared.settings['max_new_tokens'],
                             context_window=shared.args.n_ctx)

        self.engine = None

        print("Model loaded")

    def get_service_context(self, embed_model="local:BAAI/bge-large-en-v1.5") :
        prompt_helper = PromptHelper(tokenizer=shared.tokenizer)

        service_context = ServiceContext.\
            from_defaults(llm=self.llm,
                          embed_model=embed_model,
                          prompt_helper=prompt_helper,
                          chunk_size=128,
                          context_window=min(128, shared.args.n_ctx))

        return service_context

    def parse_documents(self):
        print("Indexing documents...")

        t0 = time.time()

        print("Wikipedia...")

        unstructuredReader = download_loader("UnstructuredReader")
        unstructuredLoader = unstructuredReader()

        toBeProcessed = []

        documents = []

        print("Files...")

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

        print("Unstructured reading files...")
        for paths in glob.glob("./examples/**/*.*", recursive=True):
            if re.match(r".*/private/.*$", paths) or not re.match(r".*\.[a-z]{1,10}$", paths) or re.match(r".*(((I|i)con(s)?)|(\.graffle)|(MACOSX)|(\/out)|(\/bin))\/.*", paths):
                print("- ", paths)
                continue

            # Skip the produced document and all unsupported files
            if re.match(r".*\.((sql)|(json)|(xsd)|(css)|(xml)|(csv)|(png)|(jpg)|(pdf)|(xlsx))", paths):
                if re.match(r".*\.((sql)|(xsd))", paths):
                    toBeProcessed.append(paths)
                print("- ", paths)
                continue

            try:
                documents += unstructuredLoader.load_data(file=Path(paths))
                print("+ ", paths)
            except Exception as e:
                print("- ", paths)
                print("Error loading file ", paths, ":", e)
                continue

        reader = SimpleDirectoryReader(input_files=toBeProcessed)
        documents += reader.load_data()

        print("Filtering documents...")
        initial_size = sum(list(map(lambda x: len(x.text), documents)))
        for document in documents:
            if (document.text.__contains__("Problem authenticating")
                    or document.text.__contains__("File Generator")
                    or document.text.__contains__("MIT P2P : Status")
                    or documents.__contains__("Defects – digest")):
                documents.remove(document)
                continue

            # remove all emails, ip
            document.text = re.sub(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)|(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", "", document.text)

            # remove all phone numbers
            document.text = re.sub(r"(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})", "", document.text)

            # remove all lines line breaks bigger than two \n
            document.text = re.sub(r"\n{2,}", "\n", document.text)

            if document.text == "":
                documents.remove(document)
                continue

        t1 = time.time()

        print("Documents loaded in", t1-t0, "seconds")

        print("Initial size:", initial_size)
        print("Final size:", sum(list(map(lambda x: len(x.text), documents))))

        # save all documents to a file
        with open("examples/private/documents.txt", "w") as f:
            for document in documents:
                f.write(document.text)
                f.write("\n\n\n")

        print("Indexing...")

        return documents

    def index_documents(self, documents: List[Document],
                        embed_model="local:BAAI/bge-large-en-v1.5"):

        service_context = self.get_service_context(embed_model=embed_model)

        graph_store = NebulaGraphStore(
            space_name=space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
        )
        storage_context = StorageContext.from_defaults(graph_store=graph_store)

        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            max_triplets_per_chunk=15,
            space_name=space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
            include_embeddings=True,
            service_context=service_context,
            show_progress=True,
        )

        vector_index = VectorStoreIndex.from_documents(
            documents=documents,
            service_context=service_context)

        # writing to disk

        vector_index.storage_context.persist(persist_dir="index/vector")
        kg_index.storage_context.persist(persist_dir="index/kg")

        print("Indexing complete")

        return service_context

    def fine_tune(self, docs: List[Document],
                  out_path="models/embedder/mig6-v1"):
        # generate a finetuning qa dataset
        print("Generating finetuning dataset...")

        base_embed_model = resolve_embed_model("local:BAAI/bge-large-en-v1.5")

        if os.path.exists(out_path):
            return LinearAdapterEmbeddingModel(base_embed_model,
                                               out_path)

        dataset_path = "examples/private/dataset.json"

        if not os.path.exists(dataset_path):
            parser = SimpleNodeParser.from_defaults()
            nodes = parser.get_nodes_from_documents(docs, show_progress=True)

            dataset = generate_qa_embedding_pairs(nodes,
                                                  llm=self.llm,
                                                  num_questions_per_chunk=1)

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
            epochs=1,
            batch_size=12,
            verbose=True,
            # optimizer_class=torch.optim.SGD,
            # optimizer_params={"lr": 0.01}
        )

        finetune_engine.finetune()

        return LinearAdapterEmbeddingModel(base_embed_model,
                                           out_path)

    def as_query_engine(self, streaming=True, embed_model="local:BAAI/bge-large-en-v1.5"):

        # count the files in index and if there are none, index the files
        if (len(list(Path("index/vector").glob("*"))) == 0
                or len(list(Path("index/kg").glob("*"))) == 0):
            documents = self.parse_documents()
            embed_model = self.fine_tune(docs=documents)
            service_context = self.index_documents(documents=documents,
                                                   embed_model=embed_model)
        else:
            service_context = self.get_service_context(embed_model=embed_model)

        print("Loading engine...")

        storage_context = StorageContext.from_defaults(persist_dir="index/vector")
        vector_index = load_index_from_storage(storage_context,
                                               service_context=service_context)

        storage_context = StorageContext.from_defaults(persist_dir="index/kg")
        kg_index = load_index_from_storage(storage_context,
                                           service_context=service_context)

        # create custom retriever
        vector_retriever = VectorIndexRetriever(index=vector_index)
        kg_retriever = KnowledgeGraphRAGRetriever(
            index=kg_index,
            retriever_mode="keyword",
            include_text=False,
            service_context=service_context,
            storage_context=storage_context,
            verbose=True,
            graph_traversal_depth=3
        )
        custom_retriever = CustomRetriever(vector_retriever, kg_retriever)

        # create response synthesizer
        response_synthesizer = get_response_synthesizer(
            service_context=service_context,
            response_mode="tree_summarize",
            streaming=streaming
        )

        self.engine = RetrieverQueryEngine(
            retriever=custom_retriever,
            response_synthesizer=response_synthesizer,
        )

        print("Engine loaded")

        return self

    def querier(self, streaming=True):
        if self.engine is None:
            self.as_query_engine(streaming=streaming)

        return self
