"""Module allowing to query and construct a LlamaIndex index."""
import json
from pathlib import Path
import glob
import os
from typing import List, Sequence

import regex as re
import pytesseract
from tqdm import tqdm

from llama_index import (
    download_loader,
    SimpleDirectoryReader,
    ServiceContext,
    PromptHelper,
    QueryBundle,
)

from llama_index.llms import HuggingFaceLLM
from llama_index.schema import NodeWithScore, BaseNode
from llama_index.retrievers import (
    BaseRetriever
)
from llama_index.node_parser import SentenceSplitter
from llama_index.embeddings import (
    resolve_embed_model,
    AdapterEmbeddingModel
)
from llama_index.embeddings.adapter_utils import TwoLayerNN
from llama_index.finetuning import EmbeddingAdapterFinetuneEngine
from llama_index.finetuning.embeddings.common import (
    generate_qa_embedding_pairs,
    EmbeddingQAFinetuneDataset
)
from llama_index.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    EntityExtractor
)

from llama_index.ingestion import IngestionPipeline

from modules import shared

from extensions.llama_index.tools.reader import ConfluenceReader, JiraReader
from extensions.llama_index.tools.connections import connect_store

from transformers import AutoConfig

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


VECTOR_STORE = "postgresql"
GRAPH_STORE = "nebulagraph"


class CustomRetriever(BaseRetriever):
    """
    Custom retriever that performs
    both Vector search and Knowledge Graph search.
    """

    def __init__(
        self,
        retriever_1: BaseRetriever,
        retriever_2: BaseRetriever,
        mode: str = "OR",
    ) -> None:
        """Init params."""

        self.retriever_1 = retriever_1
        self.retriever_2 = retriever_2
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        nodes_1 = self.retriever_1.retrieve(query_bundle)
        nodes_2 = self.retriever_2.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in nodes_1}
        kg_ids = {n.node.node_id for n in nodes_2}

        combined_dict = {n.node.node_id: n for n in nodes_1}
        combined_dict.update({n.node.node_id: n for n in nodes_2})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


class IndexEngine():
    """
    This class is used to index documents and retrieve them.
    It will interact with a vector store and can also interact with a knowledge graph.
    """
    def __init__(self, embed_model="local:BAAI/bge-large-en-v1.5"):
        print("Loading model...")

        path_to_model = Path(f'{shared.args.model_dir}/{shared.model_name}')
        path_to_tokenizer = Path(f'{shared.args.model_dir}/{shared.model_name}' + "/tokenizer_config.json")

        config = AutoConfig.from_pretrained(path_to_model)
        tokenizer_config = json.load(open(path_to_tokenizer, "r"))

        llm = HuggingFaceLLM(model=shared.model,
                             model_name=shared.model_name,
                            tokenizer=shared.tokenizer,
                            tokenizer_name=shared.model_name,
                            max_new_tokens=shared.settings['max_new_tokens'],
                            context_window=shared.args.n_ctx,
                            model_kwargs=config.to_dict(),
                            tokenizer_kwargs=tokenizer_config)

        print("Max new tokens:", shared.settings['max_new_tokens'])
        print("Context window:", shared.args.n_ctx)

        self.llm = llm

        self.index_name = None
        self.dataset = None

        self._load_embedding_model(embed_model)

        self._load_service_context()

    def _load_embedding_model(self, embed_model: str):
        """
        Loads a custom retriever if the embed_model name is
        of the form "custom:embed_model_name".

        Args:
            embed_model (str): The name of the embedding model
        """
        if isinstance(embed_model, str) and embed_model.startswith("custom:"):
            self.embed_model = AdapterEmbeddingModel(
                resolve_embed_model(self.embed_model),
                embed_model[7:],
                device="cuda:0",
                adapter_cls=TwoLayerNN
            )
        else:
            self.embed_model = resolve_embed_model(embed_model)

        self.embed_model.embed_batch_size = 4

    def _load_service_context(self):
        """
        Loads the service context as self.service_context.

        Returns:
            ServiceContext: The service context
        """

        prompt_helper = PromptHelper(tokenizer=shared.tokenizer)

        self.service_context = ServiceContext.\
            from_defaults(llm=self.llm,
                        embed_model=self.embed_model,
                        prompt_helper=prompt_helper,
                        chunk_size=1024,
                        context_window=shared.args.n_ctx)

    def read_documents(self):
        """
        Read the documents from the examples folder, subfolder named "index_name".
        If JSON are found they are assumed to be conforming to the definition of the CustomReader.

        Args:
            documents (List[Document]): _description_
        """

        unstructured_reader = download_loader("UnstructuredReader")
        unstructured_loader = unstructured_reader()

        to_be_processed = []
        documents = []

        if self.dataset is None:
            raise Exception("The dataset has not been set.")

        print("Reading documents...")
        for paths in tqdm(glob.glob("./examples/" + self.dataset + "/**/*.*", recursive=True)):
            # Skip the produced document and all unsupported files
            if re.match(r".*\.((sql)|([cj]?json)|(xsd)|(css)|(xml)|(csv)|(png)|(jpg)|(xlsx))", paths):
                if re.match(r".*\.(([cj]json))", paths):
                    to_be_processed.append(str(paths))
                continue

            documents += unstructured_loader.load_data(file=Path(paths))

            # Creates a fake url pointing to the saved file
            documents[-1].metadata["url"] = paths
            documents[-1].metadata["title"] = paths.split("/")[-1]

        if len(to_be_processed) > 0:
            print("Reading custom JSON files...")
            reader = SimpleDirectoryReader(input_files=to_be_processed,
                                        encoding="utf_8",
                                        file_extractor={".jjson": JiraReader(),
                                                        ".cjson": ConfluenceReader()})
            documents += reader.load_data()

        print("Size of all documents :", sum(list(map(lambda x: len(x.text), documents))))

        return documents

    def parse_documents(self):
        """
        Read the documents and parse them into nodes.

        Returns:
            List[BaseNode]: The list of nodes
        """

        documents = self.read_documents()

        # transform documents into nodes
        node_parser = SentenceSplitter.from_defaults(chunk_size=1024,
                                                     chunk_overlap=128)

        extractors = [
            TitleExtractor(nodes=5, llm=self.llm),
            QuestionsAnsweredExtractor(questions=3, llm=self.llm),
            EntityExtractor(prediction_threshold=0.6, device="cuda:0"),
            SummaryExtractor(summaries=["prev", "self"], llm=self.llm),
            KeywordExtractor(keywords=10, llm=self.llm),
        ]

        transformations = [node_parser, extractors[2]]  # for now we only use the entity extractor

        pipeline = IngestionPipeline(transformations=transformations)

        nodes = pipeline.run(documents=documents, in_place=True, show_progress=True)

        # print the average size
        print("Average node size:", sum(list(map(lambda x: len(x.text), nodes))) / len(nodes))

        # print the maximum size
        print("Maximum node size:", max(list(map(lambda x: len(x.text), nodes))))

        return nodes

    def save_nodes(self, nodes: Sequence[BaseNode], kg=False):
        """
        Saves the nodes into a vector store and a optionally knowledge graph.

        Args:
            nodes (Sequence[BaseNode]): The sequence of nodes to be saved
            kg (bool, optional): Whether to save the nodes into a knowledge graph.
                                 Defaults to False.
        """

        if self.index_name is None:
            raise Exception("The index name has not been set.")

        print("Indexing into a vector store...")
        vector_store = connect_store(VECTOR_STORE, self.index_name, service_context=self.service_context)
        vector_store.build_index_from_nodes(nodes=nodes)

        if kg:
            print("Indexing into a knowledge graph...")
            kg_store = connect_store(GRAPH_STORE, "llamaindex", service_context=self.service_context)
            kg_store.build_index_from_nodes(nodes=nodes)

    def get_retrievers(self, kg=True):
        """
        Get the retrievers for the vector store and optionally the knowledge graph.

        Args:
            kg (bool, optional): Whether to get the retriever for the knowledge graph.
            Defaults to True.

        Returns:
            VectorIndexRetriever, KnowledgeGraphRAGRetriever: The retrievers
        """

        print("Getting vector store...")

        if self.index_name is None:
            raise Exception("The index name has not been set.")

        # vector_index = connect_elastic(self.index_name, service_context=self.service_context)
        vector_index = connect_store(VECTOR_STORE, self.index_name, service_context=self.service_context)

        vec_retriever = vector_index.as_retriever(
            similarity_top_k=3,
            # vector_store_query_mode="mmr"
        )

        kg_retriever = None
        if kg:
            print("Getting knowledge graph...")
            # Graph store params
            graph_store = connect_store(GRAPH_STORE, "llamaindex", service_context=self.service_context)

            kg_retriever = graph_store.as_retriever()

        return vec_retriever, kg_retriever

    def fine_tune(self, nodes: Sequence[BaseNode] | None,
                  out_path: str):
        """
        Creates a fine_tuned embedding model based on the sequence of nodes provided.
        If the sequence of nodes is None or the name of the model already exists,
        the model is loaded instead.

        Args:
            nodes (Sequence[BaseNode] | None): The sequence of nodes to be used for fine-tuning
            out_path (str): The path to save the model to

        Raises:
            FileNotFoundError: If the dataset is not found and nodes is None
        """

        # TODO: Remove hardcoded "cuda:0" as the device in resolve_embed_model
        base_embed_model = resolve_embed_model("local:BAAI/bge-large-en-v1.5")

        if os.path.exists(out_path):
            print("Loading already trained finetuned model...")
            return AdapterEmbeddingModel(base_embed_model,
                                               out_path, device="cuda:0", adapter_cls=TwoLayerNN)

        dataset_path = out_path + ".json"

        print("Dataset path:", dataset_path)

        if not os.path.exists(dataset_path) and nodes is not None:
            print("Generating QA dataset...")
            dataset = generate_qa_embedding_pairs(list(nodes),
                                                  llm=self.llm,
                                                  num_questions_per_chunk=2)

            dataset.save_json(dataset_path)
        elif nodes is None:
            dataset = EmbeddingQAFinetuneDataset.from_json(dataset_path)
        else:
            raise FileNotFoundError("Dataset not found and nodes is None")

        print("Finetuning...")
        finetune_engine = EmbeddingAdapterFinetuneEngine(
            dataset,
            base_embed_model,
            model_output_path=out_path,
            epochs=10,
            batch_size=500,
            adapter_model=TwoLayerNN(1024, 1024, 1024),
            verbose=True,
            optimizer_params={"lr": 0.0001},
            device="cuda:0"
        )

        finetune_engine.finetune()

        self.embed_model = finetune_engine.get_finetuned_model(adapter_cls=TwoLayerNN, device="cuda:0")

    def as_retriever(self, dataset: str, index_name: str,
                     kg=False, fine_tune=False,
                     build_index=False):
        """
        Creates the retriever for the index.

        Args:
            kg (bool, optional): Wether to include a graph store.
                                 Defaults to False.
            fine_tune (bool, optional): Wether to create / use a fined-tuned model.
                                        Defaults to False.
            build_index (bool, optional): Wether to process documents under example
                                          and create a corresponding index.
                                          Defaults to False.

        Returns:
            BaseRetriever: The retriever
        """

        self.dataset = dataset
        self.index_name = index_name

        print("Loading engine...")
        if build_index:
            nodes = self.parse_documents()
            self.save_nodes(nodes, kg=kg)
        else:
            nodes = None

        if fine_tune:
            self.fine_tune(nodes=nodes, out_path="models/embedder/" + self.index_name)

        vec_retriever, kg_retriever = self.get_retrievers(kg=kg)

        retriever = vec_retriever

        if kg_retriever is not None:
            retriever = CustomRetriever(vec_retriever, kg_retriever)

        print("Engine loaded")
        return retriever
