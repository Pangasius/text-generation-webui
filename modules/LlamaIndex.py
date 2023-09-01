# package: code/backend

from llama_index import StorageContext, VectorStoreIndex, download_loader, load_index_from_storage

from llama_index.indices.service_context import ServiceContext
from llama_index.indices.prompt_helper import PromptHelper

from pathlib import Path
import glob
import regex as re

from llama_index.llms import HuggingFaceLLM

from pdf2docx import Converter

# package: code/backend

import modules.shared as shared

from llama_index import (
    KnowledgeGraphIndex,
    ServiceContext, 
)

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

from compress import Compressor

import time

from zipfile import ZipFile

#NEBULA QUERY GRAPH

from llama_index.graph_stores import NebulaGraphStore

import os
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

from llama_index import QueryBundle

# import NodeWithScore
from llama_index.schema import NodeWithScore

# Retrievers
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever

from typing import List

from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KGTableRetriever,
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
        prompt_helper = PromptHelper(tokenizer=shared.tokenizer)
        
        print("Loading model...")
        
        llm = HuggingFaceLLM(model=shared.model, tokenizer=shared.tokenizer, max_new_tokens=shared.settings['max_new_tokens'])
        embedder = "local" #Embedding()
        self.service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedder, prompt_helper=prompt_helper)
        
        self.engine = None
        
        print("Model loaded")
        
    def index_files(self):
        print("Indexing documents...")
        
        t0 = time.time()
        
        print("Wikipedia...")

        WikipediaReader = download_loader("WikipediaReader")
        loader = WikipediaReader()
        documents = loader.load_data(pages=["Électricité_de_France", "France", "Electricity pricing"])
        
        UnstructuredReader = download_loader("UnstructuredReader")
        loader = UnstructuredReader()
        
        print("Files...")
        
        print("Unzipping and converting files...")
        #first unzip all zip files and turn all pdf files into docx files
        for paths in glob.glob("./examples/**/*", recursive=True):
            try :
                if re.match(r".*\.zip$", paths) :
                    print("Found zip file", paths)
                    with ZipFile(paths, 'r') as zipObj:
                        zipObj.extractall("/".join(paths.split("/")[:-1]))
                    os.remove(paths)
                    print("Extracted zip file", paths)
                elif re.match(r".*\.pdf$", paths) :
                    print("Found pdf file", paths)
                    cv = Converter(paths)
                    cv.convert(paths.replace(".pdf", ".docx"))
                    os.remove(paths)
                    print("Converted pdf file", paths)
                else :
                    continue
            except :
                print("Error extracting zip file", paths)
                continue
                
        print("Unstructured reading files...")
        for paths in glob.glob("./examples/**/*.*", recursive=True):
            if not re.match(r".*\.[a-z]{1,10}$", paths) or re.match(r".*(((I|i)con(s)?)|(\.graffle)|(MACOSX)|(\/out)|(\/bin))\/.*", paths):
                print("- ", paths)
                continue
            
            if re.match(r".*/documents.txt$", paths) or re.match(r".*((\.css)|(\.xml)|(\.csv))", paths):
                print("- ", paths)
                continue
            
            try :
                documents += loader.load_data(file=Path(paths))
                print("+ ", paths)
            except :
                print("- ", paths)
                print("Error loading file")
                continue
            
        #most documents are useless because they contain a lot of noise, so we will have to rand them through a filter
        #first we will separate each documents in chunks of a maximum of 1000 words
        #then we will run each chunk through the filter
        #compose back the chunks into documents
        
        print("Filtering documents...")
        initial_size = sum(list(map(lambda x : len(x.text), documents)))
        c = Compressor()
        for document in documents :
            #split the document into chunks of 5000 words
            chunks = []
            
            for i in range(0, len(document.text.split(" ")), 5000) :
                chunks.append(" ".join(document.text.split(" ")[i:i+5000]))
                
            #try to compress each chunks
            compressed_chunks = []
            for i in range(len(chunks)) :
                compressed_chunks.append(c.compress(chunks[i].encode("utf-8")))
            
            #map the chunks to their compressed ratio
            compressed_chunks = list(map(lambda x : 1 if len(chunks[i]) == 0 else len(x)/len(chunks[i]), compressed_chunks))
            
            print("Compressed chunks:", compressed_chunks)
            
            #reconstruct the document if ratio is below 0.5
            document.text = ""
            for i in range(len(chunks)) :
                if compressed_chunks[i] < 0.9:
                    document.text += chunks[i]
                    
            if document.text == "" or document.text.__contains__("Problem authenticating") or document.text.__contains__("File Generator"):
                documents.remove(document)
            else :
                #remove all lines with less than 10 characters
                #remove all lines that contains more than 10% numbers
                document.text = "\n".join(list(filter(lambda x : False if len(x) < 10 else len(re.findall(r"[0-9]", x))/len(x) < 0.1, document.text.split("\n"))))
            
                
        final_size = sum(list(map(lambda x : len(x.text), documents)))
        print("Initial size:", initial_size)
        print("Final size:", final_size)
        print("Removed", initial_size-final_size, "characters")
            
        t1 = time.time()
        
        print("Documents loaded in", t1-t0, "seconds")
        
        #save all documents to a file
        with open("examples/documents.txt", "w") as f:
            for document in documents:
                f.write(document.text)
                f.write("\n\n")

        print("Indexing...")
        
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
            max_triplets_per_chunk=10,
            space_name=space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
            include_embeddings=True,
            service_context=self.service_context,
        )
        
        vector_index = VectorStoreIndex.from_documents(documents=documents, service_context=self.service_context)
        
        #writing to disk
        
        vector_index.storage_context.persist(persist_dir="index/vector")
        kg_index.storage_context.persist(persist_dir="index/kg")
        
        print("Indexing complete")
        
        return self
        
    def as_query_engine(self, streaming=True):
        
        #count the files in index and if there are none, index the files
        if len(list(Path("index/vector").glob("*"))) == 0 or len(list(Path("index/kg").glob("*"))) == 0 :
            self.index_files()
        
        print("Loading engine...")
        
        storage_context = StorageContext.from_defaults(persist_dir="index/vector")
        vector_index = load_index_from_storage(storage_context, service_context=self.service_context)
        
        storage_context = StorageContext.from_defaults(persist_dir="index/kg")
        kg_index = load_index_from_storage(storage_context, service_context=self.service_context)

        # create custom retriever
        vector_retriever = VectorIndexRetriever(index=vector_index)
        kg_retriever = KGTableRetriever(
            index=kg_index, retriever_mode="keyword", include_text=False
        )
        custom_retriever = CustomRetriever(vector_retriever, kg_retriever)

        # create response synthesizer
        response_synthesizer = get_response_synthesizer(
            service_context=self.service_context,
            response_mode="tree_summarize",
            streaming=streaming
        )
        
        self.engine = RetrieverQueryEngine(
            retriever=custom_retriever,
            response_synthesizer=response_synthesizer,
        )
        
        print("Engine loaded")
        
        return self
        
    def querier(self, streaming=True) :
        if self.engine is None :
            self.as_query_engine(streaming=streaming)
        
        return self