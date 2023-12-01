"""Module to provide access to different llama_index connections."""
import os

from llama_index.graph_stores import NebulaGraphStore
from llama_index.vector_stores import PGVectorStore, ElasticsearchStore

from llama_index import (
    KnowledgeGraphIndex as KGI,
    ServiceContext,
    StorageContext,
    VectorStoreIndex as VSI,
)

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


def connect_store(store: str, index_name: str, service_context: ServiceContext):
    """Choose the store to connect to and return the corresponding index object."""

    if store == "nebula":
        return connect_nebulagraph(index_name, service_context)
    if store == "postgresql":
        return connect_postgresql(index_name, service_context)
    if store == "elastic":
        return connect_elastic(index_name, service_context)

    raise ValueError("Store not supported")


def connect_nebulagraph(index_name: str, service_context: ServiceContext) -> KGI:
    """
    Connect to NebulaGraph and return a KnowledgeGraphIndex object.

    Args:
        service_context (ServiceContext): ServiceContext object

    Returns:
        kg_store (KnowledgeGraphIndex): KnowledgeGraphIndex object
    """

    graph_store = NebulaGraphStore(
        space_name=index_name,
        edge_types=["relationship"],
        rel_prop_names=["relationship"],
        tags=["entity"],
    )

    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    kg_store = KGI(storage_context=storage_context, service_context=service_context)

    return kg_store


def connect_postgresql(index_name: str, service_context: ServiceContext) -> VSI:
    """
    Connect to PostgreSQL and return a VectorStoreIndex object.

    Args:
        index_name (str): Name of the index
        service_context (ServiceContext): ServiceContext object

    Returns:
        vector_index (VectorStoreIndex): VectorStoreIndex object
    """

    vector_store = PGVectorStore.from_params(
        database="llamaindex",
        host="localhost",
        password="index",
        port="5432",
        user="llama",
        table_name=index_name,
        embed_dim=1024,  # embedding dimension
    )

    vector_index = VSI.from_vector_store(vector_store=vector_store,
                                                           service_context=service_context)

    return vector_index


def connect_elastic(index_name: str, service_context: ServiceContext) -> VSI:
    """
    Connect to ElasticSearch and return a VectorStoreIndex object.

    Args:
        index_name (str): Name of the index
        service_context (ServiceContext): ServiceContext object

    Returns:
        vector_index (VectorStoreIndex): VectorStoreIndex object
    """

    vector_store = ElasticsearchStore(index_name=index_name,
                                      es_url="http://localhost:9200")

    vector_index = VSI.from_vector_store(vector_store=vector_store,
                                                           service_context=service_context)

    return vector_index
