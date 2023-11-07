import os

from llama_index import KnowledgeGraphIndex, StorageContext, VectorStoreIndex

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


def connect_NebulaGraph(service_context):
    from llama_index.graph_stores import NebulaGraphStore

    graph_store = NebulaGraphStore(
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
    )

    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    kg_store = KnowledgeGraphIndex(storage_context=storage_context, service_context=service_context)

    return kg_store


def connect_PostgreSQL(index_name: str, service_context):
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

    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

    return vector_index
