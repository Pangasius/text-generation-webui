"""This module is used to analyze diverse statistics about an index."""

# Use a pipeline as a high-level helper
import torch
import pandas as pd
from transformers import pipeline

from transformers import ZeroShotClassificationPipeline

from glob import glob

from tqdm import tqdm

from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser

from extensions.llamaindex.reader import JiraReaderComments

CANDIDATE_LABELS = ["informative", "uninformative"]


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, nodes):
        self.chunks = [node.text for node in nodes]

    def __getitem__(self, index):
        return self.chunks[index]

    def __len__(self):
        return len(self.chunks)


class EmotionAnalyzer():

    def __init__(self):
        self.pipe: ZeroShotClassificationPipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device="cuda")

    def json_chunker(self, paths):

        reader = SimpleDirectoryReader(input_files=paths,
                                       encoding="utf_8",
                                       file_extractor={".jjson": JiraReaderComments()})

        documents = reader.load_data()

        # transform documents into nodes
        node_parser = SimpleNodeParser.from_defaults(chunk_size=1024,
                                                     chunk_overlap=128)

        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

        return nodes

    def analyze_all_under(self, path):
        results = []

        all_chunks = self.json_chunker(glob(path))

        dataset = ChunkDataset(all_chunks)

        for result in tqdm(self.pipe(dataset, CANDIDATE_LABELS, multi_label=False, batch_size=24), total=len(dataset)):
            results += [result]

        return results, all_chunks

    def plot(self, results, chunks):
        # plot the distribution of emotions
        import matplotlib.pyplot as plt
        import numpy as np

        # get the scores
        scores = np.array([result["scores"] for result in results])

        # bar plot of the informativeness (first label)
        informative_scores = scores[:, 0]

        bins = np.linspace(0, 1, 100)
        plt.hist(informative_scores, bins=bins)
        plt.title("Distribution of informativeness")
        plt.xlabel("Informativeness score")
        plt.ylabel("Count")
        plt.show()


if __name__ == "__main__":
    analyzer = EmotionAnalyzer()

    results, chunks = analyzer.analyze_all_under("examples/f_embed_jira_raw/jira_f/Tickets/ALTERNA/*.jjson")

    # save the results and chunks in a csv
    df = pd.DataFrame(results)
    #df["chunks"] = chunks
    df.to_csv("scripts/analysis.csv", index=False)

    analyzer.plot(results, chunks)
