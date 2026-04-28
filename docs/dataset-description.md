Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking And Retrieval
Industrial-scale music recommendation dataset with organic/recommendation interactions and audio embeddings

📌 Overview • 🔑 Key Features • 📊 Statistics • 📝 Format • 🏆 Benchmark • ⬇️ Download • ❓ FAQ

Overview
The Yambda-5B dataset is a large-scale open database comprising 4.79 billion user-item interactions collected from 1 million users and spanning 9.39 million tracks. The dataset includes both implicit feedback, such as listening events, and explicit feedback, in the form of likes and dislikes. Additionally, it provides distinctive markers for organic versus recommendation-driven interactions, along with precomputed audio embeddings to facilitate content-aware recommendation systems.

The Yambda dataset is published exclusively for scientific and research purposes.

Preprint: https://arxiv.org/abs/2505.22238

License
Apache 2.0

Key Features
🎵 4.79B user-music interactions (listens, likes, dislikes, unlikes, undislikes)
🕒 Timestamps with global temporal ordering
🔊 Audio embeddings for 7.72M tracks
💡 Organic and recommendation-driven interactions
📈 Multiple dataset scales (50M, 500M, 5B interactions)
🧪 Standardized evaluation protocol with baseline benchmarks
About Dataset
Statistics
Dataset	Users	Items	Listens	Likes	Dislikes
Yambda-50M	10,000	934,057	46,467,212	881,456	107,776
Yambda-500M	100,000	3,004,578	466,512,103	9,033,960	1,128,113
Yambda-5B	1,000,000	9,390,623	4,649,567,411	89,334,605	11,579,143
User History Length Distribution
user history length

user history length log-scale

Item Interaction Count
item interaction count log-scale

Data Format
File Descriptions
File	Description	Schema
listens.parquet	User listening events with playback details	uid, item_id, timestamp, is_organic, played_ratio_pct, track_length_seconds
likes.parquet	User like actions	uid, item_id, timestamp, is_organic
dislikes.parquet	User dislike actions	uid, item_id, timestamp, is_organic
undislikes.parquet	User undislike actions (reverting dislikes)	uid, item_id, timestamp, is_organic
unlikes.parquet	User unlike actions (reverting likes)	uid, item_id, timestamp, is_organic
multi_event.parquet	Unified events	uid, item_id, timestamp, is_organic, event_type, played_ratio_pct, track_length_seconds
embeddings.parquet	Track audio-embeddings	item_id, embed, normalized_embed
Common Event Structure (Homogeneous)
Most event files (listens, likes, dislikes, undislikes, unlikes) share this base structure:

Field	Type	Description
uid	uint32	Unique user identifier
item_id	uint32	Unique track identifier
timestamp	uint32	Delta times, binned into 5s units.
is_organic	uint8	Boolean flag (0/1) indicating if the interaction was algorithmic (0) or organic (1)
Sorting: All files are sorted by (uid, timestamp) in ascending order.

Unified Event Structure (Heterogeneous)
For applications needing all event types in a unified format:

Field	Type	Description
uid	uint32	Unique user identifier
item_id	uint32	Unique track identifier
timestamp	uint32	Timestamp binned into 5s units.granularity
is_organic	uint8	Boolean flag for organic interactions
event_type	enum	One of: listen, like, dislike, unlike, undislike
played_ratio_pct	Optional[uint16]	Percentage of track played (1-100), null for non-listen events
track_length_seconds	Optional[uint32]	Total track duration in seconds, null for non-listen events
Notes:

played_ratio_pct and track_length_seconds are non-null only when event_type = "listen"
All fields except the two above are guaranteed non-null
Sequential (Aggregated) Format
Each dataset is also available in a user-aggregated sequential format with the following structure:

Field	Type	Description
uid	uint32	Unique user identifier
item_ids	List[uint32]	Chronological list of interacted track IDs
timestamps	List[uint32]	Corresponding interaction timestamps
is_organic	List[uint8]	Corresponding organic flags for each interaction
played_ratio_pct	List[Optional[uint16]]	(Only in listens and multi_event) Play percentages
track_length_seconds	List[Optional[uint32]]	(Only in listens and multi_event) Track durations
Notes:

All lists maintain chronological order
For each user, len(item_ids) == len(timestamps) == len(is_organic)
In multi-event format, null values are preserved in respective lists
Benchmark
Code for the baseline models can be found in benchmarks/ directory, see Reproducibility Guide

Download
Simplest way:

from datasets import load_dataset

ds = load_dataset("yandex/yambda", data_dir="flat/50m", data_files="likes.parquet")

Also, we provide simple wrapper for convenient usage:

from typing import Literal
from datasets import Dataset, DatasetDict, load_dataset

class YambdaDataset:
    INTERACTIONS = frozenset([
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ])

    def __init__(
        self,
        dataset_type: Literal["flat", "sequential"] = "flat",
        dataset_size: Literal["50m", "500m", "5b"] = "50m"
    ):
        assert dataset_type in {"flat", "sequential"}
        assert dataset_size in {"50m", "500m", "5b"}
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size

    def interaction(self, event_type: Literal[
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ]) -> Dataset:
        assert event_type in YambdaDataset.INTERACTIONS
        return self._download(f"{self.dataset_type}/{self.dataset_size}", event_type)

    def audio_embeddings(self) -> Dataset:
        return self._download("", "embeddings")

    def album_item_mapping(self) -> Dataset:
        return self._download("", "album_item_mapping")

    def artist_item_mapping(self) -> Dataset:
        return self._download("", "artist_item_mapping")


    def _download(data_dir: str, file: str) -> Dataset:
        data = load_dataset("yandex/yambda", data_dir=data_dir, data_files=f"{file}.parquet")
        # Returns DatasetDict; extracting the only split
        assert isinstance(data, DatasetDict)
        return data["train"]

dataset = YambdaDataset("flat", "50m")
likes = dataset.interaction("likes")  # returns a HuggingFace Dataset

FAQ
Are test items presented in training data?
Not all, some test items do appear in the training set, others do not.

Are test users presented in training data?
Yes, there are no cold users in the test set.

How are audio embeddings generated?
Using a convolutional neural network inspired by Contrastive Learning of Musical Representations (J. Spijkervet et al., 2021).

What's the is_organic flag?
Indicates whether interactions occurred through organic discovery (True) or recommendation-driven pathways (False)

Which events are considered recommendation-driven?
Recommendation events include actions from:

Personalized music feed
Personalized playlists
What counts as a "listened" track or Listen+?
A track is considered "listened" if over 50% of its duration is played.

What does it mean when played_ratio_pct is greater than 100?
A played_ratio_pct greater than 100% indicates that the user rewound and replayed sections of the track—so the total time listened exceeds the original track length. These values are expected behavior and not log noise. See discussion