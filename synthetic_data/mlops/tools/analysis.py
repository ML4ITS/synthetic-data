from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import Dataset


def visualization(
    original_data: torch.Tensor,
    generated_data: torch.Tensor,
    analysis_type: str,
    percent: float = 0.1,
):
    """Vizualizes the original and generated data using PCA and TSNE.

    Args:
        original_data (torch.Tensor): original_data
        generated_data (torch.Tensor): generated data
        analysis_type (str): PCA or t-SNE
        percent (float, optional): percentage size of the total data. Defaults to 0.1.

    Raises:
        ValueError: raises error on unsupported analysis_type

    refactored version of https://github.com/jsyoon0823/TimeGAN
    """
    max_samples = len(original_data)
    n_samples = int(max_samples * percent)
    idx = np.random.permutation(max_samples)[:n_samples]

    # convert to numpy array, and extract given indices
    original_data = np.asarray(original_data)[idx]
    generated_data = np.asarray(generated_data)[idx]

    _, seq_len, _ = original_data.shape

    for i in range(n_samples):
        if i == 0:
            prep_data = np.reshape(np.mean(original_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(
                np.mean(generated_data[0, :, :], 1), [1, seq_len]
            )
        else:
            prep_data = np.concatenate(
                (
                    prep_data,
                    np.reshape(np.mean(original_data[i, :, :], 1), [1, seq_len]),
                )
            )
            prep_data_hat = np.concatenate(
                (
                    prep_data_hat,
                    np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len]),
                )
            )

    if analysis_type.lower() == "pca":
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        plt.figure(figsize=(10, 10), dpi=100)
        plt.scatter(
            pca_results[:, 0],
            pca_results[:, 1],
            c="red",
            alpha=0.2,
            label="Original",
        )
        plt.scatter(
            pca_hat_results[:, 0],
            pca_hat_results[:, 1],
            c="blue",
            alpha=0.2,
            label="Synthetic",
        )

        plt.legend()
        plt.title("PCA")
        plt.xlabel("x-pca")
        plt.ylabel("y-pca")
        plt.show()

    elif analysis_type.lower() == "tsne":

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        plt.figure(figsize=(10, 10), dpi=100)

        plt.scatter(
            tsne_results[:n_samples, 0],
            tsne_results[:n_samples, 1],
            c="red",
            alpha=0.2,
            label="Original",
        )
        plt.scatter(
            tsne_results[n_samples:, 0],
            tsne_results[n_samples:, 1],
            c="blue",
            alpha=0.2,
            label="Synthetic",
        )

        plt.legend()
        plt.title("t-SNE")
        plt.xlabel("x-tsne")
        plt.ylabel("y-tsne")
        plt.show()
    else:
        raise ValueError(f"Invalid analysis type: {analysis_type}")


def sequence_to_feature(sequence: torch.Tensor) -> torch.Tensor:
    """Extract meaningful features from sequence, such as median, mean,
       standard deviation, variance, root mean square,
       maximum and minimum values of each input sequence

    Args:
      - seq: sequence (1x 1024)

    Returns:
      - features: extracted features [median, mean, standard deviation, variance, root
        mean square, maximum, and minimum values of each input sequence]
    """
    assert sequence.ndim == 1
    _median = sequence.median()
    _mean = sequence.mean()
    _std = sequence.std()
    _var = sequence.var()
    _rms = torch.sqrt(torch.mean(sequence**2))
    _max = sequence.max()
    _min = sequence.min()
    return torch.tensor(
        [_median, _mean, _std, _var, _rms, _max, _min], dtype=torch.float
    )


def sequences_to_features(sequences: torch.Tensor) -> torch.Tensor:
    """Extract meaningful features from sequence batch.

    Args:
      - sequences: sequences

    Returns:
      - features: features
    """
    assert sequences.ndim == 2
    return torch.stack([sequence_to_feature(seq) for seq in sequences])


def cos_sim(seq1: torch.Tensor, seq2: torch.Tensor) -> torch.Tensor:
    """Normalizes between [-1, 0] and computes the cosine similarity between them.

    Args:
      - seq1: (torch.Tensor) sequence 1
      - seq2: 8torch.Tensor) sequence 2

    Returns:
      - cos_sim: (torch.Tensor) the cosine similarity
    """
    LHP_RANGE = [-1, 0]
    # normalize sequences into LHP_RANGE
    nseq1 = normalize(seq1, *LHP_RANGE)
    nseq2 = normalize(seq2, *LHP_RANGE)

    # cosine_similarity requires values in the LHP_RANGE
    return torch.cosine_similarity(nseq1, nseq2)


def avg_cos_sim(seq1: torch.Tensor, seq2: torch.Tensor) -> float:
    """Compute average cosine similarity between two sequences.

    Args:
      - seq1: (torch.Tensor) sequence 1
      - seq2: 8torch.Tensor) sequence 2

    Returns:
      - cos_sim: (float) average cosine similarity
    """
    return cos_sim(seq1, seq2).mean().item()


def genererate_sequences(
    generator: torch.nn.Module, labels: torch.Tensor
) -> torch.Tensor:
    """Generate sequences from a generator.

    Args:
        generator (torch.nn.Module): the generator
        labels (torch.Tensor): the labels of the sequences to generate

    Returns:
        torch.Tensor: the generated sequences
    """
    z_dim = 100
    n_samples = labels.shape[0]
    noise = torch.randn((n_samples, z_dim))
    return generator(noise, labels)


def normalize(data: torch.Tensor, minval: int, maxval: int):
    """Normalize data to range [minval, maxval]"""
    return (maxval - minval) * (
        (data - data.min()) / (data.max() - data.min())
    ) + minval


def extract_sequences(dataset: Dataset) -> torch.Tensor:
    """Extracts only the sequences from dataset"""
    assert isinstance(dataset, Dataset), "Dataset must be an instance of Dataset"
    return dataset.data


def generate_labels(
    n_samples: int, n_samples_per_dataset: int, n_classes: int
) -> torch.Tensor:
    """Generates a dataset of labels with shape (n_samples,),
      divided into samples_per_dataset blocks, where
      the first block contains indices from from the first_class,
      the second block contains indices from the second_class,
      and so on.

      Example: n_samples = 4, samples_per_dataset = 1, n_classes = 4
      Returns: [0,
                1,
                2,
                3]

    Args:
        n_samples (int): number of samples to combined in all the datasets
        n_samples_per_dataset (int): samples per dataset
        n_classes (int): number of classes

    Returns:
        torch.Tensor: a combined dataset of labels
    """
    labels = torch.empty((n_samples,), dtype=torch.int)
    for i in range(n_classes):
        labels[i * n_samples_per_dataset : (i + 1) * n_samples_per_dataset] = i
    return labels


def sample_datasets(
    index: int, data1: torch.Tensor, data2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """retreives certain similar indexes sequences from two datasets,
      NOTE: assumes the datasets both are ordered or both random shuffled

    Args:
        index (int): indices to sample from the datasets
        data1 (torch.Tensor): the first dataset
        data2 (torch.Tensor): the second dataset

    Returns:
        datasets: (torch.Tensor, torch.Tensor), the sampled datasets
    """
    # required ordered datasets
    indices = np.arange(index * 1000, (index + 1) * 1000)
    datasets = data1[indices], data2[indices]
    return datasets


def slerp(
    value: float,
    noise_1: torch.Tensor,
    noise_2: torch.Tensor,
    threshold: float = 0.9995,
):
    """Interpolate between two noise vectors by a given decimal (t) value.

    Args:
        value (float): the time value between 0 and 1.
        noise_1 (torch.Tensor): the first noise vector.
        noise_2 (torch.Tensor): the second noise vector.
        threshold (float, optional): stop threshold. Defaults to 0.9995.

    Returns:
        torch.Tensor: the interpolated noise vector.

    """

    dot = _calc_dot(noise_1, noise_2)

    if torch.abs(dot) > threshold:
        noise = _lerp(value, noise_1, noise_2)
    else:
        noise = _slerp(value, noise_1, noise_2, dot)
    return noise


def _slerp(t, p0, p1, dotp):
    """Spherical linear interpolation between two vectors.

    Args:
        t (float): the time value between 0 and 1.
        p0 (torch.Tensor): the first noise vector.
        p1 (torch.Tensor): the second noise vector.
        dotp (torch.Tensor): some dot product.

    Returns:
        torch.Tensor: the interpolated noise vector.

    refactor version of  https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
    """
    theta_0 = torch.arccos(dotp)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    noise = s0 * p0 + s1 * p1
    return noise


def _lerp(t, p0, p1):
    """Linear interpolation between two vectors.

    Args:
        t (float): the time value between 0 and 1.
        p0 (torch.Tensor): the first noise vector.
        p1 (torch.Tensor): the second noise vector.

    Returns:
        torch.Tensor: the interpolated noise vector.
    """
    noise = (1 - t) * p0 + t * p1
    return noise


def _calc_dot(noise_1: torch.Tensor, noise_2: torch.Tensor):
    return torch.sum(
        noise_1 * noise_2 / (torch.linalg.norm(noise_1) * torch.linalg.norm(noise_2))
    )


def create_embedded_noise(
    embedder: torch.nn.Module, fixed_noise: torch.Tensor, label: str
) -> torch.Tensor:
    """Create embedded noise from a given label using the embedder
    from the conditional GAN model.

    NB: should be equal to architecture used in the conditional GAN model class
    (e.g. see ../models/cgan.py)

    Args:
        embedder (torch.nn.Module): the embedder from the trained Conditional GAN model.
        fixed_noise (torch.Tensor): the fixed noise vector.
        label (str): The first frequency.

    Returns:
        torch.Tensor: the embedded noise vector.
    """
    label = torch.tensor(label).to(int).unsqueeze(0)
    label_embedding = embedder(label)
    noise = torch.cat((label_embedding, fixed_noise), -1)
    return noise
