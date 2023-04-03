"""
    Load / normalize / shortens model data (google's model requires gensim)
    Does NOT load the model into memory, just pre-processes the data, line by line
    Author: Wolf Paulus, https://wolfpaulus.com
"""

from math import sqrt


def google_news():
    """ bin to text, takes a really long time and creates a 10GB file"""
    model = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
    model.save_word2vec_format('models/googlenews.txt')


def normalize_to_str(v: [float]) -> str:
    """ normalize vector and return as string (space separated) """
    length = sqrt(sum([x * x for x in v]))
    return " ".join([f"{x / length:.4f}" for x in v])


def pre_process_data(in_file: str, out_file: str, features: int = 300, normalize: bool = True, skip_1st: bool = True,
                     lines: int = 0) -> None:
    """
    Pre-process word vectors
    :param in_file: input file
    :param out_file: output file
    :param features: number of features, defaults to 300
    :param normalize: normalize vectors, defaults to True
    :param skip_1st: skip first line defaults to False
    :param lines: number of lines to process, 0 for all
    :return: None
    """
    with open(in_file, 'r') as in_file:
        with open(out_file, mode="w", encoding="utf8") as out_file:
            if skip_1st:
                in_file.readline()
            for line in in_file:
                sa = line.split()
                if len(sa) != features + 1:
                    continue
                if normalize:
                    word, vec = sa[0], [float(x) for x in sa[1:]]
                    line = f"{word} {normalize_to_str(vec)}\n"
                out_file.write(line)
                if lines > 0:
                    lines -= 1
                    if lines == 0:
                        break


if __name__ == "__main__":
    # from gensim.models import KeyedVectors
    # google_news()
    # pre_process_data("models/googlenews.txt", "models/google_short.txt", lines=100_000)
    pre_process_data("models/wiki-news-300d-1M.txt", 'models/wiki_short.txt', lines=100_000)
    pre_process_data("models/glove.840B.300d.txt", "models/glove_short.txt", lines=100_000, skip_1st=False)
