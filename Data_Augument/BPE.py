from bpemb import BPEmb
# load English BPEmb model with default vocabulary size (10k) and 50-dimensional embeddings
bpemb_en = BPEmb(lang="en", dim=50)
#downloading https://nlp.h-its.org/bpemb/en/en.wiki.bpe.vs10000.model
#downloading https://nlp.h-its.org/bpemb/en/en.wiki.bpe.vs10000.d50.w2v.bin.tar.gz