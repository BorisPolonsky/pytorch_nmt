from dataset.registry import register

register("pytorch-seq2seq-tutorial",
         entry_point="dataset.mt:EngFraDataProcessor")
