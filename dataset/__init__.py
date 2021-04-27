from dataset.registry import register

register("pytorch-seq2seq-tutorial",
         entry_point="dataset.mt:EngFraDataProcessor")
register("translation2019zh",
         entry_point="dataset.mt:Translation2019ZhProcessor")
