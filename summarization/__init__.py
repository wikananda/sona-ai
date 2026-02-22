from .SummarizationBase import SummarizationBase
from .SummarizationTrainer import SummarizationTrainer
from .SummarizationInferencer import SummarizationInferencer
from .summarization_dataset import SummarizationDataset

# Backward-compatible aliases — old imports from notebooks/scripts still work
FlanT5Base = SummarizationBase
FlanT5Trainer = SummarizationTrainer
FlanT5Inferencer = SummarizationInferencer
