# Contrastive Learning Reduces Hallucination in Conversations

We propose *MixCL*, a contrastive learning framework to reduce the hallucination of LM-based dialogue agents.

The code is developed upon [pytorch](https://pytorch.org/) and [huggingface transformers](https://github.com/huggingface/transformers).

The code for extrating spans is available at `mixup.py`, where we use  [stanza](https://github.com/stanfordnlp/stanza/) and  [spacy](https://github.com/explosion/spaCy) to identify entities and constituencies in text.

The code for model training and testing is available at `run.py`

The dataset (i.e., [Wizard-of-Wikipedia](https://parl.ai/projects/wizard_of_wikipedia/)) is placed in `/dataset`, and `/utils` provides the code for IO and evaluation. 



