| Topic  | Document-level Grammar Error Correction | 
| ------------- | ------------- |
| Issue | Currently, all GEC systems process each sentence independently. However, document-level context may be needed to correct certain errors (e.g. present or past tense). Also, processing at the sentence level may lead to inconsistent modifications throughout whole document. |
| Solution | Incorporate document-level context |
| Contributions | <ol><li>Propose document-level GEC systems</li> <li>Employ three-step training strategy</li> <li>Compare with NMT-based models for document-level context evalutation</li></ol> |
| Model | Sentence level (baseline): encoder-decoder transformer </br> Document-context level: (1) Single Encoder Model, (2) Multi-encoder encoder side, (3) Multi-encoder decoder side |
| Single-encoder Model | Same architecture as the baseline model for sentence level. However, the difference is that this model considers context by concatenating previous sentence(s) to current sentence.
| Multi-encoder Models | Has an extra encoder to process the current sentence and its context separately. The resulting context from two encoders can be integrated in the encoder side and decoder side hence leading to two models based on this. |
| Dataset | Train dataset <ol><li>Cambridge English Write and Improve (W&I)</li> <li>First Certificate in English (FCE)</li> <li>National University of Singapore Corpus of Learner English (NUCLE)</li> <li>Cambridge Learner Corpus (CLC)</li></ol> Develoment dataset <ol><li>FCE-dev</li></ol> Test dataset <ol><li>FCE-test</li> <li>BEA-dev</li> <li>CoNLL-2014 test set</li></ol>
| Three-step Training Strategy | Proposed due to limited <ol><li>large-scale document-level GEC corpora</li> <li>existing methods for artificial generation work</li></ol> Three Steps: <ol><li>Pretrain on sentence-level data from CLC + FCE-train + W&I-train + NUCLE </li> <li>Continue train on document-level data from CLC</li> <li>Fine-tune on combination of small, in-domain document-level data from FCE-train + W&I-train + NUCLE</li></ol> |
| Results | Single-encoder Model does not improve consistently for  F0.5 score. Recall improves the cost of precision. |
| | Document-level context models gives better results than baseline model at sentence-level |
| | Multi-encoder on decoder side integration performs better than multi-encoder on encoder side integration for FCE-test and CoNLL-2014. However, both multi-encoder gave similar results on BEA-dev |
| | Found using three-step training crucial for training models. Dropping either pretraining or fine-tuning steps in the ablation study showed a drop in performance. |
| | Found incorporating one previous sentence to the current sentence for dataset FCE-test and BEA-dev and one for CoNLL-2014 to give the best performance for including context. This may be due to CoNLL-2014 containing twice as many sentences on average. Overall, the long-distance context had a limited impact on GEC. |
| | Displayed biggest improve for subject-verb agreement, preposition, noun number, determiner and pronoun errors. |
| | Multi-encoder model on decoder integration outperforms NMT-based document-level context models by large margins. |
| | Achieves state of the art on FCE-test. |
