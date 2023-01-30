| Topic | Ensembling and Knowledge Distilling of Large Sequence Taggers for Grammatical Error Correction |
| --- | --- |
| Aim | Carry out an investigation of improving the GEC sequence tagging architecture by (1) ensembling multiple Transformer-based encoders, and (2) encoders in large configurations (more number of layers) |
|  | Perform knowledge distillation to train a single sequence tagger model  |
| Motivation | Improving GEC sequence tagging architecture |
| | Performing knowledge distillation since large models, especially due to ensembling, is expensive and convenient for deployment|
| Model | GECToR was the chosen sequence tagging model for the experiment |
|  | Possibly due to the model using linear layers instead of decoders for error detection and error correction. This potentially makes the model several times faster than other seq2seq models. |
| Experiments | Use encoders, BERT, DeBERTa, RoBERTa and XLNet, in their large configuration versions |
| | Try out two methods for ensembling models: (1) Ensembling by average of output tag probabilities, (2) Ensembling by majority votes on output edit spans |
| | Use knowledge distillation to transfer learned knowledge/patterns from an ensemble of multiple models to a single sequence model |
| Results | Using encoders in large configurations does improve results but the drawback is the 2.3-2.5 slower inference speed |
| | The best ensemble model (encoders in large configuration), DeBERTa + RoBERTa + XLNet achieved an F0.5 score of 76.05 on the BEA-2019 test dataset which is the SOTA result during that time. Generally, models are pre-trained on the synthetic model. It is noted that this ensemble model achieved this score without pre-training on synthetic data |
| | The ensemble method, majority votes on output edit spans, performed better than the method, averaging output tag probabilities |
| | Single sequence tagging model using RoBERTa encoder in large configuration achieves a near SOTA score of 73.21 and 72.69 for which slightly different variations of training were used respectively |