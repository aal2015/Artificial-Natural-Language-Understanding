| Topic | Attention Temperature Matters in Abstractive Summarization Distillation |
| ------------- | ------------- |
| Paper Link | https://aclanthology.org/2022.acl-long.11/ |
| Primary focus | Knowledge distillation and manipulating attention temperatures |
| Problem Addressed | Abstractive text summarization largely relies on large pre-trained seq-to-seq Transformer models. This makes it computationally expensive. |
| Solution | Distill large models into smaller ones |
| Method for Performing Knowledge Distillation | Transfer knowledge from the larger (teacher) model to the smaller (student) model. Main aim: minimize predictions  between teacher and student model
| Data for Knowledge Distillation | Use pseudo-labeling where the teacher model generates pseudo summaries for all documents in the training set. Result: document-pseudo-summary pairs. |
| Issues of Teacher Model| Attention distribution might be too sharp (weights much larger at the leading part of a document). This causes copy bias (tendency to copy some texts more) and leading bias (tendency to summarize using mostly the leading part of a document). |
| Proposed Solution | PLATE (Pseudo-labeling with Larger Attention Temperature). A method to re-scale attention weights in all attention modules for softer attention distribution. |
| Results | Consistent improvement over vanilla pseudo-labeling based methods |
| | Summaries produced by the student model are shorter and more abstractive due to lesser copy and lead biases. |
