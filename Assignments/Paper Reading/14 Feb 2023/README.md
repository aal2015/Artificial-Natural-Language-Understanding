| Topic | Attention Temperature Matters in Abstractive Summarization Distillation |
| ------------- | ------------- |
| Paper Link | https://aclanthology.org/2022.acl-long.11/ |
| Primary focus | Knowledge distillation and manipulating attention temperatures |
| Problem Addressed | Abstractive text summarization largely relies on large pre-trained seq-to-seq Transformer models. This makes it computationally expensive. |
| Solution | Distill large models into smaller ones |
| Method for Performing Knowledge Distillation | Transfer knowledge from the larger (teacher) model to the smaller (student) model. Main aim: minimize predictions  between teacher and student model
| Data for Knowledge Distillation | Use pseudo-labeling where the teacher model generates pseudo summaries for all documents in the training set. Result: document-pseudo-summary pairs. |
| Issues of Teacher Model| Attention distribution might be too sharp (weights much larger at the leading part of a document). This causes copy bias (tendency to copy some texts more) and leading bias (tendency to summarize using mostly the leading part of a document). |
| Proposed Solution | PLATE (Pseudo-labeling with Larger Attention Temperature). A method to re-scale attention weights in all attention modules for softer attention distribution (for teacher model only). This is done by using a higher attention temperature as shown in the equation where lambda is added.  |
| Results | Consistent improvement over vanilla pseudo-labeling based methods |
| | Summaries produced by the student model are shorter and more abstractive due to lesser copy and lead biases. |

![attention formula](https://user-images.githubusercontent.com/28766535/221816547-4a19bbb0-b9b4-4932-981b-6da27b5856b1.png)
![lambda](https://user-images.githubusercontent.com/28766535/221816910-424513b4-f915-4b74-a9c5-5809b18e02f8.png)
