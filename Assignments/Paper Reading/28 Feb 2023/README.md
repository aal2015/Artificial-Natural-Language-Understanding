| Topic | BERT Learns to Teach: Knowledge Distillation with Meta Learning |
| --- | --- |
| Aim | Improve teacher model to better transfer knowledge to student model |
| Current issues with knowledge distillation (KD) | The teacher is unaware of the student’s capacity |
| | The teacher is not optimized for distillation |
| Proposed Solution  | Knowledge Distillation with Meta Learning (MetaDistil) |
| | Allows to exploit feedback from student's learning progress to improve teacher's knowledge transfer ability throughout the distillation process |
| | Add (proposed) a 'pilot update' mechanism in the MetaDistil framework to align the learning of the bi-level learners |
| Tasks to test MetaDistil | Text and image classification tasks |
| Results | MetaDistil outperforms knowledge distillation by a large margin |
| | Achieves state-of-the-art performance compressing BERT  on the GLUE benchmark |
| | Shows competitive results compressing ResNet and VGG on CIFAR-100 |
| | More robust than conventional KD to different student capacities  and hyperparameters |
