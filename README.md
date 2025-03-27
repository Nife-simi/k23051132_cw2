# Machine Learning Coursework 2: Reproducibility Challenge

**algorithm.py:** Implementation of the TPCRP algorithm.

**fully_supervised.py:** Evaluation of the TPCRP algorithm in a fully supervised framework.

**fully_supervised_sse.py:** Evaluation of the TPCRP algorithm in a fully supervised framework with self-supervised embeddings.

**semi_supervised.py:** Evaluation of the TPCRP algorithm in a semi-supervised framework.

**plot.py:** Generates visualizations of the results from the evaluation frameworks.

**statistical_a.py:** Performs statistical analysis on the evaluation results.

**modification.py:** Modifies the original TPCRP algorithm.

**modified_plot.py:** Generates visualizations of the results from the modified algorithm.

**accuracies.txt:** Contains the accuracy results from the evaluation frameworks applied to the original TPCRP algorithm.

**accuracies_random.txt:** Contains the accuracy results from evaluation frameworks using random sampling.

**accuracies_modified.txt:** Contains the accuracy results of the modified TPCRP algorithm evaluated in the fully supervised framework with self-supervised embeddings.

**flexmatch_model.pth:** Pretrained FlexMatch model.

**selflabel_cifar-10.pth:** Pretrained model trained on CIFAR-10 dataset.

**resnet56-4bfd9763.th:** Pretrained ResNet56 model.

## Pretrained Models
To access the pretrained models, download them from the following repositories:

**CIFAR-10 Self-Labelling Model:**
Download the self-labelling model for CIFAR-10 from the SCAN repository: https://github.com/liqing-ustc/SCAN

**ResNet56 Model:**
Download the PyTorch ResNet CIFAR-10 repository to access the ResNet56 model: https://github.com/akamaster/pytorch_resnet_cifar10

**FlexMatch Model:**
Download the TorchSSL repository to access the FlexMatch model: https://github.com/TorchSSL/TorchSSL
