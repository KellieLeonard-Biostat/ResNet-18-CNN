# Automated Classification of Colorectal Cancer Histology Images Using a Fine-Tuned ResNet-18 Convolutional Neural Network


**Abstract**
Accurate histopathological classification of colorectal tissue is critical for guiding clinical management of colorectal cancer (CRC). We developed and deployed an automated classification system using a fine-tuned ResNet-18 convolutional neural network (CNN) trained on the NCT-CRC-HE-100K dataset, comprising over 100,000 hematoxylin and eosin (H&E)-stained image patches. Our approach achieved a peak validation accuracy of 99.39%, demonstrating the potential of deep learning to assist pathologists in high-throughput diagnostic workflows.
 
**Introduction**
Colorectal cancer is a leading cause of cancer-related morbidity and mortality worldwide. Histopathological analysis of H&E-stained tissue sections remains the diagnostic gold standard; however, it is time-consuming and subject to inter-observer variability. Recent advances in deep learning, particularly CNNs, offer promising tools for automating tissue classification and potentially enhancing diagnostic accuracy and efficiency. In this study, we present a robust pipeline for the classification of colorectal histology images into nine tissue subtypes, leveraging a transfer learning approach with ResNet-18.
 
**Materials and Methods**
**Dataset**
We utilised the NCT-CRC-HE-100K dataset, publicly available via Zenodo, containing 100,000+ non-overlapping image patches extracted from digitised whole-slide images of colorectal tissue sections stained with H&E. All image patches are uniformly sized at 224 × 224 pixels, acquired at 20× magnification, and stored in RGB format. The dataset is annotated into nine tissue categories:
•	Adipose tissue (ADI)
•	Background (BACK)
•	Debris (DEB)
•	Lymphocytes (LYM)
•	Mucus (MUC)
•	Smooth muscle (MUS)
•	Normal colon mucosa (NORM)
•	Cancer-associated stroma (STR)
•	Colorectal adenocarcinoma epithelium (TUM)

**Model Architecture**
A ResNet-18 architecture, pre-trained on ImageNet, was selected as the backbone CNN. The final fully connected layer was replaced to output nine classes, matching the dataset’s tissue categories. Transfer learning was employed to leverage low-level feature representations learned from natural images, followed by fine-tuning on histology data to adapt to domain-specific features.

**Preprocessing and Augmentation**
Input images were normalized using ImageNet statistics (mean: [0.485, 0.456, 0.406]; standard deviation: [0.229, 0.224, 0.225]). Data augmentation strategies, including random horizontal and vertical flips, rotations, and colour jittering, were applied to improve model generalisation and mitigate overfitting.

**Training Procedure**
The model was trained for 10 epochs using a cross-entropy loss function. Training loss decreased steadily from 0.1012 at initialisation to 0.0090 at completion, indicating effective convergence. Validation loss ranged between 0.0203 and 0.0365, suggesting stable generalisation across unseen data. The model achieved a peak validation accuracy of 99.39%.
 
**Results**
•	Training loss: Reduced from 0.1012 to 0.0090 over 10 epochs.
•	Validation loss: Maintained between 0.0203 and 0.0365.
•	Validation accuracy: Peaked at 99.39%, consistently above 98% throughout final epochs.
 
**Deployment**
To facilitate real-world applicability, the trained model was integrated into an interactive web-based application using Streamlit. The platform enables clinicians to upload H&E-stained CRC images (in TIFF, PNG, or JPEG format), which are automatically preprocessed and classified in real time. TIFF images with single-channel or multiple-channel configurations are appropriately converted to RGB to ensure robust compatibility.
An interpretability layer was included to flag predicted outputs as "urgent" or "non-urgent" based on tissue subtype, enabling rapid triage. For example, predictions corresponding to TUM (colorectal adenocarcinoma epithelium) and STR (cancer-associated stroma) were prioritized for immediate attention.
 
**Discussion**
Our study demonstrates the feasibility of employing a fine-tuned ResNet-18 CNN for automated classification of CRC histology images with exceptionally high accuracy. The use of transfer learning allowed us to leverage rich feature hierarchies from ImageNet, while domain-specific fine-tuning adapted these features to histopathological structures.
The integration into an interactive application bridges the gap between algorithmic development and clinical utility, providing an assistive tool for rapid tissue type identification. While our current model shows excellent performance, future work should explore external validation on independent datasets and investigate interpretability approaches (e.g., class activation maps) to enhance trust and adoption among pathologists.
 
**Conclusion**
We present a robust, high-accuracy, CNN-based pipeline for colorectal histology classification, trained on a large-scale, diverse dataset. The resulting model achieves near-perfect validation accuracy and is deployed as a user-friendly application, providing a potential adjunct tool to support pathologists in CRC diagnosis and triage. Future studies will focus on further generalizability and clinical validation.
 
**References**
Kather JN, et al. "Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study." PLoS Med 16(1): e1002730. https://zenodo.org/record/1214456
 
