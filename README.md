# Spatial Image Segmentation for Fish Cage Detection using U-Net
## Project Overview
This project focuses on spatial data mining by applying the U-Net Convolutional Neural Network (CNN) architecture to perform semantic image segmentation. The primary objective of this model is to automatically detect and map the distribution of fish cages (keramba) located within the Saguling Reservoir (Waduk Saguling) using spatial imagery.

<img width="1212" height="868" alt="2427860460-db5b979f-fa41-4662-995d-7579253edd2a" src="https://github.com/user-attachments/assets/63f852d9-00a7-489d-95e6-69a1fbe6d5fd" />

## Data Preprocessing and Augmentation
To enhance the model's ability to learn and generalize, extensive data augmentation techniques were applied to the training dataset.
* The augmentation process increased data variance by altering the original imagery's visual aspects, including brightness, contrast, and gamma levels.
* Geometric transformations such as rotation and flipping were utilized to change the spatial orientation of the images.
* Unlike the original static dataset, the augmented data provided a much broader and more complex range of variations for the network to study.
* Implementing these variations proved to be a critical step in mitigating overfitting, ensuring the model performed reliably during both the training and testing phases.

## Model Architecture

<img width="1388" height="768" alt="image" src="https://github.com/user-attachments/assets/7c4f0f2f-c381-4697-a570-1aa05e7471e7" />

The segmentation model is built upon the foundational U-Net architecture, which is highly effective for spatial image segmentation. 
* The network structure consists of a contracting encoder path for context capture and an expanding decoder path for precise localization.
* The architecture utilizes 3x3 Convolutional layers paired with ReLU activation functions to extract deep features.
* 2x2 Max Pooling layers are heavily employed in the encoder to downsample the spatial dimensions.
* The decoder path uses 2x2 Up-convolutions to restore spatial resolution, uniquely combined with "Crop and Concatenate" skip connections to merge high-level semantic features with the localized spatial information preserved from the encoder.

## Training Performance and Evaluation
The model's performance was rigorously evaluated using Accuracy and the Intersect Over Union (IoU) metric, which is based on the Jaccard Index.
* IoU measures the area of overlap between the predicted bounding box and the ground truth, divided by the area of union between them.
* This metric is highly effective in evaluating how accurately the CNN algorithms classify objects by observing the displacement between the actual object and the resulting segmented prediction.

### Training Results (Epochs 1 - 25)
* The model exhibited rapid initial learning, with the Training IoU increasing significantly from epoch 1 to 5 as the network quickly grasped the primary shapes and patterns of the fish cages.
* Following the early epochs, the IoU continued to increase gradually as the network began learning finer structural details of the cages, such as edges, corners, and internal frameworks.
* While the Validation IoU generally followed the upward trend of the Training IoU, slight drops indicating minor overfitting were observed between epochs 17 and 25.
* The Training Loss started relatively high at 0.2918 and steadily decreased, signifying that the model actively reduced the amount of unrecognized, "discarded" data over time.
* Minor fluctuations in Validation Loss occurred at epochs 6, 21, and 24, which were likely caused by the model adjusting to the high visual variance introduced by the data augmentation process.

<img width="1402" height="562" alt="image" src="https://github.com/user-attachments/assets/6d9528d9-1419-48c3-b38c-8ee06589654f" />


### Final Model Metrics
The final evaluation metrics logged at Epoch 25 demonstrated strong generalizability on unseen data:

| Metric | Value |
| :--- | :--- |
| **Training Accuracy** | 97.56%  |
| **Training IoU** | 0.6276  |
| **Testing Accuracy** | 97.42%  |
| **Testing IoU** | 0.5784  |
| **Testing Loss** | 0.0834  |

## Spatial Classification Results
The trained U-Net model was deployed to generate a complete segmentation map of the Saguling Reservoir.
* The final classification values in the map range from 1 to 16, representing the frequency or confidence level of the model detecting a fish cage in a specific pixel area during the detection phase.
* Areas highlighted in red (values 14-16) indicate a very high probability of being fish cages, meaning the system confidently recognized these structures up to 16 times.
* Lower values (1-10) indicate a smaller probability; however, the vast majority of these lower confidence detections still correctly aligned with actual fish cage locations, pointing to room for higher epoch limits.
* **Known Limitations:** The model occasionally misclassified certain land features as fish cages due to similar geometric shapes and visual patterns learned during training.
* **Proposed Solution:** Implementing a spatial land-masking process post-classification can effectively eliminate these false positive misclassifications.

## Future Enhancements
* **Hyperparameter Optimization:** Increasing the total number of epochs alongside implementing an early stopping mechanism could further optimize the IoU without risking severe overfitting.
* **Advanced Architectures:** Transitioning the network to variations like **Attention U-Net** could improve segmentation by utilizing attention mechanisms to focus specifically on the most relevant features and detailed structural information of the target objects.
* **Foundation Models:** Leveraging Pre-trained or Foundation models trained on massive global remote sensing datasets could provide a much more robust starting point, significantly reducing computational overhead while improving baseline accuracy.

## References
* Mulia, S. B., Nugraha, N. W., & Robbani, M. H. (2023). Implementasi Machine Learning Untuk Identifikasi Orang Batuk/Bersin. *Journal Of Energy And Electrical Engineering, 4*(2), 81-86. 
* Padilla, R., Netto, S., & da Silva, E. (2020). A Survey on Performance Metrics for Object-Detection Algorithms. *10.1109/IWSSIP48289.2020*. 
* Sun, Y., Bi, F., Gao, Y., Chen, L., & Feng, S. (2022). A Multi-Attention UNet for Semantic Segmentation in Remote Sensing Images. *Symmetry, 14*, 1-19. 
* Wu, C., et al. (2021). Building Damage Detection Using U-Net with Attention Mechanism from Pre- and Post-Disaster Remote Sensing Datasets. *Remote Sensing, 13*, 1-22. 
