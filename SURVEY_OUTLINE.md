# Survey: Privacy Attacks in Federated Learning via GradInversion

## Title Page
- Title
- Author name & student ID
- Course: Federated Learning Advanced Topics
- Institution: IIT Patna
- Date

## 1. Abstract (150-200 words)
- Brief overview of gradient inversion attacks
- Main contributions of GradInversion
- Key results (88% success rate on ImageNet batches)

## 2. Introduction (500-700 words)
2.1 Federated Learning Background
- Privacy advantages of FL
- Why gradients were assumed safe
- Motivation for privacy attacks

2.2 Threat Model
- Honest-but-Curious server
- Gradient leakage scenario
- Attack objectives

2.3 Paper Contributions
- Batch recovery from averaged gradients
- Label restoration algorithm
- Group consistency regularization

## 3. Related Work (600-800 words)
3.1 Network Inversion
- DeepInversion (Yin et al., 2020)
- Activation maximization
- Generative priors

3.2 Gradient-Based Inversion
- DLG (Zhu et al., 2019)
- iDLG (Zhao et al., 2020)
- Geiping et al. (2021)

3.3 Federated Learning Security
- Membership inference
- Model inversion attacks
- Defense mechanisms

## 4. GradInversion Method (1000-1200 words)
4.1 Problem Formulation
- Objective function (Eq. 1)
- Gradient matching loss (Eq. 3)

4.2 Label Restoration (Sec 3.2)
- FC layer gradient analysis
- Sign preservation property
- Batch-level label recovery (Eq. 8)

4.3 Fidelity Regularization (Sec 3.3)
- Total variation & L2 priors
- Batch norm statistics (DeepInversion)

4.4 Group Consistency Regularization (Sec 3.4)
- Multi-seed optimization
- Image registration
- Consensus image formation

4.5 Optimization Details
- Langevin updates
- Learning rate schedule
- Hyperparameter settings

## 5. Experimental Results (1200-1500 words)
5.1 Experimental Setup
- Datasets: CIFAR-10, ImageNet
- Models: ResNet-18, ResNet-50
- Metrics: MSE, PSNR, LPIPS, Attack Success Rate

5.2 Ablation Studies
- Contribution of each loss term
- Label restoration accuracy (Table 1)
- Impact of BN statistics

5.3 Comparison with State-of-the-Art
- vs. DLG (Zhu et al.)
- vs. iDLG (Zhao et al.)
- vs. Geiping et al.
- vs. DeepInversion
- Visual comparisons (Fig. 4)

5.4 Scaling to Large Batches
- Batch size impact (4→48 images)
- Information leakage vs batch size
- Image Identifiability Precision (IIP) metric

5.5 Key Findings
- Surprising accuracy of reconstruction
- Vulnerable population discovery
- Communication efficiency implications

## 6. Privacy Implications (400-600 words)
6.1 Threat to Federated Learning
- When averaging is not sufficient
- Implications for batch sizes
- Real-world attack scenarios

6.2 Insights about Gradient Information
- Redundancy in gradients
- Information loss quantification
- Batch size recommendations

## 7. Defenses & Mitigations (400-500 words)
7.1 Existing Defense Mechanisms
- Differential privacy
- Secure aggregation
- Gradient compression

7.2 Limitations of Defenses
- Privacy-utility tradeoff
- Computational overhead

## 8. Discussion & Open Challenges (300-400 words)
- Vanishing objects problem
- Text & digit recovery challenges
- Human face reconstruction
- Future research directions

## 9. Conclusion (200-300 words)
- Summary of contributions
- Importance for FL security
- Call for stronger privacy mechanisms

## 10. References
- Complete bibliography (20-30 papers)

---

## Page Count Breakdown
- Abstract: 0.5 page
- Introduction: 1 page
- Related Work: 1 page
- Methods: 1.5 pages
- Results: 2 pages
- Privacy Implications: 0.5 page
- Defenses: 0.5 page
- Discussion: 0.5 page
- Conclusion: 0.5 page
- **Total: 8 pages (2-column format)**

---

## Key Tables & Figures
- Table 1: Label restoration accuracy
- Table 2: Ablation study
- Table 3: Comparison with SOTA
- Table 4: Main results
- Figure 1: Method overview
- Figure 2: Reconstruction variation
- Figure 3: Group consistency
- Figure 4: Visual comparison
- Figure 5: IIP vs batch size
- Figure 6: Reconstruction quality vs batch size