# Depression Recognition — Related Work Reference List

This file collects real literature relevant to the three Related Work subsections of the AFGNN paper.
Entries are grouped by topic; **★** marks papers already cited in `overleaf_upload/AFGNN.tex`.

---

## 1. Depression Detection from Multimodal Video / Vlogs

### ★ Foundational / already cited

**Yoon et al., 2022** — *D-vlog: Multimodal Vlog Dataset for Depression Detection*  
Proc. AAAI Conf. Artif. Intell., 36(11), 12226–12234.  
The D-Vlog dataset paper; introduces 68-point facial landmarks + 25 openSMILE descriptors for 961 vlogs.

**Zhou et al., 2023** — *TAMFN: Time-Aware Attention Multimodal Fusion Network for Depression Detection*  
IEEE Trans. Neural Syst. Rehabil. Eng., 31, 669–679.  
Treats face/audio as frame sequences; uses time-aware attention for multimodal fusion. Reported on D-Vlog.

**Tao et al., 2024** — *DepMSTAT: Multimodal Spatio-Temporal Attentional Transformer for Depression Detection*  
IEEE Trans. Knowl. Data Eng., 36(7), 2956–2966.  
Transformer-based sequence model for depression detection from vlogs.

**Hua et al., 2026** — *ASYM: Multimodal Depression Recognition via Mamba-Enhanced Attentive Feature Fusion*  
Front. Psychiatry, 17, 1704005.  
Recent Mamba-based multimodal fusion method reported on D-Vlog.

### Additional / candidate papers

**Zhou et al., 2023** — *CAIINET: Neural network based on contextual attention and information interaction mechanism for depression detection*  
Digital Signal Processing, 137, 103986, 2023. DOI: 10.1016/j.dsp.2023.103986.  
BiLSTM + contextual attention + local/global information interaction; evaluated on D-Vlog, DAIC-WOZ, MODMA.  
✅ Verified via multiple sources (arXiv citations, dblp, ADS, ResearchGate).

**Cha et al., 2024** — *MOGAM: A Multimodal Object-oriented Graph Attention Model for Depression Detection*  
arXiv:2403.15485. Authors: Junyeop Cha, Seoyun Kim, Dongjae Kim, Eunil Park.  
Builds an object co-occurrence graph from vlog frames and fuses it with visual/metadata features via cross-attention. Also reports D-Vlog results.  
✅ Verified via arXiv abstract page and HTML version.

**He, Bakker & Lew, 2024** — *DPD (DePression Detection) Net: A Deep Neural Network for Multimodal Depression Detection*  
Health Information Science and Systems, 12, 53, 2024. DOI: 10.1007/s13755-024-00311-9. Authors: M. He, E. M. Bakker, M. S. Lew.  
GNN-enhanced Transformer using text, audio, and visual features; cross-dataset evaluation on E-DAIC, Twitter depression dataset, MODMA, and D-vlog.  
✅ Verified via Springer link and citations in other papers.

**Mou et al., 2025** — *Disentangled Representation Learning via Transformer with Graph Attention Fusion for Depression Detection*  
Proc. 1st Int. Workshop on Cognition-oriented Multimodal Affective and Empathetic Computing (ACM ICMI), pp. 20–29. Authors: Luntian Mou, Siqi Zhen, Shasha Mao, Nan Ma.  
Disentangles homogeneous/heterogeneous/noise features and uses cross graph attention fusion; evaluated on D-Vlog and LMVD.  
✅ Verified (BibTeX provided by user).

---

## 2. Graph Neural Networks for Facial Behaviour / Depression

### ★ Foundational / already cited

**Zhong et al., 2019** — *A Graph-Structured Representation with BRNN for Static-based Facial Expression Recognition*  
Proc. 14th IEEE Int. Conf. on Automatic Face and Gesture Recognition (FG), 1–5.  
Early work that builds fully-connected landmark graphs and processes them with BRNNs.

**Liao et al., 2022** — *FERGCN: Facial Expression Recognition Based on Graph Convolution Network*  
Mach. Vis. Appl., 33(3), 40.  
Integrates CNN features with GCN for facial expression recognition.

**Kipf & Welling, 2017** — *Semi-Supervised Classification with Graph Convolutional Networks*  
Proc. Int. Conf. on Learning Representations (ICLR).  
Foundational spectral GCN paper.

**Veličković et al., 2018** — *Graph Attention Networks*  
Proc. Int. Conf. Learn. Represent. (ICLR).  
Foundational GAT paper; learns adaptive edge weights.

### Additional / candidate papers

**Fwa, 2022** — *Fine-grained Detection of Academic Emotions with Spatial Temporal Graph Attention Networks using Facial Landmarks*  
Proc. 14th Int. Conf. on Computer Supported Education (CSEDU), SciTePress. DOI: 10.5220/0010921200003182. Author: Hua Leong Fwa.  
Uses GAT over facial landmarks for spatial modeling and GRU for temporal modeling; supports the claim that GAT is well suited to landmark graphs.  
✅ Verified via SMU repository and SciTePress PDF.

**Guo et al., 2021** — *Deep Neural Networks for Depression Recognition Based on 2D and 3D Facial Expressions Under Emotional Stimulus Tasks*  
Front. Neurosci., 15, 609760. DOI: 10.3389/fnins.2021.609760. Authors: W. Guo, H. Yang, Z. Liu, Y. Xu, B. Hu.  
Constructs facial contour maps from 3D landmarks and AUs; combines DBN and LSTM for depression recognition.  
✅ Verified via Frontiers official page and multiple citations.

**Georgiou et al., 2026** — *Facial Expression Recognition in the Deep Learning Era: A Systematic Multi-Criteria Review of Methods, Models, Datasets, Performance, Challenges, and Future Research Directions*  
arXiv preprint arXiv:2606.08612. Authors: Spyridon Georgiou, Aggelos Psiris, Spyridon Evangelatos, Thomas Lagkas, Vasileios Argyriou, Panagiotis Sarigiannidis, Iraklis Varlamis, Georgios Th. Papadopoulos.  
Comprehensive review of deep learning-based FER including GNN-based methods (landmark/region graphs, AU-correlation graphs, spatio-temporal graphs).  
✅ Verified (BibTeX provided by user).

---

## 3. Cross-Modal Fusion

### ★ Foundational / already cited

**Tsai et al., 2019** — *Multimodal Transformer for Unaligned Multimodal Language Sequences*  
Proc. 57th ACL, 6558–6569.  
Introduces directional cross-modal attention for unaligned multimodal sequences.

### Additional / candidate papers

**Dhanith et al., 2024** — *Multimodal Emotion Recognition using Audio-Video Transformer Fusion with Cross Attention (AVT-CA)*  
arXiv:2407.18552. Authors: Joe Dhanith P R, Shravan Venkatraman, Modigari Narendra, Vigya Sharma, Santhosh Malarvannan, Amir H. Gandomi.  
Proposes intermediate transformer fusion + cross-attention for audio-visual emotion recognition; evaluated on RAVDESS, CMU-MOSEI, CREMA-D.  
✅ Verified via arXiv abstract page and GitHub repository.

**Gnana Praveen & Alam, 2024** — *Cross-Modal Transformers for Audio-Visual Person Verification*  
The Speaker and Language Recognition Workshop (Odyssey 2024), ISCA, pp. 240–246. Authors: R. Gnana Praveen, Jahangir Alam.  
Compares feature concatenation, self-attention, cross-attention, and Cross-Modal Transformers; shows cross-attention outperforms simple concatenation.  
✅ Verified via ISCA Odyssey 2024 archive and author homepage.

---

## Suggested Priority for Citation Updates

If you want to strengthen Related Work with minimal additions, the highest-impact inserts are:

1. **Cha et al., 2024 (MOGAM)** — most related recent vlog + GNN + cross-attention work.
2. **Fwa, 2022** — directly supports GAT on facial landmarks.
3. **Dhanith et al., 2024 (AVT-CA)** or **Gnana Praveen & Alam, 2024** — supports cross-modal attention fusion.
4. **Georgiou et al., 2026** — useful as a review/survey citation in the GNN subsection.
