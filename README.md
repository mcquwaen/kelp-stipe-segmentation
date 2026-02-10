# kelp-stipe-segmentation
Deep learning–based semantic segmentation framework for kelp stipe detection and quantification in underwater imagery.

# Overview
This repository contains the core implementation developed for the Master’s thesis investigating automated kelp stipe quantification using convolutional neural network–based semantic segmentation.
The framework is designed to:
	•	Train a U-Net segmentation model on annotated underwater imagery
	•	Evaluate model performance using standard segmentation metrics
	•	Generate qualitative visualisations of segmentation outputs
	•	Quantify kelp stipes from predicted segmentation masks
The methodology, data preparation procedures, model architecture, and evaluation framework are described in detail within the associated thesis (see Sections 2.1–2.3.6).

# Repository Structure
The repository includes the primary scripts used in model development and analysis:
	•	train_unet.py
Implements model training, including data loading, augmentation, optimisation, and checkpoint saving.
	•	evaluate_model.py
Computes segmentation performance metrics (e.g., Dice coefficient, IoU, precision, recall) on validation and test datasets.
	•	count_stipes.py
Post-processing workflow for translating segmentation masks into biologically meaningful kelp stipe count estimates.
	•	make_qual_figures.py
Generates qualitative visualisations comparing input imagery, ground truth masks, and predicted outputs.

# Requirements
To install dependencies using: pip install -r requirements.txt
The framework was developed and tested using Python 3.10. Specific package versions are listed in requirements.txt to facilitate reproducibility.
It is recommended to use a virtual environment to avoid dependency conflicts.

# Data
Due to data access restrictions and ethical considerations, the raw underwater imagery and expert-annotated segmentation masks are not publicly distributed through this repository.
The scripts assume a directory structure containing:
	•	Raw RGB underwater images
	•	Corresponding ground truth segmentation masks
Users should adapt file paths within scripts to match their local directory configuration.
Details of dataset construction, annotation procedures, and preprocessing workflows are described in Sections 2.1 and 2.2 of the thesis.

# Reproducibility Statement
This repository provides the complete computational pipeline used for segmentation, evaluation, qualitative analysis, and stipe quantification.
All scripts included here correspond to the implementation described in the thesis Data and Methods chapter. Results reported in the thesis were generated using these scripts, with parameter configurations as documented therein.









