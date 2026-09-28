# AI-Driven SDN Intrusion Detection System for Real-Time Network Security

MSc Cybersecurity and Digital Forensics Project

## Project Overview

This project implements an AI-driven Network Intrusion Detection System (NIDS) for a Software-Defined Networking (SDN) environment.

The system combines:

- Software-Defined Networking (SDN)
- Mininet network emulation
- Open vSwitch (OVS)
- POX/OpenFlow controller
- CICIDS2017 intrusion-detection dataset
- Machine Learning classification
- Behavioural port-scan detection
- Real-time flow monitoring
- Controller-supported mitigation
- Streamlit monitoring dashboard
- Wireshark traffic validation

## System Architecture

Traffic is generated inside an emulated enterprise SDN network.

Mininet Hosts → Open vSwitch → POX Controller → Flow Statistics → Feature Adaptation → ML/Behavioural Detection → Alert → Controller-Supported Mitigation

## Machine Learning

Two supervised machine-learning approaches were evaluated:

### SGD Baseline
Validation Accuracy: 77.98%

### Extra Trees
Validation Accuracy: 97.61%

Held-out Test Accuracy: 97.58%

The final Extra Trees model uses 70 CICIDS2017 features.

## Real-Time Detection

Live OpenFlow monitoring provides fewer features than the offline CICIDS2017 dataset.

31 of the 70 model features were mapped or derived from the live SDN environment.

Therefore, the final system uses a hybrid approach:

- Extra Trees ML classification
- Behavioural flow aggregation for reconnaissance/port-scan detection

## Port-Scan Detection

The behavioural detector monitors unique destination ports contacted by a source within a defined time window.

During testing, the system successfully detected controlled port-scanning activity and generated a mitigation request.

## Mitigation

The system requests temporary controller-supported blocking of the detected malicious source.

Configured mitigation duration:

600 seconds (10 minutes)

Functional testing showed that attacker connectivity was interrupted while legitimate host connectivity remained available.

Direct verification of the expected OpenFlow DROP rule in the OVS flow table was not independently confirmed, so mitigation is reported as functionally supported rather than conclusively verified at the switch-rule level.

## Project Structure

- `src/data/` - CICIDS2017 preparation and splitting
- `src/models/` - SGD and Extra Trees training/evaluation
- `src/realtime/` - real-time feature adaptation and intrusion detection
- `src/dashboard/` - Streamlit monitoring dashboard
- `topology/` - Mininet enterprise topology
- `controller/` - controller-side project components
- `outputs/` - trained models, evaluation metrics and selected results
- `output/wireshark/` - Wireshark validation evidence

## Dataset

CICIDS2017 was used for offline model development and evaluation.

The original dataset, processed dataset and train/validation/test CSV files are not included in this repository because of their large file sizes.

## Key Results

- SGD validation accuracy: 77.98%
- Extra Trees validation accuracy: 97.61%
- Extra Trees held-out test accuracy: 97.58%
- Live feature mapping: 31/70 features
- Behavioural port-scan detection: 86+ unique ports within 10 seconds
- Temporary mitigation request: 600 seconds

## Technologies

Python, Scikit-learn, Pandas, Mininet, Open vSwitch, POX, OpenFlow, Streamlit, Wireshark and CICIDS2017.

## Research Artefact

This repository contains the implementation artefact developed for the MSc project:

**AI-Driven SDN Intrusion Detection System for Real-Time Network Security**
