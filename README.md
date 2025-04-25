[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zjSXGKeR)
# Our Great Package

**Team**: [Soul - Finders]

## Overview

Welcome to our very different project of wind turbine modeling. The project aims to develop a sophisticated Blade Element Momentum (BEM) model to predict the aerodynamic performance of wind turbines, specifically focusing on the IEA 15 MW offshore reference turbine. This package utilizes advanced aerodynamics and modeling techniques to assist in the design, analysis, and optimization of wind turbine blades. 

The work integrates the Blade Element Momentum theory with detailed turbine data, including geometry, airfoils, and polar data, to provide accurate and efficient predictions for turbine performance.

## Quick-start guide

### 1. **Clone the repository** 

   To get started with this project, clone the repository to your local machine using:
   ```bash
   git clone https://github.com/YourUsername/your-repository-name.git

### 2. **Install dependencies**

Ensure that you have the required libraries and packages installed. You can do this by running:
```bash
pip install -r requirements.txt
```

### 3. **Run the model**

Once everything is set up, you can run the main script to start modeling the wind turbine blades:
```bash
python main.py
```

### 4. **Output**

The script will output visualizations of the blade parameters, including rotor design characteristics, aerodynamic forces, and other relevant data.

### 5. **Customize inputs**

You can modify the input files for different wind turbine models, airfoil shapes, or rotor configurations. The default settings are based on the IEA 15 MW offshore reference turbine.

## Architecture

This project is designed with modularity and flexibility in mind. It consists of the following main components:

### 1. `data_loader.py`
Responsible for importing the geometry, airfoil shapes, and polar data from external files. It parses and stores the information needed for the aerodynamic calculations.

### 2. `bem_model.py`
Contains the core of the Blade Element Momentum theory. This module computes the forces and performance metrics of the wind turbine based on the input data.

### 3. `analysis.py`
Used for performing advanced calculations on the blade parameters and visualizing the aerodynamic forces, such as lift, drag, and thrust at each blade element.

### 4. `visualization.py`
Generates visualizations of the blade's geometry, aerodynamic performance, and other key metrics for analysis and optimization purposes.

### 5. `utils.py`
Includes helper functions that support various operations, such as data conversion, interpolation, and performance calculations.


## Peer review

Our team followed a structured peer review process to ensure code quality, maintainability, and correctness:

- **Code Reviews:** Each member reviewed another teammate’s module, checking for logical consistency, readability, and documentation.
- **Testing:** We tested each module independently and in integration to confirm the accuracy of calculations and data handling.
- **Feedback Loop:** Constructive feedback was shared via Git, ensuring timely fixes and improvements.
- **Version Control:** All changes were tracked using Git branches and pull requests, enabling collaborative and conflict-free development.

This approach helped us maintain high standards and align our code with the project's objectives.
