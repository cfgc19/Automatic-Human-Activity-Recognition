# Automatic-Human-Activity-Recognition

In this project were developed several classifiers for Human Activity Recognition (HAR) using data collected from 30 subjects performing activities of daily living while carrying a waist-mounted smartphone with embedded inertial sensors.

Are considered two scenarios:

 • Scenario A (Binary Problem): where one where one want to discriminate if a given person is walking or not;

 • Scenario B (Multiclass Problem): where differentiation among six different activities are required (walking, walking upstairs, walking downstairs, sitting, standing and laying).

showGui_RP.py -> creation of a graphical user interface

OpenDocs.py -> data preprocessing; feature selection and feature reduction; classification

functions_RP.py -> functions invoked in OpenDocs.py

All the steps included in this project (Data preprocessing, Feature Reduction(PCA & LDA), Feature Selection, Classifiers) are described in report.pdf

