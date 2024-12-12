# 3D and 3D+t Images: Segmentation of Cardiac MRI
This project, part of my studies in Imaging at Télécom Paris, focuses on the segmentation of the left ventricle in cardiac MRI images from 150 patients. The goal is to apply fundamental image processing techniques without using AI to extract and analyze relevant anatomical structures. This project is a collaboration with my partner, [Mamannne](https://github.com/Mamannne), and together, we explore essential techniques in medical image analysis.  
## **Project Objective**
Certain pathologies leading to heart failure cause remodeling of the heart. This is due to the myocardium's response to the various insults occurring during these syndromes. An important point in computer-assisted diagnosis is the segmentation of cardiac anatomy to simplify the analysis of the heart's structure. The objective of our project is the segmentation of the left ventricle cavity on MRI data from each patient.
## **Studied Population**  
The target population for this study consists of 150 patients divided into 5 subgroups as follows:

- **30 normal subjects** – **NOR**
- **30 patients with a history of myocardial infarction**  
  (left ventricular ejection fraction < 40% and multiple myocardial segments with abnormal contraction) – **MINF**
- **30 patients with dilated cardiomyopathy**  
  (left ventricular end-diastolic volume > 100 mL/m² and left ventricular ejection fraction < 40%) – **DCM**
- **30 patients with hypertrophic cardiomyopathy**  
  (left ventricular mass > 110 g/m², multiple myocardial segments with a thickness > 15 mm in diastole, and normal ejection fraction) – **HCM**
- **30 patients with an abnormal right ventricle**  
  (right ventricular cavity volume > 110 mL/m² or right ventricular ejection fraction < 40%) – **RV**

## **File Structure and Explanation**
A research paper on the topic is available in French under the name [Final_Report.pdf](./Final_Report.pdf), along with a Project folder containing a `main.ipynb` file where the code and usage examples can be found, a `logs.json` file providing the DICE results for the 100 training patients, and two other JSON files that help optimize the computation time, as explained in the final report.


