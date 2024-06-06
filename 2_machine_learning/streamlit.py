import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")



# Dataset Reading and Pre-Processing steps
st.header("Dataset Reading and Pre-Processing")
st.markdown("""
**Attribute information:**
1. **target**: DIE (1), LIVE (2)
2. **age**: 10, 20, 30, 40, 50, 60, 70, 80
3. **gender**: male (1), female (2)
4. **steroid**: no, yes 
5. **antivirals**: no, yes 
6. **fatique**: no, yes 
7. **malaise**: no, yes 
8. **anorexia**: no, yes 
9. **liverBig**: no, yes 
10. **liverFirm**: no, yes 
11. **spleen**: no, yes 
12. **spiders**: no, yes
13. **ascites**: no, yes 
14. **varices**: no, yes
15. **histology**: no, yes
16. **bilirubin**: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
17. **alk**: 33, 80, 120, 160, 200, 250
18. **sgot**: 13, 100, 200, 300, 400, 500
19. **albu**: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
20. **protime**: 10, 20, 30, 40, 50, 60, 70, 80, 90
""")

