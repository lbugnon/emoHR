# SVR quadprog

A simple script to train a Support Vector Machine for Regression (SVR) using MATLAB quadratic programming functions. 

Modified from original version of Ronny Clark (https://uk.mathworks.com/matlabcentral/fileexchange/43429-support-vector-regression), optimizing kernel and matrix operations for a better performance.

---
Example usage:
```
svrobj = svr_trainer(x_train,y_train,400,0.000000025,'gaussian',0.5);
y = svrobj.predict(x_test);
```
---
sinc(i) - http://fich.unl.edu.ar/sinc/.
Leandro Bugnon (lbugnon@sinc.unl.edu.ar)
