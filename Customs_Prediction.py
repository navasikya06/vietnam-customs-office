"""
Ngay 8-9-2017
"""

import pyodbc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import matplotlib.mlab as mlab

# from scipy import stats
# from scipy.stats import norm


"""
A. Cau truc chuong trinh:
1. Doc tung chuoi tu bang dactachuoi va chuoi vao bien array
2. Ve do thi
2.1. Co 16 module ve do thi, moi module ung voi chuoi goc va 1 to hop cac chuoi tu 4 mo hinh
     (bac1, tuyen tinh, mua vu cong va mua vu nhan)
2.2. Moi module ve 1 do thi "line" cua chuoi goc la chuoi lam tron cua cac mo hinh
3. Xay dung mo hinh: chon 1 trong 4 mo hinh: 
3.1. Bac 1: co 2 lua chon
3.1.1. Tu dong: chuong trinh tu dong chon alpha toi uu
       + Voi y la array doc duoc, goi alpha_1f(y) de tinh alpha_1 toi uu
         (cho alpha_1 chay tu 0.05 den 0.3, buoc 0.01, moi buoc goi SSE_1f(y,alpha_1) de tinh SSE tuong ung,
          chon alpha_1 co SSE dat min)
       + Goi bac1(y,alpha_1) de tinh [y_1,e_1,bt_1,SSE_1,muy_1,xichma_1] (chuoi lam tron, error, bat thuong, SSE, muy, xichma)
3.1.2. Chuyen gia
       + Chuyen gia chon alpha_1
       + Goi bac1(y,alpha_1) de tinh [y_1,e_1,bt_1,SSE_1,muy_1,xichma_1] (chuoi lam tron, error, bat thuong, SSE, muy, xichma)
3.2. Tuyen tinh: co 2 lua chon
3.2.1. Tu dong: chuong trinh tu dong chon alpha, beta toi uu
       + Voi y la array doc duoc, goi alphabeta_f(y) de tinh alpha_tt, beta_tt toi uu
         (cho alpha_tt va beta_tt chay long nhau tu 0.02 den 0.2, buoc 0.01, moi buoc goi SSE_ttf(y,alpha_tt,beta_tt) de tinh SSE tuong ung,
          chon alpha_tt va beta_tt co SSE dat min)
       + Goi tt(y,alpha_tt,beta_tt) de tinh [y_tt,e_tt,bt_tt,SSE_tt,muy_tt,xichma_tt] (chuoi lam tron, error, bat thuong, SSE, muy, xichma)
3.2.2. Chuyen gia
       + Chuyen gia chon alpha_tt, beta_tt
       + Goi tt(y,alpha_tt,beta_tt) de tinh [y_tt,e_tt,bt_tt,SSE_tt,muy_tt,xichma_tt] (chuoi lam tron, error, bat thuong, SSE, muy, xichma)
3.3. Mua vu cong: co 2 lua chon
3.3.1. Tu dong: chuong trinh tu dong chon alpha, beta, gama toi uu
       + Voi y la array doc duoc, goi abg_congf(y,m) de tinh alpha_cong, beta_cong, gama_cong toi uu
         (cho alpha_cong va beta_cong va gama_cong chay long nhau tu 0.02 den 0.2, buoc 0.01, moi buoc goi SSE_congf(y,alpha_cong,beta_cong,gama_cong,m) de tinh SSE tuong ung,
          chon alpha_cong va beta_cong va gama_cong co SSE dat min)
       + Goi cong(y,alpha_cong,beta_cong,gama_cong,m) de tinh [y_cong,e_cong,bt_cong,SSE_cong,muy_cong,xichma_cong] (chuoi lam tron, error, bat thuong, SSE, muy, xichma)
3.3.2. Chuyen gia
       + Chuyen gia chon alpha_cong, beta_cong, gama_cong
       + Goi cong(y,alpha_cong,beta_cong,gama_cong) de tinh [y_cong,e_cong,bt_cong,SSE_cong,muy_cong,xichma_cong] (chuoi lam tron, error, bat thuong, SSE, muy, xichma)
3.4. Mua vu nhan: co 2 lua chon
3.4.1. Tu dong: chuong trinh tu dong chon alpha, beta, gama toi uu
       + Voi y la array doc duoc, goi abg_nhanf(y,m) de tinh alpha_nhan, beta_nhan, gama_nhan toi uu
         (cho alpha_nhan va beta_nhan va gama_nhan chay long nhau tu 0.02 den 0.2, buoc 0.01, moi buoc goi SSE_nhanf(y,alpha_nhan,beta_nhan,gama_nhan,m) de tinh SSE tuong ung,
          chon alpha_nhan va beta_nhan va gama_nhan co SSE dat min)
       + Goi nhan(y,alpha_nhan,beta_nhan,gama_nhan,m) de tinh [y_nhan,e_nhan,bt_nhan,SSE_nhan,muy_nhan,xichma_nhan] (chuoi lam tron, error, bat thuong, SSE, muy, xichma)
3.4.2. Chuyen gia
       + Chuyen gia chon alpha_nhan, beta_nhan, gama_nhan
       + Goi nhan(y,alpha_nhan,beta_nhan,gama_nhan) de tinh [y_nhan,e_nhan,bt_nhan,SSE_nhan,muy_nhan,xichma_nhan] (chuoi lam tron, error, bat thuong, SSE, muy, xichma)

B. Thu vien dung chung
1. Bac 1:
bac1(y,alpha_1)
SSE_1f(y,alpha_1)
alpha_1f(y)
2. Tuyen tinh:
tt(y,alpha_tt,beta_tt)
SSE_ttf(y,alpha_tt,beta_tt)
alphabeta_f(y)
3. Cong:
cong(y,alpha_cong,beta_cong,gama_cong,m)
SSE_congf(y,alpha_cong,beta_cong,gama_cong_f,m)
abg_congf(y,m)
4. Nhan:
nhan(y,alpha_nhan,beta_nhan,gama_nhan,m)
SSE_nhanf(y,alpha_nhan,beta_nhan,gama_nhan,m)
abg_nhanf(y,m)

Chu y:
Truyen tham so theo nguyen tac REFERENCE. Tham so thu tuc def va tham so goi phai giong nhau hoan toan
Moi array deu co chi so tu 0, de phu hop model nen khong tinh array[0]
"""

def bac1(y, alpha_1):
    """
    Input: bien mang y va alpha_1
    Output: y_1,e_1,bt_1,SSE_1,muy_1,xichma_1
    Bo chi so 0 dau tien
    """
    n = len(y) # n=91 test HQ 8/9/2017, chuoi thoi gian 90 phan tu 
    nn=n-1
    y_1 = np.zeros(n)
    e_1 = np.zeros(n)
    bt_1 = np.zeros(n)
    yd_1 = np.zeros(n+1)

    # Chay mo hinh
    y_1[1] = y[1]
    yd_1[2]=y_1[1]
    #chi so i tu 2 den n-1
    for i in range(2, n): 
        y_1[i] = alpha_1 * y[i] + (1 - alpha_1) * y_1[i - 1]
        yd_1[i+1]=y_1[i]

    # Tinh cac SSE va muy va e
    SSE_1 = 0.0
    muy_1 = 0.0
    #chi so i tu 2 den n-1
    for i in range(2, n): 
        e_1[i] = y[i] - yd_1[i]
        SSE_1 = SSE_1+(e_1[i]**2)
        muy_1 = muy_1 + e_1[i]
    muy_1 = muy_1/(nn-1)

    # Tinh xichma
    xichma_1 = 0.0
    #chi so i tu 2 den n-1
    for i in range(2, n): 
        xichma_1 = xichma_1+(e_1[i]-muy_1)**2
    xichma_1 = math.sqrt(xichma_1/(nn-1))

    # Tinh error phan bo chuan
    #chi so i tu 2 den n-1
    for i in range(2, n):
        if e_1[i] > (muy_1+(2*xichma_1)) or e_1[i] < (muy_1-(2*xichma_1)):
            bt_1[i] = 1 # BT phan phoi chuan

    # Tinh loi IQR (Interquartile Range)
    e = np.zeros(n-2)
    for i in range(2, n):
        e[i-2]=e_1[i]
    ee=np.array(e)
    q1=np.percentile(ee,25)
    q3=np.percentile(ee,75)
    iqr=q3-q1
    for i in range(2, n):
        if (e_1[i] > q3+(1.5*iqr)) or (e_1[i] < q1-(1.5*iqr)):
            if bt_1[i]==1:
                bt_1[i] = 3 # Ca 2 cach phat hien BT
            else:
                bt_1[i] = 2 # BT IQR
        
    return y_1, e_1, bt_1, yd_1, SSE_1, muy_1, xichma_1

def SSE_1f(y, alpha_1):
    """
    Gan tuong tu bac1, nhung khong tinh muy, xichma va bt
    Input: bien mang y va alpha_1
    Output: SSE_1
    Bo chi so 0 dau tien
    """
    n = len(y)
    y_1 = np.zeros(n)
    e_1 = np.zeros(n)
    yd_1 = np.zeros(n+1)
    y_1[1] = y[1]
    yd_1[2]=y_1[1]
    for i in range(2, n):
        y_1[i] = alpha_1 * y[i] + (1 - alpha_1) * y_1[i - 1]
        yd_1[i+1]=y_1[i]
    SSE_1 = 0.0
    # Tinh SSE
    for i in range(2, n):
        e_1[i] = y[i]-yd_1[i]
        SSE_1 = SSE_1+(e_1[i]**2)
    return SSE_1

def alpha_1f(y):
    """
    Input: bien mang y
    Output: alpha toi uu trong [0.05,0.3], buoc 0.01
    """
    SSE_1op = 1.e+30
    alpha_op = 0.5
    alpha_1 = 0.01
    while alpha_1 <= 1:
        SSE_1 = SSE_1f(y, alpha_1)
        if SSE_1 < SSE_1op:
            alpha_op = alpha_1
            SSE_1op = SSE_1
        alpha_1 = alpha_1 + 0.01
    return alpha_op


def tt(y, alpha_tt, beta_tt):
    """
    Input: bien mang y va alpha_tt,beta_tt
    Output: y_tt,e_tt,bt_tt,SSE_tt,muy_tt,xichma_tt
    Bo chi so 0 dau tien
    """
    n = len(y)
    nn=n-1
    y_tt = np.zeros(n)
    e_tt = np.zeros(n)
    b = np.zeros(n)
    bt_tt = np.zeros(n)
    yd_tt = np.zeros(n+1)
    y_tt[1] = y[1]
    b[1] = y[2] - y[1]
    yd_tt[2]=y_tt[1]+b[1]
    
    # Chay mo hinh
    # chi so i tu 2 den n-1
    for i in range(2, n):
        y_tt[i] = alpha_tt * y[i] + (1 - alpha_tt) * (y_tt[i - 1] + b[i - 1])
        b[i] = beta_tt * (y_tt[i] - y_tt[i - 1]) + (1 - beta_tt) * b[i - 1]
        yd_tt[i+1]=y_tt[i]+b[i]

    # Tinh SSE va muy va e  
    SSE_tt = 0.0
    muy_tt = 0.0           
    # chi so i tu 2 den n-1
    for i in range(2, n):
        e_tt[i] = y[i] - yd_tt[i]
        SSE_tt = SSE_tt + (e_tt[i]**2)
        muy_tt = muy_tt + e_tt[i]
    muy_tt = muy_tt/(nn-1)
    
    # Tinh xichma
    xichma_tt = 0.0           
    # chi so i tu 2 den n-1
    for i in range(2, n):
        xichma_tt = xichma_tt+(e_tt[i]-muy_tt)**2
    xichma_tt = math.sqrt(xichma_tt/(nn-1))
    
    # Tinh error           
    # chi so i tu 2 den n-1
    for i in range(2, n):
        if e_tt[i] > (muy_tt + 2 * xichma_tt) or e_tt[i] < (muy_tt - 2 * xichma_tt):
            bt_tt[i] = 1 # BT phan phoi chuan

    # Tinh loi IQR (Interquartile Range)
    e = np.zeros(n-2)
    for i in range(2, n):
        e[i-2]=e_tt[i]
    ee=np.array(e)
    q1=np.percentile(ee,25)
    q3=np.percentile(ee,75)
    iqr=q3-q1
    for i in range(2, n):
        if (e_tt[i] > q3+(1.5*iqr)) or (e_tt[i] < q1-(1.5*iqr)):
            if bt_tt[i]==1:
                bt_tt[i] = 3 # Ca 2 cach phat hien BT
            else:
                bt_tt[i] = 2 # BT IQR
            
    return y_tt, e_tt, bt_tt, yd_tt, SSE_tt, muy_tt, xichma_tt

def SSE_ttf(y, alpha_tt, beta_tt):
    """
    Gan tuong tu tt, nhung khong tinh muy, xichma va bt
    Input: bien mang y va alpha_1
    Output: SSE_tt
    Bo chi so 0 dau tien
    """
    n = len(y)
    y_tt = np.zeros(n)
    b = np.zeros(n)
    e_tt = np.zeros(n)
    yd_tt = np.zeros(n+1)
    y_tt[1] = y[1]
    b[1] = y[2] - y[1]
    yd_tt[2]=y_tt[1]+b[1]
    # Tinh SSE
    for i in range(2, n):
        y_tt[i] = alpha_tt * y[i] + (1 - alpha_tt) * (y_tt[i - 1] + b[i - 1])
        b[i] = beta_tt * (y_tt[i] - y_tt[i - 1]) + (1 - beta_tt) * b[i - 1]
        yd_tt[i+1]=y_tt[i]+b[i]
    SSE_tt = 0.0
    for i in range(2, n):
        e_tt[i] = y[i] - yd_tt[i]
        SSE_tt = SSE_tt + e_tt[i] ** 2
    return SSE_tt

def alphabeta_f(y):
    """
    Input: bien mang y
    Output: alpha, beta toi uu trong [0.02,0.2], buoc 0.01
    """
    SSE_ttop = 1.e+30
    alpha_op=0.1
    beta_op=0.1
    alpha_tt = 0.02
    while alpha_tt <= 0.2:        
        beta_tt = 0.02
        while beta_tt <= 0.2: 
            SSE_tt = SSE_ttf(y, alpha_tt, beta_tt)
            if SSE_tt < SSE_ttop:
                alpha_op = alpha_tt
                beta_op = beta_tt
                SSE_ttop = SSE_tt
            beta_tt = beta_tt + 0.01
        alpha_tt = alpha_tt + 0.01
    return alpha_op, beta_op


def cong(y, alpha_cong, beta_cong, gama_cong, m):
    """
    Input: bien mang y va alpha, beta, gama, m=4/12
    Output: y_cong,e_cong,bt_cong,SSE_cong,muy_cong,xichma_cong
    """
    n = len(y)
    nn=n-1
    y_cong = np.zeros(n)
    b = np.zeros(n)
    s = np.zeros(n)    
    e_cong = np.zeros(n)
    bt_cong = np.zeros(n)    
    yd_cong = np.zeros(n+1)
    y_cong[m] = 0
    # chi so i tu 1 den m
    for i in range(1, m+1): 
        y_cong[m] = y_cong[m] + y[i]
    if y_cong[m]==0:
        SSE_cong=-1
        muy_cong=-1
        xichma_cong=-1
        return y_cong, e_cong, bt_cong, yd_cong, SSE_cong, muy_cong, xichma_cong
    y_cong[m] = y_cong[m] / m
    b[m] = 0
    # chi so i tu 1 den m
    for i in range(1, m+1):
        s[i] = y[i] / y_cong[m]
    yd_cong[m+1] = y_cong[m] + b[m] + s[m-m+1]
    
    # Chay mo hinh
    # chi so i tu 2 den n-1
    for i in range(m + 1, n):
        y_cong[i] = alpha_cong * (y[i] - s[i-m]) + (1-alpha_cong) * (y_cong[i-1] + b[i-1])
        b[i] = beta_cong * (y_cong[i] - y_cong[i-1]) + (1-beta_cong) * b[i-1]
        s[i] = gama_cong * (y[i] - y_cong[i]) + (1 - gama_cong) * s[i-m]
        yd_cong[i+1] = y_cong[i] + b[i] + s[i-m+1]
        
    # Tinh SSE va muy va e           
    SSE_cong = 0.0
    muy_cong = 0.0
    # chi so i tu m+1 den n-1
    for i in range(m+1, n):
        e_cong[i] = y[i] - yd_cong[i]
        SSE_cong = SSE_cong + (e_cong[i]**2)
        muy_cong = muy_cong + e_cong[i]
    muy_cong = muy_cong/(nn-m)
    
    # Tinh xichma
    xichma_cong = 0.0
    # chi so i tu m+1 den n-1
    for i in range(m + 1, n):
        xichma_cong = xichma_cong + ((e_cong[i]-muy_cong)**2)
    xichma_cong = math.sqrt(xichma_cong/(nn-m))
    
    # Tinh error
    # chi so i tu m+1 den n-1
    for i in range(m + 1, n):
        if e_cong[i] > (muy_cong + (2*xichma_cong)) or e_cong[i] < (muy_cong - (2*xichma_cong)):
            bt_cong[i] = 1 # BT phan phoi chuan

    # Tinh loi IQR (Interquartile Range)
    e = np.zeros(n-m-1)
    for i in range(m+1, n):
        e[i-m-1]=e_cong[i]
    ee=np.array(e)
    q1=np.percentile(ee,25)
    q3=np.percentile(ee,75)
    iqr=q3-q1
    for i in range(m+1, n):
        if (e_cong[i] > q3+(1.5*iqr)) or (e_cong[i] < q1-(1.5*iqr)):
            if bt_cong[i]==1:
                bt_cong[i] = 3 # Ca 2 cach phat hien BT
            else:
                bt_cong[i] = 2 # BT IQR
            
    return y_cong, e_cong, bt_cong, yd_cong, SSE_cong, muy_cong, xichma_cong

def SSE_congf(y, alpha_cong, beta_cong, gama_cong, m):
    """
    Input: bien mang y va alpha, beta, gama, m=4/12
    Output: SSE_cong
    """
    n = len(y)
    y_cong = np.zeros(n)
    b = np.zeros(n)
    s = np.zeros(n)  
    e_cong = np.zeros(n)
    yd_cong = np.zeros(n+1)
    y_cong[m] = 0
    for i in range(1, m + 1):
        y_cong[m] = y_cong[m] + y[i]
    if y_cong[m]==0:
        SSE_cong=1.e+30
        return SSE_cong    
    y_cong[m] = y_cong[m] / m
    b[m] = 0
    for i in range(1, m + 1):
        s[i] = y[i] / y_cong[m]
    yd_cong[m+1] = y_cong[m] + b[m] + s[m-m+1]
    
    # Chay mo hinh
    for i in range(m + 1, n):
        y_cong[i] = alpha_cong * (y[i] - s[i-m]) + (1 - alpha_cong) * (y_cong[i-1] + b[i-1])
        b[i] = beta_cong * (y_cong[i] - y_cong[i-1]) + (1 - beta_cong) * b[i-1]
        s[i] = gama_cong * (y[i] - y_cong[i]) + (1 - gama_cong) * s[i-m]
        yd_cong[i+1] = y_cong[i] + b[i] + s[i-m+1]

    # Tinh SSE           
    SSE_cong = 0.0
    for i in range(m+1, n):
        e_cong[i] = y[i] - yd_cong[i]
        SSE_cong = SSE_cong + (e_cong[i]**2)
    return SSE_cong

def abg_congf(y, m):
    """
    Input: bien mang y
    Output: alpha, beta, gama toi uu trong [0.02,0.2], buoc 0.01
    """
    SSE_congop = 1.e+30
    alpha_op=0.2
    beta_op=0.2
    gama_op=0.2
    alpha_cong = 0.02
    while alpha_cong <= 0.2:
        beta_cong = 0.02
        while beta_cong <= 0.2:
            gama_cong = 0.02
            while gama_cong <= 0.2:
                SSE_cong = SSE_congf(y, alpha_cong, beta_cong, gama_cong, m)
                if SSE_cong < SSE_congop:
                    alpha_op = alpha_cong
                    beta_op = beta_cong
                    gama_op = gama_cong
                    SSE_congop = SSE_cong
                gama_cong = gama_cong + 0.01
            beta_cong = beta_cong + 0.01
        alpha_cong = alpha_cong + 0.01
    return alpha_op, beta_op, gama_op


def nhan(y, alpha_nhan, beta_nhan, gama_nhan, m):
    """
    Input: bien mang y va alpha, beta, gama, m=4/12
    Output: y_nhan,e_nhan,bt_nhan,SSE_nhan,muy_nhan,xichma_nhan
    """
    n = len(y)
    nn=n-1
    y_nhan = np.zeros(n)
    b = np.zeros(n)
    s = np.zeros(n)    
    e_nhan = np.zeros(n)
    bt_nhan = np.zeros(n)    
    yd_nhan = np.zeros(n+1)
    y_nhan[m] = 0
    # chi so chay tu 1 den m
    for i in range(1, m+1):
        y_nhan[m] = y_nhan[m] + y[i]
    if y_nhan[m]==0:
        SSE_nhan=-1
        muy_nhan=-1
        xichma_nhan=-1
        y_nhan = np.zeros(n)
        e_nhan = np.zeros(n)
        bt_nhan = np.zeros(n)    
        yd_nhan = np.zeros(n+1)
        return y_nhan, e_nhan, bt_nhan, yd_nhan, SSE_nhan, muy_nhan, xichma_nhan        
    y_nhan[m] = y_nhan[m] / m
    b[m] = 0
    # chi so i tu 1 den m
    for i in range(1, m+1):
        s[i] = y[i] / y_nhan[m]
    yd_nhan[m+1] = (y_nhan[m] + b[m])*s[m-m+1]
    
    # Chay mo hinh
    # chi so i tu m+1 den n-1
    SSE_nhan=0
    for i in range(m + 1, n):
        if s[i-m]==0:
            SSE_nhan=-1
            muy_nhan=-1
            xichma_nhan=-1
            y_nhan = np.zeros(n)
            e_nhan = np.zeros(n)
            bt_nhan = np.zeros(n)    
            yd_nhan = np.zeros(n+1)
            return y_nhan, e_nhan, bt_nhan, yd_nhan, SSE_nhan, muy_nhan, xichma_nhan
        else:
            y_nhan[i] = alpha_nhan * (y[i]/s[i-m]) + (1-alpha_nhan) * (y_nhan[i-1] + b[i-1])
        b[i] = beta_nhan * (y_nhan[i]- y_nhan[i-1]) + (1-beta_nhan) * b[i-1]
        if (y_nhan[i-1] + b[i-1])==0:
            SSE_nhan=-1
            muy_nhan=-1
            xichma_nhan=-1
            y_nhan = np.zeros(n)
            e_nhan = np.zeros(n)
            bt_nhan = np.zeros(n)    
            yd_nhan = np.zeros(n+1)
            return y_nhan, e_nhan, bt_nhan, yd_nhan, SSE_nhan, muy_nhan, xichma_nhan
        else:
            s[i] = gama_nhan * (y[i]/(y_nhan[i-1] + b[i-1])) + (1-gama_nhan)*s[i-m]
        yd_nhan[i+1] = (y_nhan[i] + b[i])*s[i-m+1]

    # Tinh SSE va muy va e
    SSE_nhan = 0.0
    muy_nhan = 0.0
    # chi so i tu m+1 den n-1
    for i in range(m + 1, n):
        e_nhan[i] = y[i] - yd_nhan[i]
        SSE_nhan = SSE_nhan + (e_nhan[i]**2)
        muy_nhan = muy_nhan + e_nhan[i]
    muy_nhan = muy_nhan / (nn-m)
    
    # Tinh xichma
    xichma_nhan = 0.0
    # chi so i tu m+1 den n-1
    for i in range(m + 1, n):
        xichma_nhan = xichma_nhan + ((e_nhan[i]-muy_nhan)**2)
    xichma_nhan = math.sqrt(xichma_nhan / (nn-m))
    
    # Tinh error
    # chi so i tu m+1 den n-1
    for i in range(m + 1, n):
        if e_nhan[i] > (muy_nhan + 2*xichma_nhan) or e_nhan[i] < (muy_nhan - 2*xichma_nhan):
            bt_nhan[i] = 1 # BT phan phoi chuan

    # Tinh loi IQR (Interquartile Range)
    e = np.zeros(n-m-1)
    for i in range(m+1, n):
        e[i-m-1]=e_nhan[i]
    ee=np.array(e)
    q1=np.percentile(ee,25)
    q3=np.percentile(ee,75)
    iqr=q3-q1
    for i in range(m+1, n):
        if (e_nhan[i] > q3+(1.5*iqr)) or (e_nhan[i] < q1-(1.5*iqr)):
            if bt_nhan[i]==1:
                bt_nhan[i] = 3 # Ca 2 cach phat hien BT
            else:
                bt_nhan[i] = 2 # BT IQR
    return y_nhan, e_nhan, bt_nhan, yd_nhan, SSE_nhan, muy_nhan, xichma_nhan

def SSE_nhanf(y, alpha_nhan, beta_nhan, gama_nhan, m):
    """
    Input: bien mang y va alpha, beta, gama, m=4/12
    Output: SSE_nhan
    """
    n = len(y)
    y_nhan = np.zeros(n)
    b = np.zeros(n)
    s = np.zeros(n) 
    e_nhan = np.zeros(n)
    yd_nhan = np.zeros(n+1)
    y_nhan[m] = 0
    for i in range(1, m+1):
        y_nhan[m] = y_nhan[m] + y[i]
    if y_nhan[m]==0:
        SSE_nhan=1.e+30
        return SSE_nhan
    y_nhan[m] = y_nhan[m] / m
    b[m] = 0
    for i in range(1, m+1):
        s[i] = y[i] / y_nhan[m]
    yd_nhan[m+1] = (y_nhan[m] + b[m])*s[m-m+1]
                            
    # Chay mo hinh
    SSE_nhan=0
    for i in range(m + 1, n):
        if s[i-m]==0:
            SSE_nhan=1.e+30
            return SSE_nhan
        else:
            y_nhan[i] = alpha_nhan * (y[i]/s[i-m]) + (1-alpha_nhan) * (y_nhan[i-1] + b[i-1])
        b[i] = beta_nhan * (y_nhan[i]- y_nhan[i-1]) + (1-beta_nhan) * b[i-1]
        if y[i]/(y_nhan[i-1] + b[i-1])==0:
            SSE_nhan=1.e+30
            return SSE_nhan
        else:
            s[i] = gama_nhan * (y[i]/(y_nhan[i-1] + b[i-1])) + (1-gama_nhan)*s[i-m]
        yd_nhan[i+1] = (y_nhan[i] + b[i])*s[i-m+1]
            
    # Tinh SSE
    SSE_nhan = 0.0
    for i in range(m + 1, n):
        e_nhan[i] = y[i] - yd_nhan[i]
        SSE_nhan = SSE_nhan + (e_nhan[i]**2)
    return SSE_nhan

def abg_nhanf(y, m):
    """
    Input: bien mang y
    Output: alpha, beta, gama toi uu trong [0.02,0.2], buoc 0.01
    """
    SSE_nhanop = 1.e+30
    alpha_op=0.2
    beta_op=0.2
    gama_op=0.2
    alpha_nhan = 0.02
    while alpha_nhan <= 0.2:
        beta_nhan = 0.02
        while beta_nhan <= 0.2:
            gama_nhan = 0.02
            while gama_nhan <= 0.2:
                SSE_nhan = SSE_nhanf(y, alpha_nhan, beta_nhan, gama_nhan, m)
                if (SSE_nhan < SSE_nhanop) and (SSE_nhan<>-1):
                    alpha_op = alpha_nhan
                    beta_op = beta_nhan
                    gama_op = gama_nhan
                    SSE_nhanop = SSE_nhan
                gama_nhan = gama_nhan + 0.01
            beta_nhan = beta_nhan + 0.01
        alpha_nhan = alpha_nhan + 0.01
    return alpha_op, beta_op, gama_op, SSE_nhanop

def ve(y, x, xnhan, ten):
    """
    Ve do thi duong (x,y), truc hoanh hien thi xnhan
    """
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(xnhan)
    # ax.invert_xaxis()  # labels read top-to-bottom
    ax.set_xlabel('Thoi gian')
    ax.set_title(ten)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

def ve1(y, x, xnhan, yd_1, bt_1,ten):
    """
    Ve 2 do thi duong (x,y) va (x,y_1), truc hoanh hien thi xnhan
    """
    #x_bt = [int(i) for i, x in enumerate(x) if bt_1[i] == 1]
    x_bt=[]
    for i in range(2,len(bt_1)):
        if (bt_1[i]==1) or (bt_1[i]==3):
            x_bt.append(i)                   
            
    plt.rcdefaults()
    fig, ax = plt.subplots()
    #line1, = ax.plot(x, y, '-o', markevery=x_bt)
    #line2, = ax.plot(x, y_1, '-o', markevery=x_bt
    line1 = ax.plot(x, y, '-o',  markevery=x_bt, label='Chuoi y')
    line2 = ax.plot(x, yd_1, label='Chuoi yd_1')
    ax.set_xticks(x)
    ax.set_xticklabels(xnhan)
    # ax.invert_xaxis()  # labels read top-to-bottom
    ax.set_xlabel('Thoi gian')
    ax.set_title(ten)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


def ve1_1(y, x, xnhan, yd_1, bt_1,ten):
    """
    Ve 2 do thi duong (x,y) va (x,y_1), truc hoanh hien thi xnhan
    """
    x_bt=[]
    for i in range(2,len(bt_1)):
        if (bt_1[i]==2) or (bt_1[i]==3):
            x_bt.append(i)                   
            
    plt.rcdefaults()
    fig, ax = plt.subplots()
    line1 = ax.plot(x, y, '-o',  markevery=x_bt)
    line2 = ax.plot(x, yd_1)
    ax.set_xticks(x)
    ax.set_xticklabels(xnhan)
    ax.set_xlabel('Thoi gian')
    ax.set_title(ten)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

def ve2(y, x, xnhan, yd_tt, bt_tt,ten):
    """
    Ve 2 do thi duong (x,y) va (x,y_tt), truc hoanh hien thi xnhan
    """
    x_bt=[]
    for i in range(2,len(bt_tt)):
        if (bt_tt[i]==1) or (bt_tt[i]==3):
            x_bt.append(i)                   
            
    plt.rcdefaults()
    fig, ax = plt.subplots()
    line1 = ax.plot(x, y, '-o',  markevery=x_bt)
    line2 = ax.plot(x, yd_tt)
    ax.set_xticks(x)
    ax.set_xticklabels(xnhan)
    ax.set_xlabel('Thoi gian')
    ax.set_title(ten)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

def ve2_1(y, x, xnhan, yd_tt, bt_tt,ten):
    """
    Ve 2 do thi duong (x,y) va (x,y_tt), truc hoanh hien thi xnhan
    """
    x_bt=[]
    for i in range(2,len(bt_tt)):
        if (bt_tt[i]==2) or (bt_tt[i]==3):
            x_bt.append(i)                   
            
    plt.rcdefaults()
    fig, ax = plt.subplots()
    line1 = ax.plot(x, y, '-o',  markevery=x_bt)
    line2 = ax.plot(x, yd_tt)
    ax.set_xticks(x)
    ax.set_xticklabels(xnhan)
    ax.set_xlabel('Thoi gian')
    ax.set_title(ten)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

def ve3(y, x, xnhan, yd_cong, bt_cong,ten):
    """
    Ve 2 do thi duong (x,y) va (x,y_cong), truc hoanh hien thi xnhan
    """
    x_bt=[]
    for i in range(2,len(bt_cong)):
        if (bt_cong[i]==1) or (bt_cong[i]==3):
            x_bt.append(i)                   
            
    plt.rcdefaults()
    fig, ax = plt.subplots()
    line1 = ax.plot(x, y, '-o',  markevery=x_bt)
    line2 = ax.plot(x, yd_cong)
    ax.set_xticks(x)
    ax.set_xticklabels(xnhan)
    ax.set_xlabel('Thoi gian')
    ax.set_title(ten)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

def ve3_1(y, x, xnhan, yd_cong, bt_cong,ten):
    """
    Ve 2 do thi duong (x,y) va (x,y_cong), truc hoanh hien thi xnhan
    """
    x_bt=[]
    for i in range(2,len(bt_cong)):
        if (bt_cong[i]==2) or (bt_cong[i]==3):
            x_bt.append(i)                   
            
    plt.rcdefaults()
    fig, ax = plt.subplots()
    line1 = ax.plot(x, y, '-o',  markevery=x_bt)
    line2 = ax.plot(x, yd_cong)
    ax.set_xticks(x)
    ax.set_xticklabels(xnhan)
    ax.set_xlabel('Thoi gian')
    ax.set_title(ten)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

def ve4(y, x, xnhan, yd_nhan, bt_nhan,ten):
    """
    Ve 2 do thi duong (x,y) va (x,y_nhan), truc hoanh hien thi xnhan
    """
    x_bt=[]
    for i in range(2,len(bt_nhan)):
        if (bt_nhan[i]==1) or (bt_nhan[i]==3):
            x_bt.append(i)                   
            
    plt.rcdefaults()
    fig, ax = plt.subplots()
    line1 = ax.plot(x, y, '-o',  markevery=x_bt)
    line2 = ax.plot(x, yd_nhan)
    ax.set_xticks(x)
    ax.set_xticklabels(xnhan)
    ax.set_xlabel('Thoi gian')
    ax.set_title(ten)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

def ve4_1(y, x, xnhan, yd_nhan, bt_nhan,ten):
    """
    Ve 2 do thi duong (x,y) va (x,y_nhan), truc hoanh hien thi xnhan
    """
    x_bt=[]
    for i in range(2,len(bt_nhan)):
        if (bt_nhan[i]==2) or (bt_nhan[i]==3):
            x_bt.append(i)                   
            
    plt.rcdefaults()
    fig, ax = plt.subplots()
    line1 = ax.plot(x, y, '-o',  markevery=x_bt)
    line2 = ax.plot(x, yd_nhan)
    ax.set_xticks(x)
    ax.set_xticklabels(xnhan)
    ax.set_xlabel('Thoi gian')
    ax.set_title(ten)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

def ve5(y, x, xnhan, yd_1, yd_tt, yd_cong, yd_nhan, bt_1, bt_tt, bt_cong, bt_nhan):
    """
    Ve 5 do thi 
    """
    x_bt=[]
    n=len(bt_1)
    for i in range(2,n):
        if (bt_1[i]==1) or (bt_tt[i]==1) or (bt_cong[i]==1) or (bt_nhan[i]==1) or (bt_1[i]==2) or (bt_tt[i]==2) or (bt_cong[i]==2) or (bt_nhan[i]==2) or (bt_1[i]==3) or (bt_tt[i]==3) or (bt_cong[i]==3) or (bt_nhan[i]==3):
            x_bt.append(i)
            
    plt.rcdefaults()
    fig, ax = plt.subplots()
    line1 = ax.plot(x, y, '-o',  markevery=x_bt)
    line2 = ax.plot(x, yd_1)
    line3 = ax.plot(x, yd_tt)
    line4 = ax.plot(x, yd_cong)
    line5 = ax.plot(x, yd_nhan)
    ax.set_xticks(x)
    ax.set_xticklabels(xnhan)
    ax.set_xlabel('Thoi gian')
    ax.set_title('Cac loai bat thuong')
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

def histo(e, muy, xichma, ten):
    x = e
    mu = muy
    sigma = xichma
    num_bins = 50

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(x, num_bins, normed=1)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, y, '--')
    ax.set_xlabel('Gia tri cua e')
    ax.set_ylabel('Mat do xac suat')
    ax.set_title(ten)
    # ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


def main():
    """
    """
    '''
    conn_str = (
        r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
        r'DBQ=D:\AChantroimoi\Trinhbay_CMC\TrabaiCMC\ThanhLyHD\\ChicucHQ\CCHQ_kn_stk.accdb;'
    )
    '''
    conn_str = (
        r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
        r'DBQ=F:\CMC\ChicucHQ\CCHQ_kn_stk.accdb'
    )
    cnxn = pyodbc.connect(conn_str)
    cursor0 = cnxn.cursor()  # Doc tu DactaChuoi
    cursor1 = cnxn.cursor()  # Doc tu DactaChuoi
    cursor2 = cnxn.cursor()  # Update DactaChuoi
    cursor3 = cnxn.cursor()  # Doc tu Chuoi
    cursor4 = cnxn.cursor()  # Update Chuoi

    while True:
        cursor0.execute("select distinct loai from dactachuoi order by loai")
        row0s = cursor0.fetchall()
        i0 = 0
        l = []
        print("Danh sach loai chuoi")
        for row0 in row0s:
            l.append(row0.loai)
            print(i0, ". Loai chuoi: ", row0.loai)
            i0 = i0 + 1
        num0 = int(input("Nhap so tuong ung: "))
        if num0 >= i0:
            print("Nhap sai, Ket thuc chuong trinh")
            break
        while True:
            """
            Doc 1 chuoi
            """
            cursor1.execute(
            "select tenchuoi,alpha_1,SSE_1,muy_1,xichma_1,alpha_tt,beta_tt,SSE_tt,muy_tt,xichma_tt, alpha_cong,beta_cong,gama_cong,SSE_cong,muy_cong,xichma_cong,alpha_nhan,beta_nhan,gama_nhan,SSE_nhan,muy_nhan,xichma_nhan from dactachuoi where loai=? order by tenchuoi", l[num0])
       
            row1s = cursor1.fetchall()
            i = 0
            t = []
            alpha_1 = []
            SSE_1 = []
            muy_1 = []
            xichma_1 = []
            alpha_tt = []
            beta_tt = []
            SSE_tt = []
            muy_tt = []
            xichma_tt = []
            alpha_cong = []
            beta_cong = []
            gama_cong = []
            SSE_cong = []
            muy_cong = []
            xichma_cong = []
            alpha_nhan = []
            beta_nhan = []
            gama_nhan = []
            SSE_nhan = []
            muy_nhan = []
            xichma_nhan = []            
            print("Danh sach cac chuoi")
            for row1 in row1s:
                t.append(row1.tenchuoi)                
                alpha_1.append(row1.alpha_1)
                SSE_1.append(row1.SSE_1)
                muy_1.append(row1.muy_1)
                xichma_1.append(row1.xichma_1)
                alpha_tt.append(row1.alpha_tt)
                beta_tt.append(row1.beta_tt)
                SSE_tt.append(row1.SSE_tt)
                muy_tt.append(row1.muy_tt)
                xichma_tt.append(row1.xichma_tt)
                alpha_cong.append(row1.alpha_cong)
                beta_cong.append(row1.beta_cong)
                gama_cong.append(row1.gama_cong)
                SSE_cong.append(row1.SSE_cong)
                muy_cong.append(row1.muy_cong)
                xichma_cong.append(row1.xichma_cong)
                alpha_nhan.append(row1.alpha_nhan)
                beta_nhan.append(row1.beta_nhan)
                gama_nhan.append(row1.gama_nhan)
                SSE_nhan.append(row1.SSE_nhan)
                muy_nhan.append(row1.muy_nhan)
                xichma_nhan.append(row1.xichma_nhan)
                print(i, ". Ten chuoi: ", row1.tenchuoi)
                i = i + 1
            num = int(input("Nhap so tuong ung: "))
            if num >= i:
                print("Nhap sai, Chuyen sang chuoi khac")
                break

            cursor3.execute(
                "select thutu,chiso,y,y_1,e_1,bt_1,yd_1,y_tt,e_tt,bt_tt,yd_tt,y_cong,e_cong,bt_cong,yd_cong,y_nhan,e_nhan,bt_nhan,yd_nhan from chuoi WHERE loai=? AND tenchuoi=? ORDER BY chiso",
                l[num0], t[num])
            row3s = cursor3.fetchall()
            xnhan = []
            x = [0]
            y = [0]
            y_1 = [0]
            e_1 = [0]
            bt_1 = [0]
            yd_1 = [0]
            y_tt = [0]
            e_tt = [0]
            bt_tt = [0]
            yd_tt = [0]
            y_cong = [0]
            e_cong = [0]
            bt_cong = [0]
            yd_cong = [0]
            y_nhan = [0]
            e_nhan = [0]
            bt_nhan = [0]
            yd_nhan = [0]
            for row3 in row3s:
                xnhan.append(row3.thutu)
                x.append(row3.chiso)
                y.append(row3.y)
                y_1.append(row3.y_1)
                e_1.append(row3.e_1)
                bt_1.append(row3.bt_1)
                yd_1.append(row3.yd_1)
                y_tt.append(row3.y_tt)
                e_tt.append(row3.e_tt)
                bt_tt.append(row3.bt_tt)
                yd_tt.append(row3.yd_tt)
                y_cong.append(row3.y_cong)
                e_cong.append(row3.e_cong)
                bt_cong.append(row3.bt_cong)
                yd_cong.append(row3.yd_cong)
                y_nhan.append(row3.y_nhan)
                e_nhan.append(row3.e_nhan)
                bt_nhan.append(row3.bt_nhan)
                yd_nhan.append(row3.yd_nhan)

            while True:
                """
                Ve chuoi doc duoc
                """        
                print("Danh sach cac do thi")
                print(" 0. Do thi chuoi")
                print(" 1. Do thi chuoi + bac1")
                print(" 2. Do thi chuoi + tuyen tinh")
                print(" 3. Do thi chuoi + mua vu cong")
                print(" 4. Do thi chuoi + mua vu nhan")
                print(" 5. Do thi chuoi + cac mo hinh")
                print(" 6. Do thi mo hinh sai so nho nhat")
            
                num_ve = int(input("Nhap so tuong ung: "))
                if num_ve >= 7:
                    print("Nhap sai, Chuyen sang tinh toan mo hinh")
                    break

                if num_ve == 0:
                    ten='Do thi (x,y)'
                    ve(y, x, xnhan,ten)
                elif num_ve == 1:
                    ten='MH bac 1 va BT theo pp chuan'
                    ve1(y, x, xnhan, yd_1, bt_1,ten)
                    ten='MH bac 1 va BT theo IQR'
                    ve1_1(y, x, xnhan, yd_1, bt_1,ten)
                    ten = "MH bac 1 va histogram cua sai so"
                    e = e_1
                    e[0] = 0
                    e[1] = 0
                    muy = muy_1[num]
                    xichma = xichma_1[num]
                    if muy<>0 and xichma<>0:
                        histo(e, muy, xichma, ten)
                elif num_ve == 2:
                    ten='MH tuyen tinh va BT theo pp chuan'
                    ve2(y, x, xnhan, yd_tt, bt_tt,ten)
                    ten='MH tuyen tinh va BT theo IQR'
                    ve2_1(y, x, xnhan, yd_tt, bt_tt,ten)
                    ten = "MH bac 1 va histogram cua sai so"
                    e = e_tt
                    e[0] = 0
                    e[1] = 0
                    muy = muy_tt[num]
                    xichma = xichma_tt[num]
                    if muy<>0 and xichma<>0:
                        histo(e, muy, xichma, ten)
                elif num_ve == 3:
                    ten='MH mua vu cong va BT theo pp chuan'
                    ve3(y, x, xnhan, yd_cong, bt_cong,ten)
                    ten='MH mua vu cong va BT theo IQR'
                    ve3_1(y, x, xnhan, yd_cong, bt_cong,ten)
                    ten = "MH mua vu cong va histogram cua sai so"
                    e = e_tt
                    e[0] = 0
                    e[1] = 0
                    e[2] = 0
                    muy = muy_tt[num]
                    xichma = xichma_tt[num]
                    if muy<>0 and xichma<>0:
                        histo(e, muy, xichma, ten)
                elif num_ve == 4:
                    ten='MH mua vu nhan va BT theo pp chuan'
                    ve4(y, x, xnhan, yd_nhan, bt_nhan,ten)
                    ten='MH mua vu nhan va BT theo IQR'
                    ve4_1(y, x, xnhan, yd_nhan, bt_nhan,ten)
                    ten = "MH mua vu cong va histogram cua sai so"
                    e = e_tt
                    e[0] = 0
                    e[1] = 0
                    e[2] = 0
                    muy = muy_tt[num]
                    xichma = xichma_tt[num]
                    if muy<>0 and xichma<>0:
                        histo(e, muy, xichma, ten)
                elif num_ve == 5:
                    ve5(y, x, xnhan, yd_1, yd_tt, yd_cong, yd_nhan, bt_1, bt_tt, bt_cong, bt_nhan)
                elif num_ve == 6:
                    if SSE_1[num] == -1:
                        SSE_1[num] = 1.e+30
                    if SSE_tt[num] == -1:
                        SSE_tt[num] = 1.e+30
                    if SSE_cong[num] == -1:
                        SSE_cong[num] = 1.e+30
                    if SSE_nhan[num] == -1:
                        SSE_nhan[num] = 1.e+30
                    if SSE_1[num] == min(SSE_1[num],SSE_tt[num],SSE_cong[num],SSE_nhan[num]):
                        ten='MH bac 1 la tot nhat va BT theo pp chuan'
                        ve1(y, x, xnhan, yd_1, bt_1,ten)
                        ten='MH bac 1 la tot nhat va BT theo IQR'
                        ve1_1(y, x, xnhan, yd_1, bt_1,ten)
                    elif SSE_tt[num] == min(SSE_1[num],SSE_tt[num],SSE_cong[num],SSE_nhan[num]):
                        ten='MH tuyen tinh la tot nhat va BT theo pp chuan'
                        ve2(y, x, xnhan, yd_tt, bt_tt,ten)
                        ten='MH tuyen tinh la tot nhat va BT theo IQR'
                        ve2_1(y, x, xnhan, yd_tt, bt_tt,ten)
                    elif SSE_cong[num] == min(SSE_1[num],SSE_tt[num],SSE_cong[num],SSE_nhan[num]):
                        ten='MH mua vu cong la tot nhat va BT theo pp chuan'
                        ve3(y, x, xnhan, yd_cong, bt_cong,ten)
                        ten='MH mua vu cong la tot nhat va BT theo IQR'
                        ve3_1(y, x, xnhan, yd_cong, bt_cong,ten)
                    elif SSE_nhan[num] == min(SSE_1[num],SSE_tt[num],SSE_cong[num],SSE_nhan[num]):
                        ten='MH mua vu nhan la tot nhat va BT theo pp chuan'
                        ve4(y, x, xnhan, yd_nhan, bt_nhan,ten)
                        ten='MH mua vu nhan la tot nhat va BT theo IQR'
                        ve4_1(y, x, xnhan, yd_nhan, bt_nhan,ten)
            while True:
                """
                Tinh toan tham so cho mo hinh
                """
                print("Danh sach cac mo hinh")
                print(" 0. Mo hinh bac 1")
                print(" 1. Mo hinh tuyen tinh")
                print(" 2. Mo hinh mua vu cong")
                print(" 3. Mo hinh mua vu nhan")
                num_mohinh = int(input("Nhap so tuong ung: "))
                if num_mohinh >= 4:
                    print("Nhap sai, Goi chuoi khac")
                    break

                if num_mohinh == 0:
                    print("Mo hinh bac 1 cua chuoi")
                    print("Thay doi tham so mo hinh")
                    print("0. Tu dong")
                    print("1. Chuyen gia")
                    num_thamso = int(input("Nhap so tuong ung: "))
                    if num_thamso == 0:
                        alpha_1 = alpha_1f(y)
                        [y_1, e_1, bt_1, yd_1, SSE_1, muy_1, xichma_1] = bac1(y, alpha_1)
                    elif num_thamso == 1:
                        alpha_1 = float(input("Nhap alphah trong [0,1], goi y [0.05,0.3]: "))
                        [y_1, e_1, bt_1, yd_1, SSE_1, muy_1, xichma_1] = bac1(y, alpha_1)

                    # Ghi vao bang dacta va chuoi
                    print("Danh sach cac tham so mo hinh bac 1 cua chuoi sau tinh toan")
                    print("Tham so alpha:  ", alpha_1)
                    print("Tham so SSE:  ", SSE_1)
                    print("Tham so muy:  ", muy_1)
                    print("Tham so xichma:  ", xichma_1)
                    cursor2.execute("UPDATE dactachuoi SET alpha_1=?,SSE_1=?,muy_1=?,xichma_1=? WHERE loai=? AND tenchuoi=?",
                                    alpha_1, SSE_1, muy_1, xichma_1, l[num0], t[num])
                    for i in range(1, len(y)):
                        cursor4.execute("UPDATE chuoi SET y_1=?,e_1=?,bt_1=?,yd_1=? WHERE loai=? AND tenchuoi=? AND chiso=?",
                                        y_1[i],e_1[i], bt_1[i], yd_1[i], l[num0], t[num], x[i])

                elif num_mohinh == 1:
                    print("Mo hinh tuyen tinh cua chuoi")
                    print("Thay doi tham so mo hinh")
                    print("0. Tu dong")
                    print("1. Chuyen gia")
                    num_thamso = int(input("Nhap so tuong ung: "))
                    if num_thamso == 0:
                        [alpha_tt, beta_tt] = alphabeta_f(y)
                        [y_tt, e_tt, bt_tt, yd_tt, SSE_tt, muy_tt, xichma_tt] = tt(y, alpha_tt, beta_tt)
                    elif num_thamso == 1:
                        alpha_tt = float(input("Nhap alphah trong [0,1], goi y [0.02,0.2]: "))
                        beta_tt = float(input("Nhap beta trong [0,1], goi y [0.02,0.2]: "))
                        [y_tt, e_tt, bt_tt, yd_tt, SSE_tt, muy_tt, xichma_tt] = tt(y, alpha_tt, beta_tt)

                    # Ghi vao bang dacta va chuoi
                    print("Danh sach cac tham so mo hinh tuyen tinh cua chuoi sau tinh toan")
                    print("Tham so alpha:  ", alpha_tt)
                    print("Tham so beta:  ", beta_tt)
                    print("Tham so SSE:  ", SSE_tt)
                    print("Tham so muy:  ", muy_tt)
                    print("Tham so xichma:  ", xichma_tt)
                    cursor2.execute("UPDATE dactachuoi SET alpha_tt=?,beta_tt=?,SSE_tt=?,muy_tt=?,xichma_tt=? WHERE loai=? AND tenchuoi=?",
                                    alpha_tt, beta_tt, SSE_tt, muy_tt, xichma_tt, l[num0], t[num])
                    for i in range(1, len(y)):
                        cursor4.execute("UPDATE chuoi SET y_tt=?,e_tt=?,bt_tt=?, yd_tt=? WHERE loai=? AND tenchuoi=? AND chiso=?",
                                    y_tt[i], e_tt[i], bt_tt[i], yd_tt[i], l[num0], t[num], x[i])
                        
                elif num_mohinh == 2:
                    print("Mo hinh mua vu cong cua chuoi")
                    m = int(input("Nhap do dai mua vu. Quy - 4, Thang -12: "))
                    print("Thay doi tham so mo hinh")
                    print("0. Tu dong")
                    print("1. Chuyen gia")
                    num_thamso = int(input("Nhap so tuong ung: "))
                    if num_thamso == 0:
                        [alpha_cong, beta_cong, gama_cong] = abg_congf(y, m)
                        [y_cong, e_cong, bt_cong, yd_cong, SSE_cong, muy_cong, xichma_cong] = cong(y, alpha_cong, beta_cong, gama_cong,m)
                    elif num_thamso == 1:
                        alpha_cong = float(input("Nhap alphah trong [0,1], goi y [0.02,0.2]: "))
                        beta_cong = float(input("Nhap beta trong [0,1], goi y [0.02,0.2]: "))
                        gama_cong = float(input("Nhap gama trong [0,1], goi y [0.02,0.2]: "))
                        [y_cong, e_cong, bt_cong, yd_cong, SSE_cong, muy_cong, xichma_cong] = cong(y, alpha_cong, beta_cong, gama_cong, m)

                    # Ghi vao bang dacta va chuoi
                    print("Danh sach cac tham so mo hinh mua vu cong cua chuoi sau tinh toan")
                    print("Tham so alpha:  ", alpha_cong)
                    print("Tham so beta:  ", beta_cong)
                    print("Tham so gama:  ", gama_cong)
                    print("Tham so SSE:  ", SSE_cong)
                    print("Tham so muy:  ", muy_cong)
                    print("Tham so xichma:  ", xichma_cong)
                    cursor2.execute("UPDATE dactachuoi SET alpha_cong=?,beta_cong=?,gama_cong=?,SSE_cong=?,muy_cong=?,xichma_cong=?,muavu=? WHERE loai=? AND tenchuoi=?",
                                    alpha_cong, beta_cong, gama_cong, SSE_cong, muy_cong, xichma_cong, m, l[num0], t[num])
                    for i in range(1, len(y)):
                        cursor4.execute("UPDATE chuoi SET y_cong=?,e_cong=?,bt_cong=?, yd_cong=? WHERE loai=? AND tenchuoi=? AND chiso=?",
                                        y_cong[i], e_cong[i], bt_cong[i], yd_cong[i], l[num0], t[num], x[i])

                elif num_mohinh == 3:
                    print("Mo hinh mua vu nhan cua chuoi")
                    m = int(input("Nhap do dai mua vu. Quy - 4, Thang -12: "))
                    print("Thay doi tham so mo hinh")
                    print("0. Tu dong")
                    print("1. Chuyen gia")
                    num_thamso = int(input("Nhap so tuong ung: "))
                    if num_thamso == 0:
                        [alpha_nhan, beta_nhan, gama_nhan, SSE_nhan] = abg_nhanf(y, m)
                        [y_nhan, e_nhan, bt_nhan, yd_nhan, SSE_nhan, muy_nhan, xichma_nhan] = nhan(y, alpha_nhan, beta_nhan, gama_nhan, m)
                    elif num_thamso == 1:
                        alpha_nhan = float(input("Nhap alphah trong [0,1], goi y [0.02,0.2]: "))
                        beta_nhan = float(input("Nhap beta trong [0,1], goi y [0.02,0.2]: "))
                        gama_nhan = float(input("Nhap gama trong [0,1], goi y [0.02,0.2]: "))
                        [y_nhan, e_nhan, bt_nhan, yd_nhan, SSE_nhan, muy_nhan, xichma_nhan] = nhan(y, alpha_nhan, beta_nhan, gama_nhan, m)

                    # Ghi vao bang dacta va chuoi
                    print("Danh sach cac tham so mo hinh mua vu nhan cua chuoi sau tinh toan")
                    print("Tham so alpha:  ", alpha_nhan)
                    print("Tham so beta:  ", beta_nhan)
                    print("Tham so gama:  ", gama_nhan)
                    print("Tham so SSE:  ", SSE_nhan)
                    print("Tham so muy:  ", muy_nhan)
                    print("Tham so xichma:  ", xichma_nhan)
                    cursor2.execute("UPDATE dactachuoi SET alpha_nhan=?,beta_nhan=?,gama_nhan=?,SSE_nhan=?,muy_nhan=?,xichma_nhan=?,muavu=? WHERE loai=? AND tenchuoi=?",
                                    alpha_nhan, beta_nhan, gama_nhan, SSE_nhan, muy_nhan, xichma_nhan, m, l[num0], t[num])
                    for i in range(1, len(y)):
                        cursor4.execute("UPDATE chuoi SET y_nhan=?,e_nhan=?,bt_nhan=?,yd_nhan=? WHERE loai=? AND tenchuoi=? AND chiso=?",
                                        y_nhan[i], e_nhan[i], bt_nhan[i], yd_nhan[i], l[num0], t[num], x[i])

    cnxn.commit()


main()
# vechuoi()
