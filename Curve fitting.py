"""
@author: Pradnya Mundargi
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np 

pics=[]
cap= cv2.VideoCapture(r'*insert path*')
if(cap.isOpened()==False):
    print('Error opening file')
    
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame= cv2.resize(frame,(480,320))
        frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(frame, 220, 255, cv2.THRESH_BINARY_INV)[1]
        pics.append(thresh)
        cv2.imshow('frame',thresh)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()

height,width = thresh.shape

x_cord=[]
y_cord=[]
x=[]
y=[]

for i in range(len(pics)):
    row=0
    for j in pics[i]:
        if j.any()!=0:
            x_cord.append(np.where(j==255))
            y_cord.append(row)
        row+=1
    
    
    for a in x_cord[0]:
        if(len(a)%2==0):
            x.append(a[int(len(a)/2)])
            x.append(a[int(len(a)/2)])
            x_cord.clear()
        else:
            x.append(a[int((len(a)/2)+1)])
            x.append(a[int((len(a)/2)+1)])
            x_cord.clear()
     
    y.append(height- int(y_cord[0]))
    y.append(height- int(y_cord[-1]))
    y_cord.clear()
x_val=[]
y_val=[]
for k in range(0,len(y),2):
    y_val.append((y[k]+y[k+1])/2)

for k in range(len(x)):
    if k%2==0:
        x_val.append(x[k])
print(x_val, len(x_val))

graph=plt.scatter(x_val,y_val)

# Least square Model Creation
def power(m,n):
    x_power = []
    for i in range(len(m)):
        x_power.append(pow(m[i],n))    
    return x_power

def Model(m,n):
    X = []
    for i in range(len(m)):
        X.append([1,m[i],pow(m[i],2)])
    Xdata = np.array(X)
    Y = np.array(n)
    P = np.dot(np.linalg.inv(np.dot(np.transpose(Xdata),Xdata)),np.dot(np.transpose(Xdata),Y))
    return P

def points(m,n,A1):
    ypred = np.array([A1[2]])*power(m,2)+np.array([A1[1]])*m+np.array([A1[0]])
    return ypred

def leastsqr(xdata,ydata):
    plt.plot(xdata,ydata,'o')
    plt.plot(xdata,points(xdata,ydata,Model(xdata,ydata)), color='red')
    plt.show()
    
leastsqr(x_val,y_val)
