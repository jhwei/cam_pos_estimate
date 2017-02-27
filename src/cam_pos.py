import cv2
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Find the fourth vertex point in QR code given the three point
def find_intersection(top,bottom,right):
	#Use the sum of two vector to find the intersection point
	#Given a,b,c, the sum of vector is b-a+c-a, the intersection point is b-a+c-a+a=b+c-a
	intersection=(bottom[0]+right[0]-top[0],bottom[1]+right[1]-top[1])
	return intersection

#Given the rotation matrix, compute the rotation angle of each axis, the result is degree format
def rotationMatrix2Angle(R):
	x=math.atan2(R[2][1],R[2][2])/2/math.pi*360.0
	y=math.atan2(-R[2][0],math.sqrt(R[2][1]*R[2][1]+R[2][2]*R[2][2]))/2/math.pi*360.0
	z=math.atan2(R[1][0],R[0][0])/2/math.pi*360.0
	return [x,y,z]

#Init the plot figure in 3D style
fig = plt.figure()
ax = fig.gca(projection='3d')

#QR code vertex point in world coordinate
qr3d = np.float32([[-44,44, 0], [-44,-44, 0], [44,-44, 0], [44,44, 0]])

#The point used to plot camera frame in figure
camera_frame = [[0,0,0], [40,0,0], [40,40,0], [0,40,0], [0,0,0]]

#Define the camera intrinsic matrix given the spec of iPhone camera
K = np.float64([[2803, 0, 1224],
				[0, 2803, 1632],
				[0.0,0.0, 1.0]])

#Define the camera distortion coefficients to be zero
dist_coef = np.zeros(4)

#Plot the QR code in the figure
x=[]
y=[]
z=[]
for point in qr3d:
	x.append(point[0])
	y.append(point[1])
	z.append(point[2])
x.append(qr3d[0][0])
y.append(qr3d[0][1])
z.append(qr3d[0][2])
ax.plot(x, y, z, label='QR code')

#Read the filename into a list
f=open('img_file.txt','r')
img_file=[]
for line in f:
	img_file.append(line[:-1])
f.close()

f=open('../output/stat.txt','w+')

#Go through the list to compute each location of camera
for filename in img_file:
	img = cv2.imread(filename)
	
	#Use Canny to detect edges in image
	edges = cv2.Canny(img,200,200)
	
	#Find square contours in image
	_,contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	#Iterate through all the contours to find the three squares in QR code
	mark = 0
	for x in range(0,len(contours)):
		k = x
		c = 0
		while(hierarchy[0][k][2] != -1):
			k = hierarchy[0][k][2]
			c = c + 1
		
		#If the inner level is 5, the contour is the square contour
		if c ==5:
			if mark == 0:
				A = x
			elif mark == 1:
				B = x
			elif mark == 2:
				C = x
			mark = mark+1
	
	#Link all the contour points into a array
	all_cont=np.concatenate((contours[A],contours[B],contours[C]))
	
	#Use function in OpenCV to calculate the contour approximation, the result points are the vertex of the detected square contours
	epsilon = 0.1*cv2.arcLength(all_cont,True)
	approx = cv2.approxPolyDP(all_cont,epsilon,True)

	A=approx[0][0]
	B=approx[1][0]
	C=approx[2][0]
	
	#Find distance between each point
	AB = np.linalg.norm(A-B)
	BC = np.linalg.norm(B-C)
	AC = np.linalg.norm(A-C)

	#Find the topleft vertex using three distances
	if(AB>BC and AB>AC):
		top = C
		bottom = A
		right = B
	elif(AC>AB and AC>BC):
		top = B
		bottom = A 
		right = C 
	elif(BC>AB and BC>AC):
		top = A 
		bottom = B
		right = C	
	
	#Rearrage bottom and right vertex using cross product
	if np.cross(bottom-top,right-top)>0:
		bottom,right=right,bottom

	#Find the last vertex given the three vertexes
	N=find_intersection(top,bottom,right)		
	
	#The corresponding QR code vertex in image coordinate
	qr_im = np.float32([top,bottom,[N[0],N[1]],right])

	#Solve the PnP problem using OpenCV function and get the rotation and translation
	ret, rvec, tvec = cv2.solvePnP(qr3d, qr_im, K, dist_coef)
	
	#Change rotation vector to rotation matrix
	rot_mat,_=cv2.Rodrigues(rvec)
	
	#Change rotation and translation from camera coordinate to world coordinate
	rot_mat=rot_mat.transpose()
	tvec=-np.dot(rot_mat,tvec)

	#draw the camera frame in the figure
	x=[]
	y=[]
	z=[]

	for point in camera_frame:
		point=np.array(point)
		out=np.dot(rot_mat,point.transpose())+tvec.transpose()
		out=out[0]
		x.append(out[0])
		y.append(out[1])
		z.append(out[2])
	
	ax.plot(x, y, z, label=filename[-8:-4])
	
	#Calculate the rotation angle from rotation matrix
	angle=rotationMatrix2Angle(rot_mat)
	
	print "The location of "+filename[-8:-4]+" is at x: %.2f cm, y: %.2f cm, z: %.2f cm, Pitch is %.2f degrees, Yaw is %.2f degrees, Roll is %.2f degrees." %(tvec[0]/10.0,tvec[1]/10.0,tvec[2]/10.0,angle[0],angle[1],angle[2])
	
	f.write("The location of "+filename[-8:-4]+" is at x: %.2f cm, y: %.2f cm, z: %.2f cm, Pitch is %.2f degrees, Yaw is %.2f degrees, Roll is %.2f degrees.\n" %(tvec[0]/10.0,tvec[1]/10.0,tvec[2]/10.0,angle[0],angle[1],angle[2]))

f.close()
ax.legend()
plt.show()


