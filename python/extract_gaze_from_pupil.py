#cd /home/gazetracker/pupil/pupil_v0.9.14-7_linux_x64/pupil/pupil_src/shared_modules
from file_methods import save_object, load_object
import os
import numpy


##check dictionary
pupil_data = load_object(os.path.join("/home/gazetracker/recordings/2018_02_17/006_95%", "pupil_data"))
#print(pupil_data.keys())
#dict_keys(['gaze_positions', 'fixations', 'notifications', 'pupil_positions'])
gaze=pupil_data['gaze_positions']
#print(gaze[0].keys())
#dict_keys(['norm_pos', 'gaze_point_3d', 'topic', 'base_data', 'timestamp', 'gaze_normals_3d', 'eye_centers_3d', 'confidence'])



##extract & save gaze_pose from dictionary
norm=[]
i=0
time=[]
while (i<len(gaze)):
     time.append(gaze[i]['timestamp'])
     norm.append(gaze[i]['norm_pos'])
     i=i+1
numpy.save("/home/gazetracker/recordings/2018_02_17/006_95%/norm_pos.npy",norm)
numpy.save("/home/gazetracker/recordings/2018_02_17/006_95%/pos_timestamp.npy",time)


##load gaze_pose
norm_pos=numpy.load("/home/gazetracker/recordings/2018_02_17/006_95%/norm_pos.npy")#(6812, 2)
pos_time=numpy.load("/home/gazetracker/recordings/2018_02_17/006_95%/pos_timestamp.npy")
world_time=numpy.load("/home/gazetracker/recordings/2018_02_17/006_95%/world_timestamps.npy")#(3306,)

##compute gaze position at world timestamps
num=len(world_time)
pos_frame=[]
former=pos_time[0]
j=0
for i in range(num):
	later=0
	while(later==0):
		if (j==len(pos_time)-1):			
			break
		if (world_time[i]>=pos_time[j]) :
			former=pos_time[j]
			j+=1
		elif (world_time[i]<pos_time[j]):
			later=pos_time[j]
	if (j==len(pos_time)-1):
		pos_frame.append(norm_pos[j,:])
	else:
		#linear fitting
		k=(world_time[i]-former)/(later-former)
		pos_frame.append(norm_pos[j-1,:]+(norm_pos[j,:]-norm_pos[j-1,:])*k)

	#if((world_time[i]-former)>(later-world_time[i])):
	#	pos_frame.append(norm_pos[j-1,:])
	#else:
	#	pos_frame.append(norm_pos[j,:])	
#numpy.save("/home/gazetracker/recordings/2018_02_17/006_95%/pos_frame.npy",pos_frame)
numpy.savetxt('/home/gazetracker/recordings/2018_02_17/006_95%/pos_frame.txt',pos_frame)
	
##plot gaze_distribution
#marker_size=5
#scatter(norm_pos[:,0]*1280,norm_pos[:,1]*720,marker_size,c='b')
#scatter(0.5*1280,0.5*720,20,c='r')
#plt.xlim((0,1280))
#plt.ylim((0,720))
#plt.title("Gaze Distributions")
#plt.xlabel('Horizental Pixel')
#plt.ylabel('Vertical Pixel')
#plt.show()
