import os
import sys
import time
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI
import numpy as np
import threading
from matplotlib import pyplot as plt
from scipy.optimize import leastsq

POSITION_PARAMETERS = ['x','y','z','roll','pitch','yaw']
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=120)

def Euler2SO3(angles):
	x, y, z = angles[0], angles[1], angles[2]
	Rx = np.array([[1,0,0],
				   [0,np.cos(x),-np.sin(x)],
				   [0,np.sin(x),np.cos(x)]])
	Ry = np.array([[np.cos(y),0,np.sin(y)],
				   [0,1,0],
				   [-np.sin(y),0,np.cos(y)]])
	Rz = np.array([[np.cos(z),-np.sin(z),0],
				   [np.sin(z),np.cos(z),0],
				   [0,0,1]])
	return(np.dot(Rx,Ry).dot(Rz))

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def SO32Euler(R) :

	assert(isRotationMatrix(R))

	sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

	singular = sy < 1e-6

	if  not singular :
		x = np.arctan2(R[2,1] , R[2,2])
		y = np.arctan2(-R[2,0], sy)
		z = np.arctan2(R[1,0], R[0,0])
	else :
		print("singular")
		x = np.arctan2(-R[1,2], R[1,1])
		y = np.arctan2(-R[2,0], sy)
		z = 0

	return np.array([x, y, z])

def Euler2SO2(theta):
	return(np.array([[np.cos(theta),-np.sin(theta)],
					 [np.sin(theta),np.cos(theta)]]))

def pose_distance(p1,p2,linear_weight = 1,rotational_weight=1):
	linear_dist = np.linalg.norm(p1[:3]-p2[:3])
	rot1 = p1[3:6]
	rot2 = p2[3:6]
	rotational_dist = np.linalg.norm(np.concatenate((np.sin(rot1),np.cos(rot1))) - np.concatenate((np.sin(rot2),np.cos(rot2))))
	return(linear_dist*linear_weight + rotational_dist*rotational_weight)

class RobotController:

	def __init__(self,robot_ip,controller_frequency = 100,speed=300,mvacc=200, print_frequency = 2):
		self.arm = XArmAPI(robot_ip, is_radian=True, enable_report=True)
		self.goal_pos = None
		self.force_axes = np.zeros(6,dtype=int)
		self.force_ref = np.zeros(6)
		self.command_force_axes = np.zeros(6,dtype=int)
		self.command_force_ref = np.zeros(6)
		self.speed = speed
		self.mvacc = mvacc
		self.Kp = 0.005  # range: 0 ~ 0.05
		self.Ki = 0.00006 # range: 0 ~ 0.0005
		self.Kd = 0.00  # range: 0 ~ 0.05
		self.v_max = 100.0
		self.force_safety_limit = 7.0
		self.force_limited = False
		self.force_control_mode = 0
		self.controller_frequency = controller_frequency
		self.print_frequency = print_frequency
		try:
			self.ft_correction_mat = np.load("ft_correction_mat.npy")
		except FileNotFoundError:
			self.ft_correction_mat = np.eye(3)
		

	def init_robot(self):
		self.arm.ft_sensor_enable(0)
		self.arm.set_state(0)
		code = self.arm.set_tcp_offset([0., 0., 235., 0., 0., 0.],wait=True) # Replace with TCP Offset from XArm settings
		assert code == 0, "Failed to set tcp offset"
		self.arm.set_state(0)
		code = self.arm.set_tcp_load(1.38,[-.96,6.41,76.21],wait=True) # Replace with TCP Load from XArm settings
		assert code == 0, "Failed to set tcp load"

		self.arm.set_force_control_pid([self.Kp]*6, [self.Ki]*6, [self.Kd]*6, [self.v_max]*6)
		self.arm.set_state(1)

		self.arm.clean_error()
		self.arm.clean_warn()
		self.arm.ft_sensor_enable(1)
		time.sleep(0.5)
		self.arm.ft_sensor_set_zero()
		time.sleep(0.2) # wait for writing zero operation to take effect, do not remove

		self.arm.motion_enable(enable=True)
		code = self.arm.set_mode(7)
		self.arm.set_state(0)
		time.sleep(1)
		assert self.arm.mode == 7, "Failed to set mode to 7, if needed, extend the sleep value"


		


		self._control_thread = threading.Thread(target=self._control_thread)
		self._print_thread = threading.Thread(target=self._print_thread)
		self._mutex = threading.Lock()
		
		self.start_controller()

	def set_speed(self,speed):
		self._mutex.acquire()
		self.speed = speed
		self._mutex.release()

	def set_mvacc(self,mvacc):
		self._mutex.acquire()
		self.mvacc = mvacc
		self._mutex.release()

	def set_goal_position(self,goal_pos):
		self._mutex.acquire()
		self.goal_pos = goal_pos
		self._mutex.release()

	def set_position(self,position):
		self._mutex.acquire()
		self.goal_pos = position
		self._mutex.release()
		while(pose_distance(self.get_position(),position) > 5):
			time.sleep(0.001)


	def get_position(self):
		return(np.array(self.arm.get_position(is_radian=True)[1]))

	def get_ft(self,frame = 0):
		#frame 0: base, 1: tool
		ft_data = np.array(self.arm.get_ft_sensor_data()[1])
		ft_data[:3] = np.dot(self.ft_correction_mat,ft_data[:3])

		if frame == 0:
			robot_pos = self.get_position()
			R = Euler2SO3(robot_pos[3:6])
			ft_data[:3] = np.dot(R,ft_data[:3])
			ft_data[3:] = SO32Euler(np.dot(R,R))

		return(ft_data)

	def test_force_axes(self, force_axis, force_ref):
		ref_frame = 0        # 0 : base , 1 : tool
		self.arm.config_force_control(ref_frame,  force_axis, force_ref, [0]*6) # limits are reserved, just give zeros

		# enable ft sensor communication
		self.arm.ft_sensor_enable(1)

		# will overwrite previous sensor zero and payload configuration
		self.arm.ft_sensor_set_zero() # remove this if zero_offset and payload already identified & compensated!
		time.sleep(0.2) # wait for writing zero operation to take effect, do not remove

		# move robot in force control
		self.arm.ft_sensor_app_set(2)
		# will start after set_state(0)
		self.arm.set_state(0)


	def set_force_axes(self,force_axes,force_ref):
		self._mutex.acquire()
		self.force_axes = force_axes
		self.force_ref = force_ref
		if self.force_axes.any():
			self.arm.config_force_control(0,  force_axes.tolist(), force_ref.tolist(), [0]*6)
			if self.force_control_mode == 0:
				self.force_control_mode = 2
				self.arm.ft_sensor_app_set(2)
		elif self.force_control_mode == 2:
			self.force_control_mode = 0
			self.arm.ft_sensor_app_set(0)
		self.arm.set_state(0)
		self._mutex.release()

			

	def _control_thread(self):
		self.last_command_time = time.time()
		command_pos = self.get_position()
		while not self.stopped:
			while time.time() < self.last_command_time + 1/self.controller_frequency or self.paused:
				time.sleep(0.001)

			
			self.last_command_time = time.time()
			self._mutex.acquire()
			ft = self.get_ft()
			axes_mask = (self.command_force_axes*-1) + 1
			ft = np.multiply(ft,axes_mask)
			neg_ft = ft[:2]*-1
			movement_vec = self.goal_pos[:2] - self.get_position()[:2]
			if np.linalg.norm(neg_ft) > self.force_safety_limit and np.dot(movement_vec,neg_ft) > 0:
				self.force_limited = True
			else:
				self.force_limited = False

			if np.any(command_pos != self.goal_pos):
				command_pos = self.goal_pos
				pos_dict = {}
				for i in range(len(POSITION_PARAMETERS)):
					if self.force_axes[i] == 0:
						pos_dict[POSITION_PARAMETERS[i]] = command_pos[i]
				pos_dict['speed'] = self.speed
				pos_dict['mvacc'] = self.mvacc
				pos_dict['is_radian'] = True
				self.arm.set_position(**pos_dict)
			
			self._mutex.release()


	def _print_thread(self):
		self.last_print_time = time.time()
		while not self.print_stopped:
			while time.time() < self.last_print_time + 1/self.print_frequency:
				time.sleep(0.001)
			self.last_print_time = time.time()
			print("FT: ",self.get_ft())
			print("Position: ",self.get_position())
			print("Set point: ",self.goal_pos)
			print("Force limited", self.force_limited)


	def stop_controller(self):
		self.stopped = True
		if self._control_thread.is_alive():
			self._control_thread.join()
		self.arm.ft_sensor_app_set(0)
		self.arm.ft_sensor_enable(0)

	def start_controller(self):
		self.arm.set_state(0)
		self.goal_pos = self.get_position()
		self.stopped = False
		self.paused = False
		self._control_thread.start()

	def pause_controller(self):
		self.paused = True

	def resume_controller(self):
		self.goal_pos = self.get_position()
		self.paused = False

	def start_printer(self):
		self.print_stopped = False
		self._print_thread.start()

	def stop_printer(self):
		self.print_stopped = True
		if self._print_thread.is_alive():
			self._print_thread.join()

	def primitive_move_along_contact(self,force_axes,force_ref,trajectory,relative_trajectory = True):
		self.arm.set_state(0)
		self.command_force_axes = force_axes
		self.command_force_ref = force_ref
		self.set_force_axes(force_axes,force_ref)
		axes_mask = (force_axes*-1) + 1
		#Wait until contact is made in the specified direction
		contact_thresh_percent = .75
		contact_made = False
		while not contact_made:
			ft = self.get_ft()
			masked_ft = np.multiply(ft,force_axes)
			masked_goals = np.multiply(force_ref,force_axes)*contact_thresh_percent
			contact_made = np.all(np.abs(masked_ft) >= np.abs(masked_goals))
			time.sleep(0.001)
		#Move along trajectory
		dist_thresh = 10
		if relative_trajectory:
			trajectory += self.get_position()
		for i in range(trajectory.shape[0]):
			self.set_goal_position(trajectory[i,:])
			while np.linalg.norm(np.multiply(self.get_position(),axes_mask) - np.multiply(trajectory[i,:],axes_mask)) > dist_thresh and not self.force_limited:
				time.sleep(0.001)
			if self.force_limited:
				time.sleep(.1)

		self.set_force_axes(np.zeros(6,dtype=int),np.zeros(6))

	def primitive_move_to_contact(self,force_axes):
		self.arm.set_state(0)
		self.command_force_axes = force_axes
		self.command_force_ref = force_ref
		self.set_force_axes(force_axes,force_ref)
		axes_mask = (force_axes*-1) + 1
		if relative_trajectory:
			trajectory += self.get_position()

		contact_made = False
		while not contact_made:
			ft = self.get_ft()
			masked_ft = np.multiply(ft,force_axes)
			masked_goals = np.multiply(force_ref,force_axes)*contact_thresh
			contact_made = np.all(np.abs(masked_ft) >= np.abs(masked_goals))


		self.set_force_axes(np.zeros(6,dtype=int),np.zeros(6))

			
	
	def primitive_freespace_move(self,trajectory,rel_axes = np.array([1,1,1,1,1,1]), relative_trajectory = True):
		self.arm.set_state(0)
		if relative_trajectory:
			trajectory += self.get_position()

		dist_thresh = 10
		for i in range(trajectory.shape[0]):
			self.set_goal_position(trajectory[i,:])

			while pose_distance(np.multiply(self.get_position(),rel_axes), np.multiply(trajectory[i],rel_axes)) > dist_thresh:
				time.sleep(0.001)
			if self.force_limited:
				time.sleep(.1)



def ft_calibration_loss(params,x,y):
	shear = np.array([[1,0],[params[0],1]])
	R = Euler2SO2(params[1])
	x_transformed = np.dot(R,np.dot(shear,x.T)).T
	loss = np.zeros(x.shape[0])
	for i in range(x.shape[0]):
		loss[i] = np.arccos(np.dot(x_transformed[i],y[i])/(np.linalg.norm(x_transformed[i])*np.linalg.norm(y[i])))
	return(loss)

def calibrate_ft_sensor(samples,gt):
	x = np.zeros(2)
	print(np.mean(ft_calibration_loss(x,samples,gt)))
	obj = lambda x: ft_calibration_loss(x,samples,gt)
	x = leastsq(obj, x, maxfev=10000)[0]
	print(np.mean(ft_calibration_loss(x,samples,gt)))
	return(x)

def collect_ft_data(robot, n_samples):
	time.sleep(3)
	n_samples = 50
	x_pos_samples = np.zeros((n_samples,2))
	x_neg_samples = np.zeros((n_samples,2))
	y_pos_samples = np.zeros((n_samples,2))
	y_neg_samples = np.zeros((n_samples,2))
	# x+
	print("Starting Sampling for X+ in...")
	for i in range(3,0,-1):
		print(i)
		time.sleep(1)
	print("Start")
	for i in range(n_samples):
		x_pos_samples[i,:] = robot.get_ft()[:2]
		time.sleep(0.1)
	print("Done")

	# x-
	print("Starting Sampling for X- in...")
	for i in range(3,0,-1):
		print(i)
		time.sleep(1)
	print("Start")
	for i in range(n_samples):
		x_neg_samples[i,:] = robot.get_ft()[:2]
		time.sleep(0.1)
	print("Done")


	# y+
	print("Starting Sampling for Y+ in...")
	for i in range(3,0,-1):
		print(i)
		time.sleep(1)
	print("Start")
	for i in range(n_samples):
		y_pos_samples[i,:] = robot.get_ft()[:2]
		time.sleep(0.1)
	print("Done")


	# y-
	print("Starting Sampling for Y- in...")
	for i in range(3,0,-1):
		print(i)
		time.sleep(1)
	print("Start")
	for i in range(n_samples):
		y_neg_samples[i,:] = robot.get_ft()[:2]
		time.sleep(0.1)
	print("Done")



	samples = np.vstack((x_pos_samples,x_neg_samples,y_pos_samples,y_neg_samples))
	gt = np.zeros((4*n_samples,2))
	gt[:n_samples,:] = np.array([1,0])
	gt[n_samples:2*n_samples,:] = np.array([-1,0])
	gt[2*n_samples:3*n_samples,:] = np.array([0,1])
	gt[3*n_samples:,:] = np.array([0,-1])
	return(samples,gt)

def clean_ft_data(samples,gt):
	sample_norms = np.linalg.norm(samples,axis=1)
	samples = samples[sample_norms > 1.,:]
	gt = gt[sample_norms > 1.,:]
	return(samples,gt)

def run_ft_sensor_calibration(robot):
	# Some FT sensors come with misaligned x-y axes. Run this function to estimate a correction matrix.
	# The arm should have a roll of -180/180, a pitch of 0 and a yaw of 0 in the world frame when calibrating.
	# During sampling, push the robot end effector in the specified direction.
	# Don't forget, in the robots frame, x+ is forward and y+ is left.
	robot.set_position(np.array([300,0,150,-np.pi,0,0]))


	samples,gt = collect_ft_data(robot,50)
	samples,gt = clean_ft_data(samples,gt)


	np.save("ft_samples",samples)
	np.save("ft_gt",gt)
	params = calibrate_ft_sensor(samples,gt)
	print("Estimated shear: {}, Estimated roll: {}".format(params[0],params[1]))

	shear = np.array([[1,0],[params[0],1]])
	R = Euler2SO2(params[1])
	world_correction_mat = np.eye(3)
	world_correction_mat[:2,:2] = np.dot(R,shear)
	ee_pose = robot.get_position()
	robot_to_world = Euler2SO3(ee_pose[3:6])
	ft_correction_mat = np.dot(robot_to_world.T,world_correction_mat)
	np.save("ft_correction_mat",ft_correction_mat)

def visualize_ft_sensor(robot):
	R = Euler2SO2(np.pi/2)
	fig = plt.figure()
	while True:
		plt.cla()
		plt.xlim(-5,5)
		plt.ylim(-5,5)
		ft = robot.get_ft()[:2]
		ft = np.dot(R,ft)
		plt.quiver(0,0,ft[0],ft[1])
		plt.draw()
		plt.pause(0.01)	

def example_circle_trajectory(robot):
	n_samples = 31
	diameter = 100
	t_arr = np.linspace(0,2*np.pi,n_samples)
	trajectory = np.zeros((n_samples,6))
	trajectory[:,0] = np.cos(t_arr)*diameter/2 - diameter/2
	trajectory[:,1] = np.sin(t_arr)*diameter/2
	robot.primitive_freespace_move(trajectory)


def main():
	ip = '192.168.1.202'

	try:
		robot = RobotController(ip,controller_frequency=100, speed=60, mvacc=100)
		robot.init_robot()

		example_circle_trajectory(robot)
		






		
	finally:
		robot.stop_controller()
		print("Controller stopped")
		robot.stop_printer()


if __name__ == '__main__':
	main()