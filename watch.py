# WATCH: A Distributed Clock Time Offset Estimation Tool for Software-Defined Radio Platforms
# Last Modified - 12/4/2024
# Author - Cassie Jeng

# Import packages
import argparse
import sys
import os
import subprocess
import json
import datetime

# IQ Generation packages
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import lfilter

# PN Generation packages
from pylfsr import LFSR

# WATCH packages
from colorama import Fore, Style
import itertools
import csv
import scipy.io as sio
import json
from scipy import signal, stats
import h5py
from matplotlib import rc
import datetime
rc('xtick',labelsize=14)
rc('ytick',labelsize=14)

# Establish DEBUG & PLOTS capabilities
DEBUG_D = input('\nDEBUG to show intermediate steps? [y/n]:')
DEBUG = False
if DEBUG_D == 'y':
	DEBUG = True

PLOTS = False
if DEBUG:
	PLOTS_D = input('PLOT to show all plots produced from program?\nNote: A plot for all iterations of each link will be produced, in addition to other program plots. These visuals will all need to be viewed, saved (if desired), and closed to be able to proceed with the program. [n] will still show a few plots of each type if DEBUG was enabled. [y/n]:')
	if PLOTS_D == 'y':
		PLOTS = True


###########################################################################################

# IQ Generation functions
def Information_Transmit_p():
	M = 2 # bits per symbol (i.e. 2 in QPSK modulation)
	Info_to_TX=input('\nPlain text message to transmit with 94 characters including spaces:\n')
	if DEBUG:
		print('information to transmit: ',Info_to_TX)

	# converts text string into binary
	binary = ''.join(format(ord(i),'07b') for i in Info_to_TX)
	if DEBUG:
		print('binary equivalent: ',binary)
		print('length of binary: ',len(binary))
	data_bits = np.zeros((len(binary),))
	for i in range(len(binary)):
		data_bits[i] = binary[i]
	if DEBUG:
		print('array equivalent: ',data_bits)

	sync_word = np.asarray([1,1,1,0,1,0,1,1,1,0,0,1,0,0,0,0])
	if DEBUG:
		print('sync word: ',sync_word)
	bit_sequence = np.hstack([sync_word, data_bits])

	preamble_code = np.asarray([1,1,0,0])
	for i in range(16):
		if i == 0:
			preamble_swap = preamble_code
		else:
			preamble = np.hstack([preamble_swap, preamble_code])
			preamble_swap = preamble
	if DEBUG:
		print('preamble: ',preamble)
		print('length of preamble: ',len(preamble))
	QPSK_frame = np.hstack([preamble, bit_sequence])
	if DEBUG:
		print('transmit frame length: ',len(QPSK_frame))
		print('transmit bit sequence: ',QPSK_frame)

	def Serial_to_Parallel(x):
		return x.reshape((len(x)//M, M))
	QPSK_bits = Serial_to_Parallel(QPSK_frame)

	if DEBUG:
		print('QPSK signal to transmit[0:5]:\n',QPSK_bits[0:5])
	return QPSK_bits

def Information_Transmit_r(inphase, quad):
	in_phase = np.asarray(inphase)
	quadrature = np.asarray(quad)
	if DEBUG:
		print('length of in-phase: ', len(in_phase))
		print('length of quadrature: ', len(quadrature))
	QPSK_bits = np.vstack([in_phase, quadrature])
	if DEBUG:
		print('QPSK signal to transmit[0:5]:\n',QPSK_bits[0:5])
	return QPSK_bits

def lut(data, inputVec, outputVec):
	if len(inputVec) != len(outputVec):
		print('ERROR in function lut: Input and Output vectors must have identical length')
		exit()
	output = np.zeros(data.shape)
	eps = np.finfo('float').eps
	for i in range(len(inputVec)):
		# find the indices where data is equal to that input value
		for k in range(len(data)):
			if abs(data[k]-inputVec[i]) < eps:
				# set those indices in the output to be the appropriate output value.
				output[k] = outputVec[i]
	return output

def oversample(x, OS_Rate):
	length = len(x[0])
	x_s = np.zeros((1,length*OS_Rate))
	# fill in one out of every OS_Rate samples with the input values
	count = 0
	h = 0
	for k in range(len(x_s[0])):
		count = count + 1
		if count == OS_Rate:
			x_s[0][k] = x[0][h]
			count = 0
			h = h + 1
	return x_s

def SRRC(alpha, N, Lp):
	# add epsilon to the n values to avoid numerical problems
	ntemp = list(range(-N*Lp, N*Lp+1))
	n = []
	for each in ntemp:
		n.append(each + math.pow(10,-9))
	# plug into time domain formula for the SRRC pulse shape
	h = []
	coeff = 1/math.sqrt(N)
	for each in n:
		sine_term = math.sin(math.pi * each * (1-alpha) / N)
		cosine_term = math.cos(math.pi * each * (1+alpha) / N)
		cosine_coeff = 4 * alpha * each / N
		numerator = sine_term + (cosine_coeff * cosine_term)
		denom_coeff = math.pi * each / N
		denom_part = 1 - math.pow(cosine_coeff, 2)
		denominator = denom_coeff * denom_part
		pulse = coeff * numerator / denominator
		h.append(pulse)
	return h

def write_complex_binary(data, filename):
	# open filename and write array to it as binary, format is interleaved float IQ
	re = np.real(data)
	im = np.imag(data)
	binary = np.zeros(len(data)*2, dtype=np.float32)
	binary[::2] = re
	binary[1::2] = im
	binary.tofile(filename)

def get_samps_from_file(filename):
	# load samples from the binary file
	# file should be in GNURadio's format, i.e., interleaved I/Q samples as float32
	samples = np.fromfile(filename, dtype= np.float32)
	samps = (samples[::2] + 1j*samples[1::2]).astype((np.complex64)) # convert to IQIQIQ
	return samps

# POWDER Experiment Set Up functions
def file_trans(filename, nodes, scp_filename, directory):
	line1 = 'for HOST in'
	for n in nodes:
		line1 = line1 + ' ' + n
	line2 = 'do'
	line3 = '    scp ' + scp_filename + ' $HOST:' + directory
	line4 = 'done'

	with open(filename,'w') as f:
		f.write('#!/bin/bash\n')
		f.write(line1 + '\n')
		f.write(line2 + '\n')
		f.write(line3 + '\n')
		f.write(line4 + '\n')
	
	print(filename + ' script written!')

	command = 'chmod +x ' + filename
	command_arr = command.split(' ')
	result = subprocess.run(command_arr, capture_output=True, text=True)

	command = './' + filename
	command_arr = command.split(' ')
	result = subprocess.run(command_arr, capture_output=True, text=True)

	print(scp_filename + ' transfered to all nodes!')

def replace_characters(line,first,last,new):
	index_first = line.find(first) + len(first)
	index_last = line.find(last, index_first)
	return line[:index_first] + new + line[index_last:]

def find_line(filename, search):
	linecount = 1
	with open(filename,'r') as file:
		for line in file:
			if search in line:
				return linecount, line
			linecount += 1
	return 0,'Not found'

# Over-the-air Narrowband QPSK Modulation and Demodulation Tutorial functions
def get_time_string(timestamp):
	date_time = datetime.datetime.fromtimestamp(int(timestamp))
	return date_time.strftime("%m-%d-%Y, %H:%M:%S")

def JsonLoad(folder, json_file):
	config_file = folder+'/'+json_file
	config_dict = json.load(open(config_file))[0]
	nsamps = config_dict['nsamps']
	rxrate = config_dict['rxrate']
	rxfreq = config_dict['rxfreq']
	wotxrepeat = config_dict['wotxrepeat']
	rxrepeat = config_dict['rxrepeat']
	txnodes = config_dict['txclients']
	rxnodes = config_dict['rxclients']
	return rxrepeat, rxrate, txnodes, rxnodes

def traverse_dataset(meas_folder):
	data = {}
	noise = {}
	txrxloc = {}

	dataset = h5py.File(meas_folder + '/measurements.hdf5', "r")
	for cmd in dataset.keys():
		if cmd == 'saveiq':
			cmd_time = list(dataset[cmd].keys())[0]
			if DEBUG:
				print('Time Collected:', get_time_string(cmd_time))
				print('Command meta data:', list(dataset[cmd][cmd_time].attrs.items()))
		elif cmd == 'saveiq_w_tx':
			cmd_time = list(dataset[cmd].keys())[0]
			if DEBUG:
				print('Time Collected:', get_time_string(cmd_time))
				print('Command meta data:', list(dataset[cmd][cmd_time].attrs.items()))
			for tx in dataset[cmd][cmd_time].keys():
				if tx == 'wo_tx':
					for rx_gain in dataset[cmd][cmd_time][tx].keys():
						for rx in dataset[cmd][cmd_time][tx][rx_gain].keys():
							repeat = np.shape(dataset[cmd][cmd_time][tx][rx_gain][rx]['rxsamples'])[0]
							samplesNotx = dataset[cmd][cmd_time][tx][rx_gain][rx]['rxsamples'][:repeat, :]
							namelist = rx.split('-')
							noise[namelist[1]] = samplesNotx
				else:
					for tx_gain in dataset[cmd][cmd_time][tx].keys():
						for rx_gain in dataset[cmd][cmd_time][tx][tx_gain].keys():
							for rx in dataset[cmd][cmd_time][tx][tx_gain][rx_gain].keys():
								repeat = np.shape(dataset[cmd][cmd_time][tx][tx_gain][rx_gain][rx]['rxsamples'])[0]
								txrxloc.setdefault(tx, []).extend([rx]*repeat)
								rxsamples = dataset[cmd][cmd_time][tx][tx_gain][rx_gain][rx]['rxsamples'][:repeat, :]
								data.setdefault(tx, []).append(np.array(rxsamples))
		else:
			print('Unsupported command: ', cmd)
	return data, noise, txrxloc

def plotOnePSDForEachLink(rx_data, txrxloc, samp_rate=250000, repNums=4):
	for txname in rx_data:
		for i in range(0, len(rx_data[txname]), repNums):
			if PLOTS == False and i > repNums:
				break
			plt.figure()
			plt.psd(rx_data[txname][i][0], Fs = samp_rate/1000)
			plt.ylim(-110,-60)
			plt.yticks(ticks=[-110,-100,-90,-80, -70, -60])
			plt.grid('on')
			plt.title('TX: {} RX: {}'.format(txname, txrxloc[txname][i]))
			plt.xlabel('Frequency (kHz)')
			# PSD plot function has default y axis label: Power Spectral Density (dB/Hz)
			plt.tight_layout()
			plt.show()

def lut_w(data, inputVec, outputVec):
	if(len(inputVec) != len(outputVec)):
		print('Input and Output vectors must have identical length')
		exit()
	output = np.zeros(data.shape)
	eps = np.finfo('float').eps
	for i in range(len(inputVec)):
		for k in range(len(data)):
			if abs(data[k]-inputVec[i]) < eps:
				output[k] = outputVec[i]
	return output

def oversample_w(x,OS_Rate):
	x_s = np.zeros(len(x)*OS_Rate)
	x_s[::OS_Rate] = x
	return x_s

def SRRC_w(alpha, N, Lp):
	n = np.arange(-N*Lp+ (1e-9), N*Lp+1)
	h = np.zeros(len(n))
	coeff = 1/np.sqrt(N)
	for i, each in enumerate(n):
		sine_term = np.sin(np.pi * each * (1-alpha) / N)
		cosine_term = np.cos(np.pi * each * (1+alpha) / N)
		cosine_coeff = 4 * alpha * each / N
		numerator = sine_term + (cosine_coeff * cosine_term)
		denom_coeff = np.pi * each / N
		denom_part = 1 - cosine_coeff**2
		denominator = denom_coeff * denom_part
		h[i] = coeff * numerator / denominator
	return h

def binary2mary(data, M):
	log2M = round(np.log2(M))
	if (len(data) % log2M) != 0:
		print('Input to binary2mary must be divisible by log2(m).')
	data.shape = (len(data)//log2M, log2M)
	binaryValuesArray = 2**np.arange(log2M)
	marydata = data.dot(binaryValuesArray)
	return marydata

def createPreambleSignal(A, N, alpha, Lp):
	preamble = np.tile([1, 1, 0, 0], 16)
	data = binary2mary(preamble, 4)

	# Modulation
	inputVec = [0, 1, 2, 3]
	outputVecI = [A, -A, A, -A]
	outputVecQ = [A, A, -A, -A]
	xI = lut_w(data, inputVec, outputVecI)
	xQ = lut_w(data, inputVec, outputVecQ)

	# Upsample
	x_s_I = oversample_w(xI, N)
	x_s_Q = oversample_w(xQ, N)
	
	# Pulse-shape filter
	pulse = SRRC_w(alpha, N, Lp)
	s_0_I = np.convolve(x_s_I, pulse, mode='full')
	s_0_Q = np.convolve(x_s_Q, pulse, mode='full')

	return (s_0_I + 1j*s_0_Q), pulse

def crossCorrelationMax(rx0, packetSignal, peaks_arr, ptdg):
	# Cross correlate with the original packet to find it in the noisy signal
	lags = signal.correlation_lags(len(rx0), len(packetSignal), mode='same')
	xcorr_out = signal.correlate(rx0, packetSignal, mode='same')
	xcorr_mag = np.abs(xcorr_out)

	length_of_packet = 3200 # hard coded
	maxIndex = np.argmax(xcorr_mag[:len(xcorr_mag)-length_of_packet])
	lagIndex = lags[maxIndex]

	# Calculate length of packet
	peak_1 = lags[np.argmax(xcorr_mag)]
	short_xcorr_mag = np.concatenate((xcorr_mag[:peak_1-300], xcorr_mag[peak_1+300:]))
	peak_2 = np.argmax(short_xcorr_mag)
	if peak_2 > peak_1-300:
		peak_2 += 300
	peak_2 = lags[peak_2]
	peaks = np.array([peak_1, peak_2])
	peaks_arr.append(abs(peak_1-peak_2))

	if ptdg:
		plt.figure()
		# plt.plot(lags, xcorr_mag, label='|X-Correlation|')
		plt.plot(lags, xcorr_mag)
		plt.legend()
		# plt.plot(peaks, xcorr_mag[peaks], "x")
		plt.ylabel('|X-Correlation| with TX Packet', fontsize=14)
		plt.xlabel('RX Packet Sample Index', fontsize=14)
		plt.ylim((0,1))
		plt.tight_layout()
		plt.show()

	return lagIndex

# WATCH Offset Estimation Functions
def make_delta(col_num, links):
	file_name = "col_" + str(col_num) + ".txt"
	d = 0
	delta = np.zeros((len(links),1))

	with open(file_name,'r') as f:
		vals = f.readlines()
	
	for val in vals:
		val = val.replace(",\n", "")
		delta[d] = (int(val))
		d += 1
	return delta

def find_e_vector(delta, A, pinvA, rx_names, snr, weighted=0):
	if weighted:
		AT = np.transpose(A)
		snr_t = snr.reshape((snr.shape[0],))
		W = np.diag(snr_t)
		estimate = np.dot(np.dot(np.dot(np.linalg.pinv(np.dot(np.dot(AT,W),A)),AT),W),delta)
	else:
		estimate = np.dot(pinvA, delta)
	
	estimate = np.vstack(([0],estimate))

	e_est = estimate[:len(rx_names)]
	T_est = estimate[len(rx_names):]
	return e_est, T_est, estimate

def correct_difference(col_num, delta, rx_names, off):
	s = 0
	limit = len(rx_names)-1
	while s < len(rx_names):
		section = delta[s*(len(rx_names)-1):(s+1)*(len(rx_names)-1)]
		small = []
		lag_ind = 0
		for lag in section:
			ind = 0
			while(ind < limit):
				if abs(lag - section[ind]) > 2000: # hard coded
					min_val = min(lag, section[ind])[0]
					if min_val == lag:
						small.append([lag_ind,min_val])
					else:
						small.append([ind,min_val])
				ind += 1
			lag_ind += 1

		small.sort()
		small = list(small for small,_ in itertools.groupby(small))
		for sm in small:
			if DEBUG:
				# print(' ----- Estimate values corrected from wrap-around ----- ')
				print('CORRECTED: Iteration '+str(col_num)+', Section '+str(s)+', Index '+str(sm[0])+' -- '+str(sm[1]))
			delta[(sm[0] + s*limit)] += off # 4072, 4096
		s += 1
	return delta

def calculate_SNR(rx_data, samp_rate=250000):
	Pxx, freqs = plt.psd(rx_data, Fs = samp_rate/1000)
	# too many hard coded values in this function
	center_ind = int(len(Pxx)/2)
	lower = center_ind - 10
	upper = center_ind + 10

	n_lower = center_ind - 25
	n_upper = center_ind + 25

	s = np.mean(np.square(Pxx[lower:upper])) # signal + noise
	n = np.mean(np.square(np.hstack((Pxx[:n_lower], Pxx[n_upper:])))) # noise
	snr = max((s/n) - 1, 1, 1.0e-5)
	snr = min(snr, 100)
	return snr, Pxx

def make_snr_vecs(col_num, links):
	file_name = "snr_" + str(col_num) + ".txt"
	sv = 0
	snr_vec = np.zeros((len(links),1))

	with open(file_name,'r') as f:
		vals = f.readlines()

	for val in vals:
		val = val.replace(",\n", "")
		snr_vec[sv] = val
		sv += 1
	return snr_vec

def plot_snr_error(snr, error, col_num, samp_rate):
	inv_snr = np.reciprocal(snr)
	for i in range(len(error)):
		error[i] = samples_to_us(error[i],samp_rate)

	plt.figure()
	plt.scatter(inv_snr, error)
	plt.xlabel('1/SNR (1/dB)')
	plt.ylabel('Estimation Error (us)')
	plt.grid()
	plt.title('Estimation Error vs. 1/SNR for Each Link in Iteration ' + str(col_num))
	plt.show()
	return inv_snr

def least_sq_error(col_num, estimate, delta, A, links, ptdg, samp_rate):
	est_delta = np.dot(A,estimate[1:])
	error = abs(delta - est_delta) #in samples
	for i in range(len(error)):
		error[i] = samples_to_us(error[i],samp_rate)

	i = np.array([*range(len(links))]).reshape(len(links),1) # counting links [0,1,2,...len(links)-1]

	if ptdg:
		#plotting error for each link
		print(' ----- Plotting Error by Link for Iteration ' + str(col_num) + ' ----- ')
		print('x-axis is links in the experiment, in order based on the node order specified earlier in program.')
		plt.plot(i,error)
		plt.grid()
		plt.xlabel('Link')
		plt.ylabel('Estimation Error (us)')
		plt.title('Estimation Error by Link for Iteration ' + str(col_num))
		plt.show()

		# plotting histogram of error for each link
		print(' ----- Histogram of Link Error for Iteration ' + str(col_num) + ' ----- ')
		plt.hist(error)
		plt.grid()
		plt.xlabel('Histogram of Estimation Error by Link for Iteration ' + str(col_num))
		plt.show()

		# plotting histogram of MSE for each link
		# print(' ----- Histogram Link Squared Error for Iteration ' + str(col_num) + ' ----- ')
		# plt.hist(np.square(error * (1/250000)))
		# plt.grid()
		# plt.xlabel('Squared Error (T^2)')
		# plt.title('Histogram of Squared Estimation Error for Iteration ' + str(col_num))
		# plt.show()

	# finding root mean squared error for repNum (col_num)
	RMSE = math.sqrt(np.square(error).mean())
	if DEBUG:
		print(' ----- Root Mean Squared Error for Iteration ' + str(col_num) + ' ----- ')
		print(format(samples_to_us(RMSE,samp_rate),'.4f') + ' us')
	return RMSE, error

def samples_to_us(value, samp_rate):
	#for ms: (float)((value/samp_rate)*1000)
	return (float)((value/samp_rate)*1000000)

def print_results_us(col_num, rx_names, e_est, samp_rate, lgr_ms, smr_ms):
	print(" -------- Iteration " + str(col_num) + " -------- ")

	max_len = -1
	for each in rx_names:
		if len(each) > max_len:
			max_len = len(each)

	title = 'rx_name'
	if len(title) > max_len:
		max_len = len(title)
	
	if len(title) < max_len:
		for i in range(max_len-len(title)):
			title += " "

	print(Fore.RED + "(~, " + title + ") ----- offset from " + rx_names[0] + " in us" + Style.RESET_ALL)
	r = 0

	for rx in rx_names:
		rx_temp = rx
		if len(rx) != max_len:
			for i in range(max_len-len(rx)):
				rx_temp += " "
		delay = samples_to_us(e_est[r][0],samp_rate)
		if delay < 0:
			print("(~, " + rx_temp + ") ----- [" + format(delay,'.3f') + "]")
		else:
			print("(~, " + rx_temp + ") ----- [" + format(delay,'.4f') + "]")

		if abs(delay) >= 1000: #1 millisecond
			lgr_ms.append(delay)
		else:
			smr_ms.append(delay)
		r += 1
	
	return lgr_ms, smr_ms

def round_string(RMSE):
	round_i = 6
	while(len(str(RMSE)) > len('iteration')):
		RMSE = round(RMSE,round_i)
		round_i -= 1
	return RMSE

###########################################################################################

# Start Program
print('\nWATCH: A Distributed Clock Time Offset Estimation Tool on POWDER')
print('Initializing program....\n')

# Which section(s) of program to run
prog_section = input('Enter (1) to run entire program (IQ generation, pre-processing, experiment, WATCH post-processing)\nEnter (2) to run just WATCH post-processing\n(1/2): ')
if prog_section == '2':
	folder = input('Input name of Shout results folder you would like to process with WATCH. Ensure folder is in current working directory. Format should match Shout_meas_MM-DD-YYYY_HH-MM-SS: ')
	fn = input('Input name of IQ file transmitted in chosen Shout results. Ensure file is in current working directory. Include .iq extension: ')

###########################################################################################

if prog_section == '1':
	# Choose which transmit data type
	data_d = input('\nInput option for IQ generation. p for plain text with preamble, r for pn codes with no preamble:')

	# IQ Generation for Plain Text Data & PN Code Data

	print('\n##################################################################\n')
	print('STEP 1: IQ Generation: Create IQ File for Message to Transmit\n')
	if data_d == 'p':
		print('Generating IQ file for Plain Text Data with Preamble')
		print('Program derived from Over-the-air Narrowband QPSK Modulation and Demodulation Tutorial, MWW 2023')
		Frame = Information_Transmit_p()
		data1 = []
		for i in range(len(Frame)):
			data1.append(2*Frame[i][0]+Frame[i][1])
		Dec4_data = np.array(data1)
		if DEBUG:
			print('decimal number equivalent: ',Dec4_data)

		# Modulation
		# INPUT: data
		# OUTPUT: modulated values, x
		A = math.sqrt(9/2)
		inputVec = [0,1,2,3]
		outputVecI = [A,-A,A,-A]
		outputVecQ = [A,A,-A,-A]
		xI = lut(Dec4_data, inputVec, outputVecI)
		xQ = lut(Dec4_data, inputVec, outputVecQ)
		xI = xI.reshape((1,len(Dec4_data)))
		xQ = xQ.reshape((1,len(Dec4_data)))

		# Upsample
		# INPUT: modulated values, x
		# OUTPUT: modulated values at sampling rate, x_s
		x_s_I = oversample(xI,8)
		x_s_Q = oversample(xQ,8)

		pulse = SRRC(0.5,8,6)
		pulse = np.array(pulse)
		pulse = np.reshape(pulse,pulse.size)
		plt.figure()
		# plt.plot(pulse,label='SRRC pulse shape')
		plt.plot(pulse)
		plt.title('SRRC pulse shape')
		plt.legend()

		x_s_I = np.reshape(x_s_I, x_s_I.size)
		x_s_Q = np.reshape(x_s_Q, x_s_Q.size)
		s_0_I = np.convolve(x_s_I, pulse, mode='full')
		s_0_Q = np.convolve(x_s_Q, pulse, mode='full')

		# create complex IQ value from the I and Q components
		QPSK_samples = s_0_I + s_0_Q*1j
		if DEBUG:
			print('QPSK_samples[0:10]: ', QPSK_samples[0:10])
		QPSK_samples_Final = np.hstack([np.zeros(1024, dtype=type(QPSK_samples[0])),QPSK_samples])
		if DEBUG:
			plt.figure()
			plt.plot(np.real(QPSK_samples_Final[1700:2000]),label='Real Signal')
			plt.plot(np.imag(QPSK_samples_Final[1700:2000]),label='Imag Signal')
			plt.xlabel('Packet to TX')
			plt.ylabel('Imag and Real Amplitude')
			plt.title('Imag and Real Samples to TX')
			plt.grid('on')
			plt.legend()
			print('QPSK_samples_Final[1020:1040]:\n',QPSK_samples_Final[1020:1040])

		fn = input('\nfilename for binary IQ (without .iq extension):\n')
		fn = fn + '.iq'
		write_complex_binary(QPSK_samples_Final,fn)
		if DEBUG:
			print(get_samps_from_file(fn)[-150:-140])
	elif data_d == 'r':
		print('Generating IQ file for PN Codes without Preamble')
		print('PN Program deprived from Dr. Neal Patwari Reference Function for LFSR in MATLAB, Washington University in St. Louis')
		print('Program derived from Over-the-air Narrowband QPSK Modulation and Demodulation Tutorial, MWW 2023')

		print('\nIn-Phase Sequence:')
		state = input('initial state vector as string (enter for default 000000001 i.e. [0,0,0,0,0,0,0,0,1]): ')
		taps = input('taps as string (enter for default 95 i.e. [9,5]): ')

		if state == '':
			state = [0,0,0,0,0,0,0,0,1]
		else:
			state = list(state)
			ll = 0
			for l in state:
				state[ll] = int(l)
				ll += 1
		
		if taps == '':
			taps = [9,5]
		else:
			taps = list(taps)
			ll = 0
			for l in taps:
				taps[ll] = int(l)
				ll += 1
		
		L = LFSR(initstate=state, fpoly=taps)
		pn_iSeq = L.getFullPeriod()

		print('\nQuadrature Sequence:')
		state = input('initial state vector as string (enter for default 000000001 i.e. [0,0,0,0,0,0,0,0,1]): ')
		taps = input('taps as string (enter for default 9643 i.e. [9,6,4,3]): ')

		if state == '':
			state = [0,0,0,0,0,0,0,0,1]
		else:
			state = list(state)
			ll = 0
			for l in state:
				state[ll] = int(l)
				ll += 1
		
		if taps == '':
			taps = [9,6,4,3]
		else:
			taps = list(taps)
			ll = 0
			for l in taps:
				taps[ll] = int(l)
				ll += 1
		
		L = LFSR(initstate=state, fpoly=taps)
		pn_qSeq = L.getFullPeriod()
		if DEBUG:
			print('\nPN In-Phase Sequence:\n',pn_iSeq)
			print('\nPN Quadrature Sequence:\n',pn_qSeq)
		
		Frame = Information_Transmit_r(pn_iSeq, pn_qSeq)
		if DEBUG:
			print(Frame.shape)
		
		# Modulation
		# INPUT: data
		# OUTPUT: modulated values, x
		A = math.sqrt(9/2)
		xI_list = []
		for each in Frame[0]:
			if each == 0:
				xI_list.append(-1)
			else:
				xI_list.append(1)
		
		xQ_list = []
		for each in Frame[1]:
			if each == 0:
				xQ_list.append(-1) # -1j
			else:
				xQ_list.append(1) # 1j
		xI_list_new = [i * A for i in xI_list]
		xQ_list_new = [i * A for i in xQ_list]

		xI = np.array(xI_list_new)
		xQ = np.array(xQ_list_new)

		xI = xI.reshape((1,len(Frame[0])))
		xQ = xQ.reshape((1,len(Frame[1])))
		if DEBUG:
			print(xI.shape)

		# Upsample
		# INPUT: modulated values, x
		# OUTPUT: modulated values at sampling rate, x_s
		x_s_I = oversample(xI,8)
		x_s_Q = oversample(xQ,8)

		pulse = SRRC(0.5,8,6)
		pulse = np.array(pulse)
		pulse = np.reshape(pulse,pulse.size)
		plt.figure()
		plt.plot(pulse)
		plt.title('SRRC pulse shape')
		#plt.legend()

		x_s_I = np.reshape(x_s_I, x_s_I.size)
		x_s_Q = np.reshape(x_s_Q, x_s_Q.size)
		s_0_I = np.convolve(x_s_I, pulse, mode='full')
		s_0_Q = np.convolve(x_s_Q, pulse, mode='full')

		# create complex IQ values from the I and Q components
		QPSK_samples = s_0_I + s_0_Q*1j
		if DEBUG:
			print('QPSK_samples[0:10]:\n',QPSK_samples[0:10])
		QPSK_samples_Final = np.hstack([np.zeros(1024, dtype=type(QPSK_samples[0])),QPSK_samples])
		if DEBUG:
			plt.figure()
			plt.plot(np.real(QPSK_samples_Final[1700:2000]),label='Real Signal')
			plt.plot(np.imag(QPSK_samples_Final[1700:2000]),label='Imag Signal')
			plt.xlabel('Packet to TX')
			plt.ylabel('Imag and Real Amplitude')
			plt.title('Imag and Real Samples to TX')
			plt.grid('on')
			plt.legend()
			print('QPSK_samples_Final[1020:1040]:\n',QPSK_samples_Final[1020:1040])
		
		fn = input('\nfilename for binary IQ (without .iq extension):\n')
		fn = fn + '.iq'
		write_complex_binary(QPSK_samples_Final,fn)
		if DEBUG:
			print(get_samps_from_file(fn)[-150:-140])
	else:
		print('Error: Choose a valid input option for IQ generation')
		print('python ' + sys.argv[0] + ' [-d --data: p/r]')
		exit()

	###########################################################################################

	step2 = input('\nContinue with Resource Reservation? (y/n): ')
	if step2 != 'y':
		exit()

	print('\n##################################################################\n')
	print('STEP 2: Reserve Resources & Instantiate an Experiment using POWDER')
	print('\n ----- Navigate to https://powderwireless.net/ ----- ')
	print('1. Login\n2. Create a resource reservation with desired nodes\n3. Start an experiment using the shout-long-measurement profile or create a new profile via Experiments -> Create Experiment Profile -> Git Repo -> add repo link (https://gitlab.flux.utah.edu/frost/proj-radio-meas) -> select the profile')
	print('Note: Make sure to check \"Start X11 VNC on all compute nodes\". You may need to schedule an experiment for a later time based on resource availability.\n')
	
	###########################################################################################
	
	step3 = input('\nContinue with Initializing Node Sessions? (y/n): ')
	if step3 != 'y':
		exit()

	print('\n##################################################################\n')
	print('STEP 3: Set up POWDER Experiment: All Node and Orchestrator Sessions')
	print('\n ----- Open the created POWDER experiment ----- ')
	print('1. Navigate to List View for the experiment nodes.\n2. Click the check box next to the gear symbol in the header of the experiment node list. Click the gear symbol drop down menu and reboot all nodes. Wait until all nodes are ready again.')

	##### Creating filetransfer.sh #####
	ssh_command = input('List each full [username]@[node-name] from the SSH command column for orch and all comp nodes, starting with orch, separated by a space: ')
	nodes = ssh_command.split(' ')
	user_arr = nodes[0].split('@')
	username = str(user_arr[0])
	if DEBUG:
		print('\nusername: ' + username)

	ft_filename = 'filetransfer.sh'
	ft_directory = '/local/repository/shout/signal_library'

	##### Set up Sessions for Nodes #####
	print('\n##### Setting up Sessions for Experiment! #####')

	num_sessions = len(nodes) + 1
	print('\nOpen ' + Fore.RED + str(num_sessions) + Style.RESET_ALL + ' new Terminal windows. Two for the orchestrator and one for each experiment node.')
	print('In the first orch terminal window, run the following command to start ssh and tmux sessions:\n')
	print(Fore.RED + 'ssh -Y -p 22 -t ' + nodes[0] + ' \'cd /local/repository/bin && tmux new-session -A -s shout1 &&  exec $SHELL\'\n' + Style.RESET_ALL)
	print('In the second orch terminal window, run the following command:\n')
	print(Fore.RED + 'ssh -Y -p 22 -t ' + nodes[0] + ' \'cd /local/repository/bin && tmux new-session -A -s shout2 &&  exec $SHELL\'\n' + Style.RESET_ALL)

	print('Run the following ' + str(len(nodes) - 1) + ' commands in the terminal windows opened for each experiment node.\n')
	for n in range(1,len(nodes)):
		print(Fore.RED + 'ssh -Y -p 22 -t ' + nodes[n] + ' \'cd /local/repository/bin && tmux new-session -A -s shout &&  exec $SHELL\'' + Style.RESET_ALL)

	print('\nSessions initialized!')

	###########################################################################################

	step4 = input('\nContinue with Modifying Node Files & Setting up for Experiment? (y/n): ')
	if step4 == 'n':
		exit()

	print('\n\n##################################################################\n')
	print('STEP 4: Set up Nodes for Experiment: IQ file, meascli.py, 3.run_cmd.sh, save_iq_w_tx_file.json')

	## Transferring IQ file to all nodes
	file_trans(ft_filename,nodes,fn,ft_directory)

	## Getting meascli.py from node
	command = 'scp ' + nodes[0] + ':/local/repository/shout/meascli.py .'
	command_arr = command.split(' ')
	scp_result = subprocess.run(command_arr, capture_output=True, text=True)

	## Editing external clock line
	ms_filename = 'meascli.py'
	linenum, clock_line = find_line(ms_filename,'useexternalclock = False')
	if linenum != 0 and clock_line != 'Not found':
		new_line = clock_line.replace('False','True')
		if DEBUG:
			print('Line Number: ' + str(linenum))
			print('Line: ' + clock_line)
			print('Line: ' + new_line)
		with open(ms_filename,'r') as f:
			ms_lines = f.readlines()
		ms_lines[linenum-1] = new_line
		with open(ms_filename,'w') as f:
			f.writelines(ms_lines)

		## Transferring meascli.py back to nodes
		ms_directory = '/local/repository/shout'
		file_trans('meascli_transfer.sh',nodes,ms_filename,ms_directory)

	## Checking 3.run_cmd.sh file on nodes
	command = 'scp ' + nodes[0] + ':/local/repository/bin/3.run_cmd.sh .'
	command_arr = command.split(' ')
	cmd_result = subprocess.run(command_arr, capture_output=True, text=True)

	cmd_filename = '3.run_cmd.sh'
	linenum, found_line = find_line(cmd_filename,'save_iq_w_tx_file')
	if linenum == 0 and found_line == 'Not found':
		# didn't find correct line for CMD so need to correct line
		linen, subline = find_line(cmd_filename,'for CMD in')
		new_l = '	for CMD in \"save_iq_w_tx_file\"\n'
		with open(cmd_filename,'r') as f:
			cmd_lines = f.readlines()
		cmd_lines[linen-1] = new_l
		with open(cmd_filename,'w') as f:
			f.writelines(cmd_lines)
		cmd_directory = '/local/repository/bin'
		file_trans('cmd_transfer.sh',nodes,cmd_filename,cmd_directory)

	## Modifying JSON file on nodes
	json_filename = 'save_iq_w_tx_file.json'
	json_directory = '/local/repository/etc/cmdfiles'
	command = 'scp ' + nodes[0] + ':' + json_directory + '/' + json_filename + ' .'
	command_arr = command.split(' ')
	json_result = subprocess.run(command_arr, capture_output=True, text=True)

	## Which lines would you like to modify in the JSON file?
	print(Fore.RED + '\nThe following are in reference to the TX JSON file. For each, if you would like to keep the default, press enter. If you would like to change the default, type the new value you would like in the same format as the given default.\n' + Style.RESET_ALL)

	txrate_c = input('Change default txrate/rxrate [250e3]?: ')
	txfreq_c = input('Change default txfreq/rxfreq [3455e6]?: ')
	txgain_c = input('Change default txgain [27.0]?: ')
	rxgain_c = input('Change default rxgain [30.0]?: ')
	rxrepeat_c = input('Change default rxrepeat [4]?: ')

	rxrate_c = txrate_c
	rxfreq_c = txfreq_c

	## Replacing default values
	elements = ['txrate', 'rxrate', 'txfreq', 'rxfreq', 'rxrepeat']
	t_count = 0

	for t in [txrate_c, rxrate_c, txfreq_c, rxfreq_c, rxrepeat_c]:
		if len(t) > 0: # only go through replacement if new value to replace with
			linenum, line = find_line(json_filename, elements[t_count])
			if linenum == 0 and line == 'Not found':
				print('Error in finding entry value to replace.')
			replace = t + ','
			new_line = replace_characters(line,': ', '\n', replace)
			if DEBUG:
				print('new ' + elements[t_count] + ': ' + new_line)

			with open(json_filename,'r') as f:
				lines = f.readlines()
			lines[linenum-1] = new_line
			with open(json_filename,'w') as f:
				f.writelines(lines)
		t_count += 1

	#txgain/rxgain
	elements = ['txgain','rxgain']
	t_count = 0
	for t in [txgain_c, rxgain_c]:
		if len(t) > 0:
			linenum, line = find_line(json_filename,elements[t_count])
			if linenum == 0 and line == 'Not found':
				print('Error in finding entry value to replace.')
			new_line = replace_characters(line,'fixed\": ', '}',t)
			if DEBUG:
				print('new ' + elements[t_count] + ': ' + new_line)
			with open(json_filename,'r') as f:
				lines = f.readlines()
			lines[linenum-1] = new_line
			with open(json_filename,'w') as f:
				f.writelines(lines)
		t_count += 1

	## Advanced elements of JSON file

	#txsamps
	linenum, line = find_line(json_filename, 'txsamps')
	if linenum == 0 and line == 'Not found':
		print('Error in finding txsamps in JSON file.')
	replace = ft_directory + '/' + fn
	new_line = replace_characters(line,'\"file\": \"', '\"},',replace)
	with open(json_filename,'r') as f:
		lines = f.readlines()
	lines[linenum-1] = new_line
	with open(json_filename,'w') as f:
		f.writelines(lines)
	if DEBUG:
		print('new txsamps: ' + new_line)
		print('txsamps updated in JSON file.')

	#txclients/rxclients
	clients_c = input('Please copy the ID names of each comp node client in your experiment, separated by a space [cbrssdr1-bes-comp cbrssdr1-browning-comp]. These are in the List View of the experiment: ')
	clients = clients_c.split(' ')
	clients_str = ''
	for c in clients:
		clients_str = clients_str + '\"' + c + '\", '
	clients_str = clients_str[:-2] #remove extra ', ' from last client
	for c in ['txclients','rxclients']:
		linenum, line = find_line(json_filename,c)
		if linenum == 0 and line == 'Not found':
			print('Error in finding ' + c + ' in JSON file.')
		new_line = replace_characters(line,'[',']',clients_str)
		with open(json_filename,'r') as f:
			lines = f.readlines()
		lines[linenum-1] = new_line
		with open(json_filename,'w') as f:
			f.writelines(lines)
		if DEBUG:
			print('new ' + c + ': ' + new_line)
			print(c + ' updated in JSON file.')

	if DEBUG:
		print('Done! JSON file ready.')

	# transfer JSON file back to nodes
	file_trans('json_transfer.sh',nodes,json_filename,json_directory)

	## Confirm connection to nodes after all modifications
	print('\nFor any cellsdr1-[site]-comp and cbrssdr1-[site]-comp nodes in the experiment, confirm connection by running ' + Fore.RED + 'uhd_usrp_probe' + Style.RESET_ALL + '.\nIf a node complains about a firmware mismatch:\n1. Run ' + Fore.RED + './setup_x310.sh' + Style.RESET_ALL + ' on that node.\n2. Power cycle/reboot that node using the gear symbol in the POWDER Experiment List View.\n3. Run ' + Fore.RED + 'uhd_usrp_probe' + Style.RESET_ALL + ' on the node again. Repeat if needed.\n')

	###########################################################################################

	step5 = input('\nContinue Running Shout Data Collection? (y/n): ')
	if step5 != 'y':
		exit()

	print('\n##################################################################\n')
	print('STEP 5: Running POWDER Experiment\n')
	print('Run the following commands in the specified node/orch Terminal window:\n')

	## EXPERIMENT COLLECTION INSTRUCTIONS
	print('1. In the first ' + Fore.RED + 'orch' + Style.RESET_ALL + ' session, run the command to start the orch: ' + Fore.RED + './1.start_orch.sh' + Style.RESET_ALL)
	print('2. In each of the ' + Fore.RED + 'non-orch nodes' + Style.RESET_ALL + ' sessions, run the following two commands in sequence to resize the buffer and start the clients:\n' + Fore.RED + 'sudo sysctl -w net.core.wmem_max=24862979\n./2.start_client.sh\n' + Style.RESET_ALL + 'Wait until all non-orch nodes say \"Waiting for command...\"')
	print('3. In the second ' + Fore.RED + 'orch' + Style.RESET_ALL + ' session, run the command to initiate data collection: ' + Fore.RED + './3.run_cmd.sh' + Style.RESET_ALL)
	print('You should see data collection information printed in STDOUT of the second orch session during collection.')
	print('\nWAIT to continue until the second orch session returns to the command prompt.')

	conf_meas = input('\nContinue? (y/n): ')
	if conf_meas != 'y':
		exit()

	print('\n##################################################################\n')
	print('Confirming and transferring measurements from remote host...')
	today = datetime.datetime.now()
	today_month = str(today.strftime("%m"))
	today_day = str(today.strftime("%d"))
	today_year = str(today.strftime("%Y"))
	today_hour = str(today.strftime("%H"))

	shout_folder_options = 'Shout_meas_' + today_month + '-' + today_day + '-' + today_year + '_' + today_hour + '*'
	if DEBUG:
		print('Folder to SCP from remote host: ' + shout_folder_options)

	# SCP data collection results from remote host
	command = 'scp -r ' + nodes[0] + ':/local/data/' + shout_folder_options + ' .'
	command_arr = command.split(' ')
	folder_result = subprocess.run(command_arr, capture_output=True, text=True)
	
	print('If no folders/files were transferred from remote host, run ' + Fore.RED + 'ls /local/data/' + Style.RESET_ALL + ' on orch node and check if there is a folder entitled \'Shout_meas_MM-DD-YYYY_HH-MM-SS\' where MM-DD-YYYY is the date of collection and HH-MM-SS is the time of collection. If this folder does exist, manually run the following command, updating the necessary fields, to scp the data to your local host. The folder should have three files: log, measurements.hdf5, and save_iq_w_tx_file.json: \n' + Fore.RED + 'scp -r <username>@<orch_node_hostname>:/local/data/Shout_meas_MM-DD-YYYY_HH-MM-SS .' + Style.RESET_ALL)
	
	pre = [filename for filename in os.listdir('.') if filename.startswith(shout_folder_options[:-1])]
	if len(pre) > 0:
		folder = pre[0]
		if DEBUG:
			print('Data collection folder found: ' + folder)
	else:
		folder = input('No data collection folder with that name found in local directory. To continue with WATCH post processing, input name of folder where Shout results were saved. Format should match Shout_meas_MM-DD-YYYY_HH-MM-SS: ')

	###########################################################################################

	step6 = input('\nContinue with Offset Estimation with WATCH? (y/n): ')
	if step6 != 'y':
		exit()

################# Continue whether choice was '1' or '2': WATCH Post-Processing ###################

print('\n##################################################################\n')
print('STEP 6: Offset Estimation with WATCH & Full-packet Cross Correlation\n')
print('WATCH: A Distributed Clock Time Offset Estimation Tool on POWDER')
print('Author: Cassie Jeng, August 2023, Version 0.1')
print('Parse collected HDF5 data files from SHOUT, calculate the index offset from the beginning of the RX packet to the index of highest cross-correlation with full transmitted packet, estimate the distributed clock time offset at each of the experiment nodes from the network\'s global time')
print('Also includes: PSD, SNR, and LSE/RMSE analysis\n')

print('Over-the-air Narrowband QPSK Modulation and Demodulation: from MMW 2023')
print('Authors: Cassie Jeng, Neal Patwari, Aarti Singh, Jie Wang, Meles Gebreyesus Weldegebriel\n')

# Loading Data
IQ_filename = fn # name used in IQ generation for file
print('Loading data from ' + folder[11:21] + ' data collection.\n')

jsonfile = 'save_iq_w_tx_file.json'
rxrepeat, samp_rate, txlocs, rxlocs = JsonLoad(folder, jsonfile)
rx_data, _, txrxloc = traverse_dataset(folder)

# setting up the links
rx_names = []
for txl in txlocs:
	if txl.split('-')[0] == 'cbrssdr1':
		rx_names.append(txl.split('-')[1])
	else:
		rx_names.append(txl.replace('-','_'))

rx_names = sorted(rx_names)
if DEBUG:
	print(' ----- rx_names ----- ')
	print(rx_names)

links_names = []
for name in rx_names:
	for n in rx_names:
		if name != n:
			links_names.append(name + '-' + n)

if DEBUG:
	print('\n')
	print(' ----- links_names ----- ')
	print(links_names)

links = []
for pair in links_names:
	pair = pair.split("-")
	links.append(pair)

if DEBUG:
	print('\n')
	print(" ----- all links ----- ")
	print(links)

# PSD plots per link for first iteration (repNum)
# plotPSDYN = input("Would you like to visualize the PSD plots for the first iteration on each link? [y/n]: ")
# if plotPSDYN == 'y':
if DEBUG:
	print('Visualizing the PSD plots for the first iteration of each link:')
	plotOnePSDForEachLink(rx_data, txrxloc, samp_rate, repNums=rxrepeat)

# Calculate Lags
A = np.sqrt(9/2)
N = 8
alpha = 0.5
Lp = 6

lag_data = []
snr_data = []
peaks_arr = []

plt_cnt = 0

for tx in txlocs: #rx_names:
	for rx in txlocs: #rx_names:
		if tx != rx:
			lags_row = []
			snr_row = []
			for repNum in range(rxrepeat):
				# pick tx - rx pair
				#txloc = 'cbrssdr1-' + tx + '-comp'
				#rxloc = 'cbrssdr1-' + rx + '-comp'
				txloc = tx
				rxloc = rx

				rx_data[txloc] = np.vstack(rx_data[txloc])
				rxloc_arr = np.array(txrxloc[txloc])
				rx0 = rx_data[txloc][rxloc_arr==rxloc][repNum]

				# Low Pass Filtering to out_of_band frequency components
				stopband_attenuation = 60.0
				transition_bandwidth = 0.05
				cutoff_norm = 0.15
				filterN, beta = signal.kaiserord(stopband_attenuation, transition_bandwidth)
				taps = signal.firwin(filterN, cutoff_norm, window=('kaiser',beta))
				filtered_rx0 = signal.lfilter(taps, 1.0, rx0)

				preambleSignal, pulse = createPreambleSignal(A,N,alpha,Lp)
				packetSignal = get_samps_from_file(IQ_filename)
				if DEBUG and PLOTS:
					lagIndex = crossCorrelationMax(filtered_rx0, packetSignal, peaks_arr, True)
				elif DEBUG and not PLOTS:
					if plt_cnt < 2:
						lagIndex = crossCorrelationMax(filtered_rx0, packetSignal, peaks_arr, True)
					else:
						lagIndex = crossCorrelationMax(filtered_rx0, packetSignal, peaks_arr, False)
				else:
					lagIndex = crossCorrelationMax(filtered_rx0, packetSignal, peaks_arr, False)
				snr, pxx = calculate_SNR(rx0)

				lags_row.append(lagIndex)
				snr_row.append(snr)
				plt_cnt += 1
			lag_data.append(lags_row)
			snr_data.append(snr_row)

lag_data = np.array(lag_data)
snr_data = np.array(snr_data)
off = 4072 # hard coded -- fix

for r in range(rxrepeat):
	file_name = 'col_' + str(r+1) + '.txt'
	with open(file_name,'w') as f:
		for dat in lag_data[:,r]:
			f.write(str(dat) + ',\n')

if DEBUG:
	print(' ----- Lag data written to files ----- ')

for r in range(rxrepeat):
	file_name = 'snr_' + str(r+1) + '.txt'
	with open(file_name,'w') as f:
		for dat in snr_data[:,r]:
			f.write(str(dat) + ',\n')

if DEBUG:
	print(' ----- SNR data written to files ----- ')
	print("Packet Length:", off)

# Setting up the A Matrix
A = np.zeros((len(links),2*len(rx_names)))

link_num = 0
for link in links:
	tx_num = rx_names.index(link[0])
	rx_num = rx_names.index(link[1])
	A[link_num, tx_num] = 1
	A[link_num, len(rx_names)+tx_num] = 1
	A[link_num, rx_num] = -1
	link_num += 1

A_old = A
A = np.delete(A, 0, axis=1)

if DEBUG:
	print(" ----- A matrix ----- ")
	print('Rank (A): ', np.linalg.matrix_rank(A))
	print('Shape (A): ', A.shape)

# Finding Peusdo-inverse of A
pinvA = np.linalg.pinv(A)

# Make Delta Vectors, Make SNR Vectors, Correct Estimate Errors, Create Estimate Vectors, Printing Results, Least Square Error, Plotting SNR Error, Printing Error Results
RMSEs = []
weighted = int(input("Invoke the weighted least squares error method? [0/1]: "))
plt_cnt = 0
lgr_ms = []
smr_ms = []
for r in range(1,rxrepeat+1):
	# print('\nIteration ' + str(r) + ' for all links:')
	# print('\nIteration ' + str(r) + ':')
	print('')
	delta_1 = make_delta(r,links)
	snr_1 = make_snr_vecs(r,links)
	snr_1 = snr_1 / (sum(snr_1))

	# print(' ----- Estimate values corrected from wrap-around ----- ')
	delta_1 = correct_difference(r,delta_1,rx_names,off)

	e_est_1, T_est_1, estimate_1 = find_e_vector(delta_1,A,pinvA,rx_names,snr_1,weighted)

	lgr_ms, smr_ms = print_results_us(r, rx_names, e_est_1, samp_rate, lgr_ms, smr_ms)

	if DEBUG and PLOTS:
		RMSE_1,error_1 = least_sq_error(r,estimate_1,delta_1,A,links,True,samp_rate)
		inv_snr_1 = plot_snr_error(snr_1,error_1,r,samp_rate)
	elif DEBUG and not PLOTS:
		if plt_cnt < 2:
			RMSE_1,error_1 = least_sq_error(r,estimate_1,delta_1,A,links,True,samp_rate)
			inv_snr_1 = plot_snr_error(snr_1,error_1,r,samp_rate)
		else:
			RMSE_1,error_1 = least_sq_error(r,estimate_1,delta_1,A,links,False,samp_rate)
	else:
		RMSE_1,error_1 = least_sq_error(r,estimate_1,delta_1,A,links,False,samp_rate)
	RMSE_1 = round_string(RMSE_1)
	RMSEs.append(RMSE_1)
	plt_cnt += 1

if len(lgr_ms) >= 2*len(smr_ms):
	print(Fore.RED + '\nNot Synchronized.\n' + Style.RESET_ALL + 'Across all iterations, this experiment has majority results on the order of milliseconds.\nOffset results on the order of 1000s of microseconds indicate a non synchronized network. Delays this large, on the order of milliseconds, show the experiment nodes\' local clocks are significantly offset from one another.\n')
elif len(smr_ms) >= 2*len(lgr_ms):
	print(Fore.RED + '\nSynchronized.\n' + Style.RESET_ALL + 'Across all iterations, this experiment has majority results on the order of microseconds.\nOffset results on the order of 10s-100s of microseconds indicate a synchronized network. However, results should not be expected to be much less than ' + str(samples_to_us(1,samp_rate)) + 'us (1/samp_rate).\n')
else:
	print(Fore.RED + '\nInconclusive.\n' + Style.RESET_ALL + 'There is no clear majority within the results as to whether the nodes are synchronized or not. The results are showing both synchronized offsets on the order of microseconds and non-synchronized offsets on the order of milliseconds.\nIt is recommeded to power-cycle the nodes and run the experiment again.\n')

# RMSE_ratio = min(RMSEs)/max(RMSEs)
if DEBUG:
	print(Fore.RED + '\nRoot Mean Squared Error (RMSE) across all links for each Iteration' + Style.RESET_ALL)
	for rme in range(1,len(RMSEs)+1):
		print('Iteration ' + str(rme) + ': ' + format(samples_to_us(RMSEs[rme-1],samp_rate),'.4f') + ' us')
# print('\nRMSE ratio between iterations:', RMSE_ratio)

if DEBUG:
	for i in range(len(RMSEs)):
		RMSEs[i] = samples_to_us(RMSEs[i],samp_rate)
	
	plt.figure()
	plt.scatter(np.array(list(range(1,len(RMSEs)+1))),np.array(RMSEs))
	plt.grid()
	plt.xlabel('Iteration Number')
	plt.ylabel('RMSE (us)')
	plt.title('Scatter Plot of RMSE in us by Iteration')
	plt.show()

	plt.figure()
	plt.bar(np.array([str(i) for i in range(1,len(RMSEs)+1)]),np.array(RMSEs))
	plt.xlabel('Iteration Number')
	plt.ylabel('RMSE (us)')
	plt.title('Bar Chart of RMSE in us by Iteration')
	plt.show()

## Program END ##
print('\n---------- WATCH PROGRAM END ----------\n\n\n\n\n')
