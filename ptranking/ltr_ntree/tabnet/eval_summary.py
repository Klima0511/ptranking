#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Haitao Yu on 05/07/2018

"""Description

"""

import os
from operator import itemgetter

def load_results_direct(model_grid_dir=None, dataset=None, height=1, do_sort=True, ex_key='validation', name_key=None):
	assert model_grid_dir is not None and dataset is not None

	list_run_name = []
	list_run_log_file = []
	for file in os.listdir(model_grid_dir):
		if 1 == height or 2 == height:
			if file.find(dataset) >= 0:
				if 1 == height:
					run_dir = os.path.join(model_grid_dir, file)
					if os.path.isdir(run_dir):
						list_run_name.append(file)
						#list_run_log_file.append(run_dir + '/log.txt')
						for log_file in os.listdir(run_dir):
							if log_file.startswith('log'):
								list_run_log_file.append(os.path.join(run_dir, log_file))

				elif 2 == height:
					run_prefix = file
					h1_dir = os.path.join(model_grid_dir, file)
					if os.path.isdir(h1_dir):
						for h1_sub in os.listdir(h1_dir):
							log_dir = os.path.join(h1_dir, h1_sub)
							if os.path.isdir(log_dir):
								#list_run_log_file.append(log_dir + '/log.txt')
								have_log_file = False
								for log_file in os.listdir(log_dir):
									if log_file.startswith('log'):
										list_run_log_file.append(os.path.join(log_dir, log_file))
										#print('log_file', log_file)
										have_log_file = True
								if have_log_file:
									run_name = '_'.join([run_prefix, h1_sub])
									list_run_name.append(run_name)
									#print('run_name', run_name, '\n')
		elif 3 == height:
			run_prefix = file
			h1_dir = os.path.join(model_grid_dir, file)
			if os.path.isdir(h1_dir):
				for h1_sub in os.listdir(h1_dir):
					if h1_sub.find(dataset) >= 0:
						h2_dir = os.path.join(h1_dir, h1_sub)
						for h2_sub in os.listdir(h2_dir):
							log_dir = os.path.join(h2_dir, h2_sub)
							if os.path.isdir(log_dir):
								#list_run_log_file.append(log_dir + '/log.txt')
								have_log_file = False
								for log_file in os.listdir(log_dir):
									if log_file.startswith('log'):
										list_run_log_file.append(os.path.join(log_dir, log_file))
										have_log_file = True
								if have_log_file:
									run_name = '_'.join([run_prefix, h1_sub, h2_sub])
									list_run_name.append(run_name)


	#print(list_run_name)
	#print(list_run_log_file)

	list_runs = []
	#print('list_run_name', len(list_run_name), list_run_name)
	#print('list_run_log_file', len(list_run_log_file), list_run_log_file)
	for i in range(len(list_run_name)):
		run_name = list_run_name[i]

		if name_key is not None:
			if not run_name.find(name_key) > 0:
				continue

		run_log_file = list_run_log_file[i]
		with open(run_log_file) as log_reader:
			lines = log_reader.readlines()
			for i in range(0, len(lines)):
				line = lines[i]
				if line.find(ex_key) > 0:
					nDCG1 = parse_line_direct(lines[i+1])
					list_runs.append((nDCG1, line.strip(), run_name))
					break
				else:
					continue

	if do_sort:
		sorted_runs = sorted(list_runs, key=itemgetter(0), reverse=True)
	else:
		sorted_runs = list_runs

	for tp in sorted_runs:
		print(tp[0], '\t', tp[1], '\t', tp[2])


def load_mul_results_direct(model_grid_dir=None, dataset=None, height=1, do_sort=(True, 1), ex_key='validation', name_key=None):
	assert model_grid_dir is not None and dataset is not None

	list_run_name = []
	list_run_log_file = []
	for file in os.listdir(model_grid_dir):
		if 1 == height or 2 == height:
			if file.find(dataset) >= 0:
				if 1 == height:
					run_dir = os.path.join(model_grid_dir, file)
					if os.path.isdir(run_dir):
						list_run_name.append(file)
						#list_run_log_file.append(run_dir + '/log.txt')
						for log_file in os.listdir(run_dir):
							if log_file.startswith('log'):
								list_run_log_file.append(os.path.join(run_dir, log_file))

				elif 2 == height:
					run_prefix = file
					h1_dir = os.path.join(model_grid_dir, file)
					if os.path.isdir(h1_dir):
						for h1_sub in os.listdir(h1_dir):
							log_dir = os.path.join(h1_dir, h1_sub)
							if os.path.isdir(log_dir):
								#list_run_log_file.append(log_dir + '/log.txt')
								have_log_file = False
								for log_file in os.listdir(log_dir):
									if log_file.startswith('log'):
										list_run_log_file.append(os.path.join(log_dir, log_file))
										#print('log_file', log_file)
										have_log_file = True
								if have_log_file:
									run_name = '_'.join([run_prefix, h1_sub])
									list_run_name.append(run_name)
									#print('run_name', run_name, '\n')
		elif 3 == height:
			run_prefix = file
			h1_dir = os.path.join(model_grid_dir, file)
			if os.path.isdir(h1_dir):
				for h1_sub in os.listdir(h1_dir):
					if h1_sub.find(dataset) >= 0:
						h2_dir = os.path.join(h1_dir, h1_sub)
						for h2_sub in os.listdir(h2_dir):
							log_dir = os.path.join(h2_dir, h2_sub)
							if os.path.isdir(log_dir):
								#list_run_log_file.append(log_dir + '/log.txt')
								have_log_file = False
								for log_file in os.listdir(log_dir):
									if log_file.startswith('log'):
										list_run_log_file.append(os.path.join(log_dir, log_file))
										have_log_file = True
								if have_log_file:
									run_name = '_'.join([run_prefix, h1_sub, h2_sub])
									list_run_name.append(run_name)


	#print(list_run_name)
	#print(list_run_log_file)
	bool_sort, top_k = do_sort

	list_runs = []
	#print('list_run_name', len(list_run_name), list_run_name)
	#print('list_run_log_file', len(list_run_log_file), list_run_log_file)
	for i in range(len(list_run_name)):
		run_name = list_run_name[i]

		if name_key is not None:
			if not run_name.find(name_key) > 0:
				continue

		run_log_file = list_run_log_file[i]

		num_metrics = 4
		with open(run_log_file) as log_reader:
			is_performance_line = False
			metric_at_k, i = None, 0
			list_performance_txt = []
			for line in log_reader.readlines():
				if is_performance_line and i < num_metrics:
					list_performance_txt.append(str.strip(line))
					i += 1
					if 1 == i:
						nDCG1 = parse_line_direct(line, top_k=top_k)
					elif num_metrics == i:
						list_runs.append((nDCG1, '\n'.join(list_performance_txt), run_name))
				elif i >= num_metrics:
					break
				elif line.find(ex_key) > 0:
					is_performance_line = True
				else:
					continue

	if bool_sort:
		sorted_runs = sorted(list_runs, key=itemgetter(0), reverse=True)
	else:
		sorted_runs = list_runs

	for tp in sorted_runs:
		print("\n{}\t{}\t{}\n{}".format(tp[0], "@" + str(top_k), tp[2], tp[1]))
	# print('\n', tp[0], '\n', tp[1], '\t', tp[2])

def load_results_direct_h1(model_grid_dir=None, dataset=None):
	assert model_grid_dir is not None and dataset is not None

	list_run_name = []
	list_run_log_file = []
	for file in os.listdir(model_grid_dir):
		if file.find(dataset) > 0:
			run_dir = os.path.join(model_grid_dir, file)
			if os.path.isdir(run_dir):
				list_run_name.append(file)
				list_run_log_file.append(run_dir + '/log.txt')

	#print(list_run_name)
	#print(list_run_log_file)

	list_runs = []
	for i in range(len(list_run_name)):
		run_name = list_run_name[i]
		run_log_file = list_run_log_file[i]
		with open(run_log_file) as log_reader:
			for line in log_reader.readlines():
				if line.find('validation') > 0:
					nDCG1 = parse_line_direct(line)
					list_runs.append((nDCG1, line.strip(), run_name))
					break
				else:
					continue
	#
	sorted_runs = sorted(list_runs, key=itemgetter(0), reverse=True)
	for tp in sorted_runs:
		print(tp[0], '\t', tp[1], '\t', tp[2])

def parse_line_direct(line=None, top_k=1):
	#nDCG1_start = line.find('@') +3
	nDCG1_start = line.find('@'+str(top_k) ) + 3
	nDCG1_end = line.find(',')
	nDCG_val = float(line[nDCG1_start:nDCG1_end])
	return nDCG_val


def cmp_performance_sample_1_MSLRWEB10K():
	##  ##
	dataset = 'MSLRWEB10K'

	# 1 grid_ListApxNDCG
	model_grid_dir = '/data/tan_haonan/Output/MSLR-WEB10K/gpu_grid_TabNet1/'
	load_results_direct(model_grid_dir=model_grid_dir, dataset=dataset, height=2)
	'''
	0.4622 	 ListApxNDCG 5-fold cross validation scores: nDCG@1:0.4622, nDCG@3:0.4419, nDCG@5:0.4455, nDCG@10:0.4600 	 ListApxNDCG_MSLRWEB10K_Hi_3_Af_RRS_Ep_100_St_1_Vd_True_Md_10_Mr_1_Alpha_50
	0.1987 	 ListApxNDCG 5-fold cross validation scores: nDCG@1:0.1987, nDCG@3:0.2038, nDCG@5:0.2163, nDCG@10:0.2438 	 ListApxNDCG_MSLRWEB10K_Hi_3_Af_RRS_Ep_100_St_1_Vd_True_Md_10_Mr_1_Alpha_100
	0.1389 	 ListApxNDCG 5-fold cross validation scores: nDCG@1:0.1389, nDCG@3:0.1506, nDCG@5:0.1650, nDCG@10:0.1952 	 ListApxNDCG_MSLRWEB10K_Hi_3_Af_RRS_Ep_100_St_1_Vd_True_Md_10_Mr_1_Alpha_150
	0.1389 	 ListApxNDCG 5-fold cross validation scores: nDCG@1:0.1389, nDCG@3:0.1506, nDCG@5:0.1650, nDCG@10:0.1952 	 ListApxNDCG_MSLRWEB10K_Hi_3_Af_RRS_Ep_100_St_1_Vd_True_Md_10_Mr_1_Alpha_200
	'''

	# 2 listNet
	# model_grid_dir = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/2019WSDM/sample_1/0725_grid_ListNet/'
	# load_results_direct(model_grid_dir=model_grid_dir, dataset=dataset, height=1)
	'''
	0.4549 	 ListNet 5-fold cross validation scores: nDCG@1:0.4549, nDCG@3:0.4384, nDCG@5:0.4422, nDCG@10:0.4598 	 ListNet_MSLRWEB10K_Hi_3_Af_RRS_Ep_100_St_1_Vd_True_Md_10_Mr_1
	'''

	# 3 listMLE
	# model_grid_dir = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/2019WSDM/sample_1/0725_grid_ListMLE/'
	# load_results_direct(model_grid_dir=model_grid_dir, dataset=dataset, height=1)

def cmp_performance_sample_1_MQ2008():
	##  ##
	dataset = 'MQ2008_Super'

	# 1 grid_ListApxNDCG
	model_grid_dir = '/data/tan_haonan/Output/MSLR-WEB10K/gpu_grid_TabNet/'
	load_results_direct(model_grid_dir=model_grid_dir, dataset=dataset, height=2)
	'''
	0.4622 	 ListApxNDCG 5-fold cross validation scores: nDCG@1:0.4622, nDCG@3:0.4419, nDCG@5:0.4455, nDCG@10:0.4600 	 ListApxNDCG_MSLRWEB10K_Hi_3_Af_RRS_Ep_100_St_1_Vd_True_Md_10_Mr_1_Alpha_50
	0.1987 	 ListApxNDCG 5-fold cross validation scores: nDCG@1:0.1987, nDCG@3:0.2038, nDCG@5:0.2163, nDCG@10:0.2438 	 ListApxNDCG_MSLRWEB10K_Hi_3_Af_RRS_Ep_100_St_1_Vd_True_Md_10_Mr_1_Alpha_100
	0.1389 	 ListApxNDCG 5-fold cross validation scores: nDCG@1:0.1389, nDCG@3:0.1506, nDCG@5:0.1650, nDCG@10:0.1952 	 ListApxNDCG_MSLRWEB10K_Hi_3_Af_RRS_Ep_100_St_1_Vd_True_Md_10_Mr_1_Alpha_150
	0.1389 	 ListApxNDCG 5-fold cross validation scores: nDCG@1:0.1389, nDCG@3:0.1506, nDCG@5:0.1650, nDCG@10:0.1952 	 ListApxNDCG_MSLRWEB10K_Hi_3_Af_RRS_Ep_100_St_1_Vd_True_Md_10_Mr_1_Alpha_200
	'''

	# 2 listNet
	# model_grid_dir = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/2019WSDM/sample_1/0725_grid_ListNet/'
	# load_results_direct(model_grid_dir=model_grid_dir, dataset=dataset, height=1)
	'''
	0.4549 	 ListNet 5-fold cross validation scores: nDCG@1:0.4549, nDCG@3:0.4384, nDCG@5:0.4422, nDCG@10:0.4598 	 ListNet_MSLRWEB10K_Hi_3_Af_RRS_Ep_100_St_1_Vd_True_Md_10_Mr_1
	'''

	# 3 listMLE
	# model_grid_dir = '/Users/dryuhaitao/WorkBench/CodeBench/Bench_Output/NeuralLTR/Listwise/2019WSDM/sample_1/0725_grid_ListMLE/'
	# load_results_direct(model_grid_dir=model_grid_dir, dataset=dataset, height=1)

if __name__ == '__main__':
	#1
	cmp_performance_sample_1_MSLRWEB10K()
