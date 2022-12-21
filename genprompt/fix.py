with open('zero-de.results') as f, open('fixed', 'w+') as out_f:
	count = 0
	for line in f.readlines():
		count += 1
		if line == '':
			print('here')
		out_f.write(line.strip()+ '\n')
		print(count)
