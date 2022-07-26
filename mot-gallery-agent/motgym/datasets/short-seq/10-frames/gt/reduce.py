import csv

out = open('gt.txt', 'w')

with open('gt_og.txt') as f:
	reader = csv.reader(f)
	writer = csv.writer(out)
	for row in reader:
		if int(row[0]) <= 10:
			writer.writerow(row)

out.close()


