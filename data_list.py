import os


data_dir = '/home/aistudio/data/'

trainA_data = os.listdir(data_dir + 'trainA')
trainA_data = [x for x in trainA_data if not x.startswith('.')]
print(len(trainA_data))

trainB_data = os.listdir(data_dir + 'trainB')
trainB_data = [x for x in trainB_data if not x.startswith('.')]
print(len(trainB_data))

testA_data = os.listdir(data_dir + 'testA')
testA_data = [x for x in testA_data if not x.startswith('.')]
print(len(testA_data))

testB_data = os.listdir(data_dir + 'testB')
testB_data = [x for x in testB_data if not x.startswith('.')]
print(len(testB_data))

f = open('data/trainA.txt', 'w')
for line in trainA_data:
    f.write(data_dir + 'trainA/' + line + '\n')
f.close()

f = open('data/trainB.txt', 'w')
for line in trainB_data:
    f.write(data_dir + 'trainB/' + line +'\n')
f.close()
f = open('data/testA.txt', 'w')
for line in testA_data:
    f.write(data_dir + 'testA/' + line +'\n')
f.close()
f = open('data/testB.txt', 'w')
for line in testB_data:
    f.write(data_dir + 'testB/' + line +'\n')
f.close()