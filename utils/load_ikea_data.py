import os
import shutil
allfilelist = os.listdir('../data/IKEA')
allfilelist.sort()
allfilelist = allfilelist[2:-2]
num = 0
cate_last = allfilelist[0].split('_')[1]
for i in range (len(allfilelist)):
    data = []
    for line in open('../data/IKEA/'+allfilelist[i]+'/obj_list.txt',"r"): 
        fname = '../data/IKEA/'+allfilelist[i]+'/'+line[:-1]+'.obj'
        data.append(fname)
    cate = allfilelist[i].split('_')[1]
    if (cate != cate_last):
        num = 0
    cate_last = cate
    for j in range (len(data)):
        print 'loading data '+str(cate)+str(num)
        shutil.copy(data[j],'../data/obj_data/'+str(cate)+str(num)+'.obj')
        num += 1
