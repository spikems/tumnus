#coding:utf-8
import xlrd
#from optparse import OptionParse
import sys 

def trim(line):
  line =line.replace('\r','').replace('\n','').replace('\t','').replace(' ','').lower().encode('utf-8')
  return line 

fname = sys.argv[1]
workbook = xlrd.open_workbook(fname)
outfile ='txtfoutfile'
texf=open(outfile,'wb')
infile = workbook.sheet_by_index(0)
#ncols=infile.ncols()
nrows=infile.nrows
sheet = []
for rowid in range(1,nrows):
  arow = []
  for gid  in [4,1,2,0,8]:
    cell_obj=infile.cell_value(rowid,gid)
    if type(cell_obj)  not in [str,unicode]:       
      arow.append(str(int(cell_obj)))
    else:
      arow.append(cell_obj)
  label = str(arow[0])
  title = trim(arow[1])
  abstract = trim(arow[2])
  id = int(arow[3])
  post_time = [4]
  texf.write('%s %s %s %s %s\n'%(label,title,abstract,post_time,id))
texf.close()




