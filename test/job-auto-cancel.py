import os
import subprocess
import sys

# yhqRe=os.popen("yhq --name=batch-alltoall.sh").readlines()

def ecancel(cab):
    if cab != None:
        yhqRe=os.popen("yhqueue -M " + cab).readlines()
    else:
        yhqRe=os.popen("yhqueue").readlines()

    # print(yhqRe)
    for yhqr in yhqRe:
        strs=(yhqr.strip().split())
        print(strs)
        for i in range(0,len(strs)-2):
            if strs[i+2].find("batch") != -1:
                jobid = strs[i]
                # print(jobid,"yhcancel -M " + cab + " "+jobid)
                if cab != None:
                    os.system("yhcancel -M " + cab + " "+jobid)
                else:
                    os.system("yhcancel "+jobid)
def main(argv):
    if(len(argv) > 1):
        cab=argv[1]
    else:
        cab=None
    ecancel(cab)

if __name__ == '__main__':
    main(sys.argv)

# yhqRe=os.popen("yhq --name=batch-allreduce.sh").readlines()
# # strs=(yhqRe.strip().split())
# print(yhqRe)
# # jobid = strs[8]
# # os.system("yhcancel "+jobid)