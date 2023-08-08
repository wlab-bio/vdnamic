import masterProcesses
import summaryAnalysis
import sysOps
import sys
import optimOps
import multiprocessing as mp
global statuslogfilename
global globaldatapath
global pipeline_command

if __name__ == '__main__':
    mp.set_start_method('forkserver')
    
    sys.argv[len(sys.argv)-1] = sys.argv[len(sys.argv)-1].strip('\r')
    sysOps.initiate_runpath('')
    sysOps.initiate_statusfilename('',make_file = False)
    #sys.argv = sys.argv[1:] #remove first argument (script call)
    sysOps.throw_status('sys.argv = ' + str(sys.argv))
    
    if len(sys.argv)>0 and sys.argv[1][(len(sys.argv[1])-2):] == '//':
        #if first argument is a directory, use this directory as the data directory for all subsequent operations
        sysOps.initiate_runpath(sys.argv[1]) #initiate data run path
        start_arg_index = 2
    else:
        start_arg_index = 1
    
    sysOps.globalmasterProcess = masterProcesses.masterProcess()
    
    if sys.argv[start_arg_index]=='lib':
        sysOps.globalmasterProcess.generate_uxi_library(sys.argv[start_arg_index+1])
    elif sys.argv[start_arg_index]=='image':
        sysOps.globalmasterProcess.dnamic_inference(sys.argv[start_arg_index+1])
    elif sys.argv[start_arg_index]=='gse':
        params = dict()
        arg_ind = start_arg_index+1
        while arg_ind < len(sys.argv):
            if sys.argv[arg_ind].startswith("-") and sys.argv[arg_ind] not in params:
                params[sys.argv[arg_ind]] = list()
                arg_ind += 1
            else:
                break
                
            if arg_ind < len(sys.argv) and not sys.argv[arg_ind].startswith("-"):
                params[sys.argv[arg_ind-1]].append(sys.argv[arg_ind])
                arg_ind += 1
                
        optimOps.run_GSE(output_name = 'GSEoutput.txt',params=params)
    else:
        sysOps.throw_exception('Unrecognized pipeline input: ' + str(sys.argv))

    print("Completed run.")
