import csv
import sys
import time
import os
import os.path
import itertools
import masterProcesses
import subprocess
from joblib import Parallel, delayed

statuslogfilename = 'statuslog.csv'
globaldatapath = ''
pipeline_command = ''
num_workers = -1

def sh(cmd_str):
    #throw_status("RUNNING: " + cmd_str)
    return subprocess.run(cmd_str,shell=True,check=True,stdout=subprocess.PIPE,universal_newlines=True).stdout

def big_sort(param_str,infilename,outfilename,path=None,splitfile_size=500000,parallel=False):
    # assumes globaldatapath not pre-appended
    if path is None:
        path = globaldatapath
    if not check_file_exists(infilename,path):
        throw_status('Sort failed. Could not find ' + path + infilename + '.')
        exitProgram()
    
    throw_status('Beginning sort ' + path + infilename + ' --> ' + path + outfilename + '.')
    if not os.path.isdir(path + 'tmp'):
        os.mkdir(path + 'tmp')
    sh("split -l " + str(splitfile_size) + " " + path + infilename + " " + path + "splitfile-")
    [subdirnames, filenames] = get_directory_and_file_list(path)
    filenames = [filename for filename in filenames if filename.startswith('splitfile')]
    if parallel:
        tmpdirs = [path + 'tmp_' + filename + '//' for filename in filenames]
        sh("mkdir " + " ".join(tmpdirs))
        Parallel(n_jobs=-1)(delayed(sh)(f"sort -T {tmpdir} {param_str} {path}{filename} > {path}sorted_{filename} && rm {path}{filename} && rm -r {tmpdir}") for filename,tmpdir in zip(filenames,tmpdirs))
    else:
        for filename in filenames:
            sh("sort -T " + path + "tmp " + param_str + " " + path + filename + " > " + path + "sorted_" + filename + " && rm " + path + filename)
        
    sh("sort -T " + path + "tmp -m " + param_str + " " + path + "sorted_splitfile-* > " + path + outfilename)
    sh("rm " + path + "sorted_splitfile-*")
    
    if not check_file_exists(outfilename,path):
        throw_status('Sort failed. Exiting.')
        exitProgram()

    return
    
            
def exitProgram():
    #add_nodes_running(-1,0,True)
    throw_status("PROGRAM ENDED.")
    sys.exit()
    
def check_file_exists(filename,path=None):
    if path is None:
        path = globaldatapath
    return os.path.isfile(path + filename)
            
def throw_exception(this_input,path=None):
    #throws exception this_input[0] to file-name this_input[1], if this_input[1] exists, or errorlog.csv otherwise
    
    if path is None:
        path =globaldatapath
    
    if(type(this_input)==list and len(this_input)==2):
        statusphrase = this_input[0]
        statuslog_filename = this_input[1]
    else:
        if(type(this_input)==list):
            statusphrase = this_input[0]
        else:
            statusphrase = this_input
        statuslog_filename = path + "errorlog.csv"

    my_datetime = time.strftime("%Y/%m/%d %H:%M:%S")
    with open(statuslog_filename,'a+') as csvfile:
        csvfile.write(my_datetime + '|' + statusphrase + '\n')

def throw_status(this_input, path = None):
    #throws status this_input[0] to file-name this_input[1], if this_input[1] exists, or statuslog.csv otherwise
    #if this_input[1] is global variable statuslogfilename, globaldatapath will already be incorporated to beginning of string, and therefore it is not included in call to file-open function
    
    if path is None:
        path = globaldatapath
    
    if(type(this_input)==list and len(this_input)==2):
        statusphrase = this_input[0]
        statuslog_filename = this_input[1]
    else:
        if(type(this_input)==list):
            statusphrase = this_input[0]
        else:
            statusphrase = this_input
        statuslog_filename = path + "statuslog.csv"

    my_datetime = time.strftime("%Y/%m/%d %H:%M:%S")
    with open(statuslog_filename,'a+') as csvfile:
        csvfile.write(my_datetime + '|' + statusphrase + '\n')

    print(my_datetime + "|" + statusphrase)
    
def get_directory_and_file_list(path = None):
    if path is None:
        path = globaldatapath + "."
    elif path == "":
        path = "."
    else:
        path = path + "."
        
    while True:
        try:
            for dirname, dirnames, filenames in os.walk(path):
                return [dirnames,filenames] #first level of directory hierarchy only
            return [list(),list()]
        except:
            print('Error during file/directory-readout. Re-trying.')

def initiate_statusfilename(prefix = '',make_file = False):
    #globaldatarunpath added directly to statuslogfilename
    global statuslogfilename
    fullprefix = prefix + 'statuslog'
    max_statuslog_index = 0
    [dirnames, filenames] = get_directory_and_file_list()
    for filename in filenames:
        if filename.startswith(fullprefix) and filename.endswith('.csv'):
            try:
                max_statuslog_index = max(max_statuslog_index, int(filename[len(fullprefix):(len(filename)-4)]))
            except: #no integer-form index substring
                pass
            
    statuslogfilename = globaldatapath + fullprefix + str(max_statuslog_index + 1) + ".csv"
    
    if make_file:
        status_outfile = open(statuslogfilename,'w')
        status_outfile.close()
    return
    
def initiate_runpath(mydatapath, autoinitialize_statusfilename=True):
    global globaldatapath
    globaldatapath = mydatapath
    print('init=' + mydatapath)
    
    if autoinitialize_statusfilename:
        initiate_statusfilename()
        
    try:
        os.mkdir(globaldatapath + "tmp")
    except:
        pass
    return
    
def delay_with_alertfile(alertfile):
    #delays until alertfile is removed from directory, at which point the alertfile is replaced and process continues
    while True:
        try:
            alertfile_handle = open(globaldatapath + alertfile,'rU')
            alertfile_handle.close()
            time.sleep(1)
        except:
            with open(globaldatapath + alertfile,'w') as alertfile_handle:
                alertfile_handle.write('1')
            break
    return
    
def remove_alertfile(alertfile):
    os.remove(globaldatapath + alertfile)
    return
