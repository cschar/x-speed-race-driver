import sys

def print_to_log():
    # http://stackoverflow.com/questions/2513479/redirect-prints-to-log-file
    old_stdout = sys.stdout

    log_file = open("message.log","w")

    sys.stdout = log_file

    print "this will be written to message.log"

    sys.stdout = old_stdout

    log_file.close()