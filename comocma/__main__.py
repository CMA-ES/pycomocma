import doctest
import comocma

doctest.ELLIPSIS_MARKER = '***' # to be able to ignore an entire output, 
    # putting the default '...' doesn't work for that.
print('doctesting `comocma`')
print(doctest.testmod(comocma.como))
