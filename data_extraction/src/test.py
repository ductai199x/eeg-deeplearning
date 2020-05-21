#!/usr/bin/env python3

import pylab
import numpy
import biosig
HDR=biosig.sopen('/home/sweet/1-workdir/eeg001-2017/S02_ME/motorexecution_subject2_run10.gdf','r');
#for i in range(HDR.NS):
#    HDR.CHANNEL[i].OnOff = 0
#HDR.CHANNEL[0].OnOff = 1
# data = biosig.sread(HDR, HDR.NRec, 0)
# biosig.sclose(HDR)
# # #biosig.destructHDR(HDR)
# pylab.ion()
# pylab.plot(numpy.transpose(data))
# pylab.show()