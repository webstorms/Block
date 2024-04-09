from pathlib import Path
from tables import *
from numpy import *
import os
import sys
sys.path.append("..")
from os.path import join as pjoin


# Credits go to npvoid as I have taken and modified this script from their repo https://github.com/npvoid/SparseSpikingBackprop.


def read2Dspikes(filename):
    '''
    Reads two dimensional binary spike file and returns a TD event.
    It is the same format used in neuromorphic datasets NMNIST & NCALTECH101.
    The binary file is encoded as follows:
        * Each spike event is represented by a 40 bit number.
        * First 8 bits (bits 39-32) represent the xID of the neuron.
        * Next 8 bits (bits 31-24) represent the yID of the neuron.
        * Bit 23 represents the sign of spike event: 0=>OFF event, 1=>ON event.
        * The last 23 bits (bits 22-0) represent the spike event timestamp in microseconds.
    Arguments:
        * ``filename`` (``string``): path to the binary file.
    Usage:
    '''
    with open(filename, 'rb') as inputFile:
        inputByteArray = inputFile.read()
    inputAsInt = asarray([x for x in inputByteArray])
    xEvent = inputAsInt[0::5]
    yEvent = inputAsInt[1::5]
    pEvent = inputAsInt[2::5] >> 7
    tEvent = ((inputAsInt[2::5] << 16) | (inputAsInt[3::5] << 8) | (inputAsInt[4::5])) & 0x7FFFFF
    return tEvent, xEvent, yEvent, pEvent


def convert(prms):
    h5file = open_file(prms['output_filename'], mode="w", title="Training dataset")
    spikes = h5file.create_group(h5file.root, "spikes", "Spikes in eventbased form")
    units = h5file.create_vlarray(spikes, 'units', Int32Atom(shape=()), "Spike neuron IDs, int")
    times = h5file.create_vlarray(spikes, 'times', Float32Atom(shape=()), "Spike Times (in seconds), float")

    labels = []

    data_count = 0
    for dirnames in os.listdir(prms['data_dir']):
        if int(dirnames) in prms['class_list']:
            print(dirnames)
            for f in os.listdir(pjoin(prms['data_dir'], dirnames)):
                timestamps, xaddr, yaddr, pol = read2Dspikes(pjoin(prms['data_dir'], dirnames, f))
                timestamps = timestamps * 1e-6
                times.append(array(timestamps))
                units.append(array(xaddr + yaddr * 34 + pol * 34 * 34))
                labels.append(int(dirnames))
                data_count = data_count + 1

    shape = (data_count,)
    labels_array = h5file.create_carray(h5file.root, 'labels', Int32Atom(), shape=shape, title="Data Labels")
    labels_array[:] = array(labels)

    h5file.close()


if __name__ == "__main__":

    # Convert Train
    prms = {'data_dir': os.path.join(Path(__file__).parent.parent, "data/N-MNIST/Train"),
            'output_filename': os.path.join(Path(__file__).parent.parent, "data/N-MNIST/train.h5"),
            'class_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

    convert(prms)

    prms = {'data_dir': os.path.join(Path(__file__).parent.parent, "data/N-MNIST/Test"),
            'output_filename': os.path.join(Path(__file__).parent.parent, "data/N-MNIST/test.h5"),
            'class_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

    convert(prms)
