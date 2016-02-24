
def trimBedFile(inFile, outfile, seqLength):
    halfLength = seqLength // 2
    resultLines = []
    count = 0
    with open(inFile,'r') as inputFile:
        for line in inputFile.readlines():
            elems = line.split('\t')
            start = int(elems[1])
            end = int(elems[2])
            mean = start + (end - start) // 2
            newStart = mean - halfLength
            newEnd = mean + halfLength
            assert newEnd - newStart == seqLength
            print "New start: " + str(newStart) + " New End: " + str(newEnd)
            # construct new line
            elems[1] = str(newStart)
            elems[2] = str(newEnd)
            line = "\t".join(elems)
            resultLines.append(line)
            #print line
            count += 1
    
    with open(outfile, 'w') as out:
        out.writelines(resultLines)
            

trimBedFile('../../data/narrowPeak/wgEncodeAwgTfbsSydhImr90MafkIggrabUniPk.narrowPeak', '../../data/narrowPeak/stemcells2_preprocessed.narrowPeak', 200)
