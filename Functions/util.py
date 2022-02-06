# FUNCTIONS
def bmiFunc(weight, height):
    if height == 0:
        return 0
    else:
        return round(weight/(height*height/10000),1)


def calculateBMI(df):
    df['bmi'] = df.apply(lambda row: bmiFunc(row['admissionweight'], row['admissionheight']), axis=1)


def calculateGCS(df):
    df['GCS'] = df.apply(lambda row: row['motor'] + row['eyes'] + row['verbal'], axis=1)


def calculatePFRatio(df):
    df['pfratio'] = df.apply(lambda row: row['pao2']/row['fio2'], axis=1)



def getCategory(l):
    if l[1] == 'Elective' or l[1] == 'Was the patient admitted from the O.R. or went to the O.R. within 4 hours of admission?':
        return l[1] + '|' + l[2]

    if l[1] == 'All Diagnosis':
        if l[2] == 'Non-operative' or l[2] == 'Operative':
            return l[2] + '|' + l[4]
        else:
            return -1

    if l[1] == 'Non-operative Organ Systems':
        return l[1] + '|' + l[3]

    if l[1] == 'Operative Organ Systems':
        return l[1] + '|' + l[3]

    if l[1] == 'Additional Apache Information':
        return 'APACHE'

    return l[1]

def nonOperative(l):
    l = l.split('|')
    if l[0] == 'Non-operative Organ Systems':
            return l[1]
    return None


def operative(l):
    l = l.split('|')
    if l[0] == 'Operative Organ Systems':
            return l[1]
    return None
